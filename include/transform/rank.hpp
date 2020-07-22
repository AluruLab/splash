/*
 *  rank.hpp
 *
 *  Created on: June 11, 2020
 *  Author: Tony Pan
 *  Affiliation: Institute for Data Engineering and Science
 *  			Georgia Institute of Technology, Atlanta, GA 30332
 */

#pragma once

#include <algorithm>   // stable_sort
#include <utility>     // pair
#include <type_traits> // std::conditional

#include "utils/memory.hpp"
#include "kernel/kernel_base.hpp"

#if defined(USE_SIMD)
#include <omp.h>
#endif

namespace splash { namespace kernel { 
/* TODO:
 * [ ] make buffer threadsafe
 */
template <typename IT>
class Sort : public splash::kernel::buffered_kernel<std::pair<IT, size_t>> {
	protected:
		using base_type = splash::kernel::buffered_kernel<std::pair<IT, size_t>>;
		using PairType = std::pair<IT, size_t>;

	public:

		inline void sort(IT const * in_vec, size_t const & count) const {
			// fprintf(stdout, "thread %d, in %p, buffer %p, count %lu\n", omp_get_thread_num(), in_vec, this->buffer, count);

			this->resize(count);  // ensure sufficient space.

			// fprintf(stdout, "thread %d, in %p, resized %p, count %lu\n", omp_get_thread_num(), in_vec, this->buffer, count);
			// fflush(stdout);
			/*get the rank vector*/
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#if defined(USE_SIMD)
#pragma omp simd
#endif
			for(size_t j = 0; j < count; ++j){
				this->buffer[j].first = in_vec[j];
				this->buffer[j].second = j;
			}

			PRINT_RT("SORT: sorting %p, count %lu, thread %d of %d\n", this->buffer, count, omp_get_thread_num(), omp_get_num_threads());
			// sort to get rank.
			std::stable_sort(this->buffer, this->buffer + count, [](PairType const & x, PairType const & y){
				return x.first < y.first;
			});

		}

};


template<typename RT>
struct RankElemType {
	RT pos;  		// position in sorted array.
	RT rank;   		// rank of value in sorted array. 
	//  note: rank can repeat. pos are unique and order follows sorting method (i.e. stable sort or not)

	void print(const char* prefix) const {
		PRINT("%s(%ld %ld)", prefix, pos, rank);
	}
};


template <typename IT, typename RT = IT>
class Rank : public splash::kernel::transform<IT, RT, splash::kernel::DEGREE::VECTOR>, public splash::kernel::Sort<IT> {
    public:
		using InputType = IT;
        using OutputType = RT;
		static_assert(std::is_arithmetic<OutputType>::value, "Rank type must be numeric");

	protected:	
        OutputType firstRank;

		inline void rank(size_t const & count, OutputType * out_vec) const {
			// unsort with rank.  can't vectorize either because of random memory access or because of forward dependency.
			OutputType rank = firstRank;

			for(size_t j = 0; j < count - 1; ++j){
				out_vec[this->buffer[j].second] = rank;
				rank += (this->buffer[j].first != this->buffer[j + 1].first);  // branchless
			}
			out_vec[this->buffer[count-1].second] = rank;
			// would a sort be faster here?  NO.  sort is much more expensive than random memory access.
// 			/*do we need to normalize the out_vec to avoid overflow? NO.  pearson convert to standard score anyway.*/
		}

    public:
		Rank(OutputType const & first = 1) : firstRank(first) {}
		virtual ~Rank() {}

		void copy_parameters(Rank const & other) {
			firstRank = other.firstRank;
		}

        inline virtual void operator()(IT const * in_vec, size_t const & count,
            OutputType * out_vec) const {
			this->sort(in_vec, count);
			this->rank(count, out_vec);
        }
};



template <typename IT, typename RT>
class Rank<IT, RankElemType<RT>> :  public splash::kernel::transform<IT, RankElemType<RT>, splash::kernel::DEGREE::VECTOR>, public splash::kernel::Sort<IT> {
    public:
        using RankType = RT;
		static_assert(std::is_arithmetic<RankType>::value, "Rank type must be numeric");

		using InputType = IT;
	    using OutputType = RankElemType<RT>;

	protected:	
        RankType firstRank;

		inline void rank(size_t const & count, OutputType * out_vec) const {

			// unsort with rank.  can't vectorize either because of random memory access or because of forward dependency.
			RankType rank = firstRank;
			RankType id;
			for(size_t j = 0; j < count - 1; ++j){
				id = this->buffer[j].second;
				out_vec[id].pos = j;
				out_vec[id].rank = rank;
				rank += (this->buffer[j].first != this->buffer[j + 1].first);  // branchless
			}
			id = this->buffer[count - 1L].second;
			out_vec[id].pos = count - 1L;
			out_vec[id].rank = rank;
			
			// would a sort be faster here?  NO.  sort is much more expensive than random memory access.
// 			/*do we need to normalize the out_vec to avoid overflow? NO.  pearson convert to standard score anyway.*/
		}

    public:
		Rank(RankType const & first = 1) :  firstRank(first) {}
		virtual ~Rank() {}

		void copy_parameters(Rank const & other) {
			firstRank = other.firstRank;
		}

        inline virtual void operator()(IT const * in_vec, size_t const & count,
            OutputType * out_vec) const {

			this->sort(in_vec, count);
			this->rank(count, out_vec);
        }
};



}}