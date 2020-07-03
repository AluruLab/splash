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

template <typename IT>
class Sort {
	protected:
		using PairType = std::pair<IT, size_t>;

		mutable PairType* sort_buffer;
		mutable size_t vecSize;

	public:
		Sort(size_t const & _count) : vecSize(_count) {
			sort_buffer = reinterpret_cast<PairType* >(splash::utils::aalloc(_count * sizeof(PairType)));
		}
		~Sort() {
			splash::utils::afree(sort_buffer);
		}

		inline void resize_buffer(size_t const & _count) const {
			if (_count > this->vecSize) {
				splash::utils::afree(sort_buffer);
				sort_buffer = reinterpret_cast<PairType* >(splash::utils::aalloc( _count * sizeof(PairType)));
				this->vecSize = _count;
			}
		}

		inline void sort(IT const * __restrict__ in_vec, size_t const & count) const {
			/*get the rank vector*/
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#if defined(USE_SIMD)
#pragma omp simd
#endif
			for(size_t j = 0; j < count; ++j){
				sort_buffer[j].first = in_vec[j];
				sort_buffer[j].second = j;
			}

			// sort to get rank.
			std::stable_sort(sort_buffer, sort_buffer + count, [](PairType const & x, PairType const & y){
				return x.first < y.first;
			});

		}


};


template<typename RT>
struct RankElemType {
	RT pos;  		// position in sorted array.
	RT rank;   		// rank of value in sorted array. 
	//  note: rank can repeat. pos are unique and order follows sorting method (i.e. stable sort or not)

	void print() const {
		PRINT("%ld", rank);
	}
};


template <typename IT, typename RT = IT>
class Rank : public splash::kernel::V2VOp<IT, RT>, public splash::kernel::Sort<IT> {
    public:
        using OutputType = RT;
		static_assert(std::is_arithmetic<OutputType>::value, "Rank type must be numeric");

	protected:	
        OutputType firstRank;

		inline void rank(size_t const & count, OutputType * __restrict__ out_vec) const {
			// unsort with rank.  can't vectorize either because of random memory access or because of forward dependency.
			OutputType rank = firstRank;

			for(size_t j = 0; j < count - 1; ++j){
				out_vec[this->sort_buffer[j].second] = rank;
				rank += (this->sort_buffer[j].first != this->sort_buffer[j + 1].first);  // branchless
			}
			out_vec[this->sort_buffer[count-1].second] = rank;
			// would a sort be faster here?  NO.  sort is much more expensive than random memory access.
// 			/*do we need to normalize the out_vec to avoid overflow? NO.  pearson convert to standard score anyway.*/

		}

    public:
  
		Rank(size_t const & _count, OutputType const & first = 1) :  splash::kernel::Sort<IT>(_count), firstRank(first) {}
		~Rank() {}

        inline void operator()(IT const * __restrict__ in_vec, size_t const & count,
            OutputType * __restrict__ out_vec) const {

			this->resize_buffer(count);

			this->sort(in_vec, count);
			this->rank(count, out_vec);
        }
};



template <typename IT, typename RT>
class Rank<IT, RankElemType<RT>> :  public splash::kernel::V2VOp<IT, RankElemType<RT>>, public splash::kernel::Sort<IT> {
    public:
        using RankType = RT;
		static_assert(std::is_arithmetic<RankType>::value, "Rank type must be numeric");

	    using OutputType = RankElemType<RT>;

	protected:	
        RankType firstRank;

		inline void rank(size_t const & count, OutputType * __restrict__ out_vec) const {

			// unsort with rank.  can't vectorize either because of random memory access or because of forward dependency.
			RankType rank = firstRank;
			RankType id;
			for(size_t j = 0; j < count - 1; ++j){
				id = this->sort_buffer[j].second;
				out_vec[id].pos = j;
				out_vec[id].rank = rank;
				rank += (this->sort_buffer[j].first != this->sort_buffer[j + 1].first);  // branchless
			}
			id = this->sort_buffer[count - 1L].second;
			out_vec[id].pos = count - 1L;
			out_vec[id].rank = rank;
			
			// would a sort be faster here?  NO.  sort is much more expensive than random memory access.
// 			/*do we need to normalize the out_vec to avoid overflow? NO.  pearson convert to standard score anyway.*/
		}

    public:
  
		Rank(size_t const & _count, RankType const & first = 1) :  splash::kernel::Sort<IT>(_count), firstRank(first) {}
		~Rank() {}

        inline void operator()(IT const * __restrict__ in_vec, size_t const & count,
            OutputType * __restrict__ out_vec) const {

			this->resize_buffer(count);

			this->sort(in_vec, count);
			this->rank(count, out_vec);
        }
};



}}