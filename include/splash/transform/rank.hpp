/*
 * Copyright 2020 Georgia Tech Research Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 * Author(s): Tony C. Pan
 */

#pragma once

#include <algorithm>   // stable_sort
#include <utility>     // pair
#include <type_traits> // std::conditional

#include "splash/ds/buffer.hpp"
#include "splash/kernel/kernel_base.hpp"

#include "splash/utils/report.hpp"

#if defined(USE_SIMD)
#include <omp.h>
#endif

namespace splash { namespace kernel { 
/* TODO:
 * [ ] make buffer threadsafe
 */
template <typename IT, bool ASCEND = true>
class Sort{
	protected:
		using PairType = std::pair<IT, size_t>;

		mutable splash::ds::buffer<std::pair<IT, size_t>> __buffer;
	public:

		inline void sort(IT const * in_vec, size_t const & count) const {
			// FMT_PRINT_RT("thread {}, in {:p}, buffer {:p}, count {}\n", omp_get_thread_num(), in_vec, __buffer.data, count);

			__buffer.resize(count);  // ensure sufficient space.

			// FMT_PRINT_RT("thread {}, in {:p}, resized {:p}, count {}\n", omp_get_thread_num(), in_vec, __buffer.data, count);
			// fflush(stdout);
			/*get the rank vector*/
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#if defined(USE_SIMD)
#pragma omp simd
#endif
			for(size_t j = 0; j < count; ++j){
				__buffer.data[j].first = in_vec[j];
				__buffer.data[j].second = j;
			}

			// sort to get rank.
			if (ASCEND) {
				std::stable_sort(__buffer.data, __buffer.data + count, [](PairType const & x, PairType const & y){
					return x.first < y.first;
				});
			} else {
				std::stable_sort(__buffer.data, __buffer.data + count, [](PairType const & x, PairType const & y){
					return x.first >= y.first;
				});
			}
		}

};


template<typename RT>
struct RankElemType {
	RT pos;  		// position in sorted array.
	RT rank;   		// rank of value in sorted array. 
	//  note: rank can repeat. pos are unique and order follows sorting method (i.e. stable sort or not)

	void print(const char* prefix) const {
		FMT_PRINT("{}({} {})", prefix, pos, rank);
	}
};


template <typename IT, typename RT = IT, long firstRank = 1, bool ASCEND = true>
class Rank : public splash::kernel::transform<IT, RT, splash::kernel::DEGREE::VECTOR>, public splash::kernel::Sort<IT, ASCEND> {
    public:
		using InputType = IT;
        using OutputType = RT;
		static_assert(std::is_arithmetic<OutputType>::value, "Rank type must be numeric");

	protected:	

		inline void rank(size_t const & count, OutputType * out_vec) const {
			// unsort with rank.  can't vectorize either because of random memory access or because of forward dependency.
			OutputType rank = firstRank;

			for(size_t j = 0; j < count - 1; ++j){
				out_vec[this->__buffer.data[j].second] = rank;
				rank += (this->__buffer.data[j].first != this->__buffer.data[j + 1].first);  // branchless
			}
			out_vec[this->__buffer.data[count-1].second] = rank;
			// would a sort be faster here?  NO.  sort is much more expensive than random memory access.
// 			/*do we need to normalize the out_vec to avoid overflow? NO.  pearson convert to standard score anyway.*/
		}

    public:
		virtual ~Rank() {}

        inline virtual void operator()(IT const * in_vec, size_t const & count,
            OutputType * out_vec) const {
			this->sort(in_vec, count);
			this->rank(count, out_vec);
        }
};



template <typename IT, typename RT, long firstRank, bool ASCEND>
class Rank<IT, RankElemType<RT>, firstRank, ASCEND> :  public splash::kernel::transform<IT, RankElemType<RT>, splash::kernel::DEGREE::VECTOR>, public splash::kernel::Sort<IT, ASCEND> {
    public:
        using RankType = RT;
		static_assert(std::is_arithmetic<RankType>::value, "Rank type must be numeric");

		using InputType = IT;
	    using OutputType = RankElemType<RT>;

	protected:	

		inline void rank(size_t const & count, OutputType * out_vec) const {

			// unsort with rank.  can't vectorize either because of random memory access or because of forward dependency.
			RankType rank = firstRank;
			RankType id;
			for(size_t j = 0; j < count - 1; ++j){
				id = this->__buffer.data[j].second;
				out_vec[id].pos = j;
				out_vec[id].rank = rank;
				rank += (this->__buffer.data[j].first != this->__buffer.data[j + 1].first);  // branchless
			}
			id = this->__buffer.data[count - 1L].second;
			out_vec[id].pos = count - 1L;
			out_vec[id].rank = rank;
			
			// would a sort be faster here?  NO.  sort is much more expensive than random memory access.
// 			/*do we need to normalize the out_vec to avoid overflow? NO.  pearson convert to standard score anyway.*/
		}

    public:
		virtual ~Rank() {}

        inline virtual void operator()(IT const * in_vec, size_t const & count,
            OutputType * out_vec) const {

			this->sort(in_vec, count);
			this->rank(count, out_vec);
        }
};



}}