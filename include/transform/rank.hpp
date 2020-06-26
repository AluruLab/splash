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


template <typename IT, typename OT = IT>
class Rank : public splash::kernel::V2VOp<IT, OT> {
    public:
        using RankType = typename std::conditional<
            (sizeof(IT) == 4),
            int, long
        >::type;
	protected:
		// conditional type so that sizeof(MyPair) is a power of 2.
		using MyPair = std::pair<IT, RankType>;
	
		MyPair* sort_buffer;
		size_t vecSize;
        RankType firstRank;

    public:
        using OutputType = OT;

		Rank(size_t const & _count, RankType const & first = 1) : vecSize(_count), firstRank(first) {
			sort_buffer = reinterpret_cast<MyPair* >(splash::utils::aalloc(_count * sizeof(MyPair)));
		}
		~Rank() {
			splash::utils::afree(sort_buffer);
		}


        inline void operator()(IT const * __restrict__ in_vec, 
            size_t const & count,
            OT * __restrict__ out_vec) const {

			size_t j;
			/*get the rank vector*/
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#if defined(USE_SIMD)
#pragma omp simd
#endif
			for(j = 0; j < count; ++j){
				sort_buffer[j].first = in_vec[j];
				sort_buffer[j].second = j;
			}

			// sort to get rank.
			std::stable_sort(sort_buffer, sort_buffer + count, [](MyPair const & x, MyPair const & y){
				return x.first < y.first;
			});

			// unsort with rank.  can't vectorize either because of random memory access or because of forward dependency.
			RankType rank = firstRank;

			for(j = 0; j < count - 1; ++j){
				out_vec[sort_buffer[j].second] = rank;
				rank += (sort_buffer[j].first != sort_buffer[j + 1].first);  // branchless
			}
			out_vec[sort_buffer[count-1].second] = rank;
			// would a sort be faster here?  NO.  sort is much more expensive than random memory access.
// 			/*do we need to normalize the out_vec to avoid overflow? NO.  pearson convert to standard score anyway.*/

        }
};



}}