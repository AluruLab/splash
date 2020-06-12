/*
 *  standard.hpp
 *
 *  Created on: June 11, 2020
 *  Author: Tony Pan
 *  Affiliation: Institute for Data Engineering and Science
 *  			Georgia Institute of Technology, Atlanta, GA 30332
 */

#pragma once

#include <stdlib.h>   // aligned_alloc
#include <algorithm> // stable_sort
#include <utility>  // pair
#include <type_traits> // stD::conditional
#include <cmath>  // sqrt


#include "common.hpp"
#include "kernel/kernel_base.hpp"

#if defined(USE_SIMD)
#include <omp>
#endif

namespace splash { namespace kernel { 


template<typename IT>
class DotProductKernel : public splash::kernel::BinaryVectorOpBase<IT, IT> {

	public:
		inline IT operator()(IT const * first, IT const * second, size_t const & count) {
			IT prod = 0;
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#if defined(USE_SIMD)
#pragma omp simd reduction(+:prod)
#endif
			// MAJORITY OF TIME HERE.  slowdown: O3 + omp simd (1x).  O3 (5x).  sanitizer + omp simd (11x), sanitizer (5.5x) 
			for (int j = 0; j < count; ++j) {
				prod += first[j] * second[j];
			}
			return prod;
		};
};


// for sample standard score.  
// Original code does not do 1/(N-1) in the standard deviation calculation, but
//      but it also does not multiply by 1/(N-1) in pearson correlation calc.
//      these 2 cancel out.
// current:
//      compute the regular standard score.
template <typename IT, typename OT = IT, bool SampleStats = true>
class StandardScore : public splash::kernel::UnaryVectorOpBase<IT, OT> {
    public:
        inline void operator()(IT const * __restrict__ in_vec, 
            size_t const & count,
            OT * __restrict__ out_vec) {

            const OT avg = 1.0L / static_cast<OT>(count);
            const OT sample_avg = 1.0L / static_cast<OT>(count - SampleStats);
            
            // compute mean
            OT meanX = 0;
            OT meanX2 = 0;
            OT x;
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#if defined(USE_SIMD)
#pragma omp simd reduction(+:meanX)
#endif
            for (size_t j = 0; j < count; ++j) {
                x = static_cast<OT>(in_vec[j]);
                meanX += x * avg;
                meanX2 += x * sample_avg * x;
            }

            /*compute the variance*/
            OT stdevX;
            if (SampleStats) 
                stdevX = sqrt(meanX2 - (static_cast<OT>(count) * sample_avg) * meanX * meanX );
            else
                stdevX = sqrt(meanX2 - meanX * meanX);

            OT invStdevX = 1.0L / stdevX;

            /*normalize the data*/
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#if defined(USE_SIMD)
#pragma omp simd
#endif
            for (size_t j = 0; j < count; ++j) {
                x = static_cast<OT>(in_vec[j]) - meanX;
                out_vec[j] = x * invStdevX;
            }
        }
};



template <typename IT, typename OT = IT>
class Rank : public splash::kernel::UnaryVectorOpBase<IT, OT> {
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
            alloc_size = (_count * sizeof(MyPair) + static_cast<size_t>(SPL_CACHELINE_WIDTH - 1)) & ~(static_cast<size_t>(SPL_CACHELINE_WIDTH - 1));
			sort_buffer = reinterpret_cast<MyPair* >(std::aligned_alloc(SPL_CACHELINE_WIDTH, alloc_size));
		}
		~Rank() {
			free(sort_buffer);
		}


        inline void operator()(IT const * __restrict__ in_vec, 
            size_t const & count,
            OT * __restrict__ out_vec) {

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