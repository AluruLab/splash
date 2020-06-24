/*
 *  zscore.hpp
 *
 *  Created on: June 11, 2020
 *  Author: Tony Pan
 *  Affiliation: Institute for Data Engineering and Science
 *  			Georgia Institute of Technology, Atlanta, GA 30332
 */

#pragma once

#include <cmath>  // sqrt

#include "kernel/kernel_base.hpp"

#if defined(USE_SIMD)
#include <omp.h>
#endif

namespace splash { namespace kernel { 



// for sample standard score.  
// Original code does not do 1/(N-1) in the standard deviation calculation, but
//      but it also does not multiply by 1/(N-1) in pearson correlation calc.
//      these 2 cancel out.
// current:
//      compute the regular standard score.
template <typename IT, typename OT = IT, bool SampleStats = true>
class StandardScore : public splash::kernel::V2VOp<IT, OT> {
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



}}