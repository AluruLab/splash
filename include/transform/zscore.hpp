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
#include "utils/precise_float.hpp"

#if defined(USE_SIMD)
#include <omp.h>
#endif

namespace splash { namespace kernel { 


template <typename IT, typename OT = std::pair<IT, IT>, bool SampleStats = true>
class GaussianParams : public splash::kernel::reduce<IT, OT, splash::kernel::DEGREE::VECTOR, splash::kernel::DEGREE::SCALAR> {
    public:
        using InputType = IT;
        using OutputType = OT;
        using FT = splash::utils::widened<IT>;

        inline virtual OT operator()(IT const * in_vec,
            size_t const & count) const {

            const FT avg = 1.0L / static_cast<FT>(count);
            const FT sample_avg = 1.0L / static_cast<FT>(count - SampleStats);
            
            // compute mean
            FT mean = 0;
            FT meanX2 = 0;
            FT x;
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#if defined(USE_SIMD)
#pragma omp simd reduction(+:mean, meanX2)
#endif
            for (size_t j = 0; j < count; ++j) {
                x = static_cast<FT>(in_vec[j]);
                mean += x * avg;
                meanX2 += x * sample_avg * x;
            }

            /*compute the stdev*/
            FT stdev;
            if (SampleStats) 
                stdev = std::sqrt(meanX2 - (static_cast<FT>(count) * sample_avg) * mean * mean );
            else
                stdev = std::sqrt(meanX2 - mean * mean);

			return {mean, stdev};
        }
};

// for sample standard score.  
// Original code does not do 1/(N-1) in the standard deviation calculation, but
//      but it also does not multiply by 1/(N-1) in pearson correlation calc.
//      these 2 cancel out.
// current:
//      compute the regular standard score.
template <typename IT, typename OT = IT, bool SampleStats = true>
class StandardScore : public splash::kernel::transform<IT, OT, splash::kernel::DEGREE::VECTOR> {
    protected:
        GaussianParams<IT, std::pair<OT, OT>, SampleStats> stats;

    public:
        using InputType = IT;
        using OutputType = OT;
        using FT = splash::utils::widened<OT>;
        
        inline virtual void operator()(IT const *  in_vec, 
            size_t const & count,
            OT *  out_vec) const {

            OT meanX, stdevX;
            std::tie(meanX, stdevX) = stats(in_vec, count);

            OT invStdevX = 1.0L / stdevX;
            OT x;
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