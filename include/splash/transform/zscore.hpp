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

#include "splash/kernel/kernel_base.hpp"
#include "splash/utils/precise_float.hpp"

#if defined(USE_SIMD)
#include <omp.h>
#endif

namespace splash { namespace kernel { 


template <typename IT, typename OT = IT, bool SampleStats = true>
class GaussianParams : public splash::kernel::reduce<IT, std::pair<OT, OT>, splash::kernel::DEGREE::VECTOR> {
    public:
        using InputType = IT;
        using OutputType = std::pair<OT, OT>;
        using FT = splash::utils::widened<OT>;

        inline virtual OutputType operator()(IT const * in_vec,
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
                stdev = std::sqrt(std::abs(meanX2 - (static_cast<FT>(count) * sample_avg) * mean * mean));
            else
                stdev = std::sqrt(std::abs(meanX2 - mean * mean));

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
        GaussianParams<IT, OT, SampleStats> stats;

    public:
        using InputType = IT;
        using OutputType = OT;
        using FT = splash::utils::widened<OT>;
        
        inline virtual void operator()(IT const *  in_vec, 
            size_t const & count,
            OT *  out_vec) const {

            OT meanX, stdevX;
            std::tie(meanX, stdevX) = stats(in_vec, count);

            if (std::abs(stdevX) < std::numeric_limits<OT>::epsilon()) {
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#if defined(USE_SIMD)
#pragma omp simd
#endif
                for (size_t j = 0; j < count; ++j)
                    out_vec[j] = 0.0;
                return;
            }

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


template <typename IT, typename OT = IT, bool SampleStats = true>
class GaussianParamsExclude1 : public splash::kernel::reduce_except1<IT, std::pair<OT, OT>, splash::kernel::DEGREE::VECTOR> {
    public:
        using InputType = IT;
        using OutputType = std::pair<OT, OT>;
        using FT = splash::utils::widened<OT>;

        inline virtual OutputType operator()(size_t const & exclude_id, IT const * in_vec,
            size_t const & count) const {
                
            bool exclusion = exclude_id < count;
            size_t cnt = count - exclusion;

            const FT avg = 1.0L / static_cast<FT>(cnt);
            const FT sample_avg = 1.0L / static_cast<FT>(cnt - SampleStats);
            
            // compute mean
            FT mean = 0;
            FT meanX2 = 0;
            FT x;
            // add all entries including the exclusion
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#if defined(USE_SIMD)
#pragma omp simd reduction(+:mean, meanX2)
#endif
            for (size_t j = 0; j < count; ++j) {
                x = static_cast<FT>(in_vec[j]);
                mean += (x * avg);
                meanX2 += (x * sample_avg * x);
            }
            if (exclusion) {
                // remove the excluded entry.
                x = static_cast<FT>(in_vec[exclude_id]);
                mean -= x * avg;
                meanX2 -= x * sample_avg * x;
            }

            /*compute the stdev*/
            FT stdev;
            if (SampleStats) 
                stdev = std::sqrt(std::abs(meanX2 - (static_cast<FT>(cnt) * sample_avg) * mean * mean));
            else
                stdev = std::sqrt(std::abs(meanX2 - mean * mean));

			return {mean, stdev};
        }
};


}}