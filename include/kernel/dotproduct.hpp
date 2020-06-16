/*
 *  dotproduct.hpp
 *
 *  Created on: June 11, 2020
 *  Author: Tony Pan
 *  Affiliation: Institute for Data Engineering and Science
 *  			Georgia Institute of Technology, Atlanta, GA 30332
 */

#pragma once

#include "kernel/kernel_base.hpp"

#if defined(USE_SIMD)
#include <omp>
#endif

namespace splash { namespace kernel { 


template<typename IT>
class DotProductKernel : public splash::kernel::VV2SOp<IT, IT> {

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




}}