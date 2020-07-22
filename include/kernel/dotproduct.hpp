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
#include <omp.h>
#endif

namespace splash { namespace kernel { 


template<typename IT, typename OT>
class DotProductKernel : public splash::kernel::inner_product<IT, OT, splash::kernel::DEGREE::VECTOR> {

	public:
		using InputType = IT;
		using OutputType = OT;

		inline virtual OT operator()(IT const * first, IT const * second, size_t const & count) const  {
			OT prod = 0;
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#if defined(USE_SIMD)
#pragma omp simd reduction(+:prod)
#endif
			// MAJORITY OF TIME HERE.  slowdown: O3 + omp simd (1x).  O3 (5x).  sanitizer + omp simd (11x), sanitizer (5.5x) 
			for (size_t j = 0; j < count; ++j) {
				prod += static_cast<OT>(first[j] * second[j]);
			}
			return prod;
		};
};




}}