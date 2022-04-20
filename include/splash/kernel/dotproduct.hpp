/*
 *  dotproduct.hpp
 *
 *  Created on: June 11, 2020
 *  Author: Tony Pan
 *  Affiliation: Institute for Data Engineering and Science
 *  			Georgia Institute of Technology, Atlanta, GA 30332
 */

#pragma once

#include "splash/kernel/kernel_base.hpp"
#include "splash/utils/precise_float.hpp"

#if defined(USE_SIMD)
#include <omp.h>
#endif

#include <x86intrin.h>

namespace splash { namespace kernel { 

template <typename T>
inline T dotp_scalar(T const * xx, T const * yy, size_t const & count) {
    T val = 0.0;
    for (size_t k = 0; k < count; ++k) {
        val += (xx[k] * yy[k]);
    }
    return val;
}   

template <typename T>
inline T dotp_omp(T const * xx, T const * yy, size_t const & count) {
    T val = 0.0;
#if defined(__INTEL_COMPILER)
#pragma vector aligned
#endif
#if defined(USE_SIMD)
#pragma omp simd reduction(+:val)
#endif
    for (size_t k = 0; k < count; ++k) {
        val += (xx[k] * yy[k]);
    }
    return val;
}  
// no float version yet.

inline double dotp_sse(double const * xx, double const * yy, size_t const & count) {
#ifdef __SSE2__
    double val[2] = {0.0, 0.0};

    __m128d a1, a2, a3, a4, b1, b2, b3, b4;
    __m128d acc1 = _mm_setzero_pd();
    __m128d acc2 = acc1;
    __m128d acc3 = acc1;
    __m128d acc4 = acc1;

    // compute the bulk
    size_t max = count & 0xFFFFFFFFFFFFFFF8;
    size_t k = 0;
    for (; k < max; k += 8) {
        a1 = _mm_loadu_pd(xx + k);
        b1 = _mm_loadu_pd(yy + k);
        acc1 = _mm_add_pd(acc1, _mm_mul_pd(a1, b1));

        a2 = _mm_loadu_pd(xx + k + 2);
        b2 = _mm_loadu_pd(yy + k + 2);
        acc2 = _mm_add_pd(acc2, _mm_mul_pd(a2, b2));

        a3 = _mm_loadu_pd(xx + k + 4);
        b3 = _mm_loadu_pd(yy + k + 4);
        acc3 = _mm_add_pd(acc3, _mm_mul_pd(a3, b3));

        a4 = _mm_loadu_pd(xx + k + 6);
        b4 = _mm_loadu_pd(yy + k + 6);
        acc4 = _mm_add_pd(acc4, _mm_mul_pd(a4, b4));
    }

    // compute the remaining.
    max = (count - k) >> 1;
    switch (max) {
        case 3:  
            a3 = _mm_loadu_pd(xx + k + 4);
            b3 = _mm_loadu_pd(yy + k + 4);
            acc3 = _mm_add_pd(acc3, _mm_mul_pd(a3, b3));
        case 2:
            a2 = _mm_loadu_pd(xx + k + 2);
            b2 = _mm_loadu_pd(yy + k + 2);
            acc2 = _mm_add_pd(acc2, _mm_mul_pd(a2, b3));
        case 1:
            a1 = _mm_loadu_pd(xx + k);
            b1 = _mm_loadu_pd(yy + k);
            acc1 = _mm_add_pd(acc1, _mm_mul_pd(a1, b3));
        default: break;
    }
    k += (max << 1);

    // handle accumulators, extract data
    acc4 = _mm_add_pd(acc4, acc3);
    acc2 = _mm_add_pd(acc2, acc1);
    acc1 = _mm_add_pd(acc2, acc4);
    _mm_storeu_pd(val, acc1);

    // last 1 if there.
    if ((count & 1) > 0) {
        val[0] += (xx[count - 1] * yy[count - 1]);
    }

    val[0] += val[1];

    return val[0];
#else
    return 0.0;
#endif
}   


inline double dotp_avx(double const * xx, double const * yy, size_t const & count) {
#ifdef __AVX__
    double val[4] = {0.0, 0.0, 0.0, 0.0};

    __m256d a1, a2, a3, a4, b1, b2, b3, b4;
    __m256d acc1 = _mm256_setzero_pd();
    __m256d acc2 = acc1;
    __m256d acc3 = acc1;
    __m256d acc4 = acc1;

    // compute the bulk
    size_t max = count & 0xFFFFFFFFFFFFFFF0;
    size_t k = 0;
    for (; k < max; k += 16) {
        a1 = _mm256_loadu_pd(xx + k);
        b1 = _mm256_loadu_pd(yy + k);
        acc1 = _mm256_add_pd(acc1, _mm256_mul_pd(a1, b1));

        a2 = _mm256_loadu_pd(xx + k + 4);
        b2 = _mm256_loadu_pd(yy + k + 4);
        acc2 = _mm256_add_pd(acc2, _mm256_mul_pd(a2, b2));

        a3 = _mm256_loadu_pd(xx + k + 8);
        b3 = _mm256_loadu_pd(yy + k + 8);
        acc3 = _mm256_add_pd(acc3, _mm256_mul_pd(a3, b3));

        a4 = _mm256_loadu_pd(xx + k + 12);
        b4 = _mm256_loadu_pd(yy + k + 12);
        acc4 = _mm256_add_pd(acc4, _mm256_mul_pd(a4, b4));
    }

    // compute the remaining.
    max = (count - k) >> 2;
    switch (max) {
        case 3:  
            a3 = _mm256_loadu_pd(xx + k + 8);
            b3 = _mm256_loadu_pd(yy + k + 8);
            acc3 = _mm256_add_pd(acc3, _mm256_mul_pd(a3, b3));
        case 2:
            a2 = _mm256_loadu_pd(xx + k + 4);
            b2 = _mm256_loadu_pd(yy + k + 4);
            acc2 = _mm256_add_pd(acc2, _mm256_mul_pd(a2, b2));
        case 1:
            a1 = _mm256_loadu_pd(xx + k);
            b1 = _mm256_loadu_pd(yy + k);
            acc1 = _mm256_add_pd(acc1, _mm256_mul_pd(a1, b1));
        default: break;
    }
    k += (max << 2);

    // handle accumulators, extract values
    acc4 = _mm256_add_pd(acc4, acc3);
    acc2 = _mm256_add_pd(acc2, acc1);
    acc1 = _mm256_add_pd(acc2, acc4);
    _mm256_storeu_pd(val, acc1);
 
    // last ones (up to 3) if there.
    for (size_t i = 0; k < count; ++i, ++k) {
        val[i] += (xx[k] * yy[k]);
    }

    val[0] += val[1];
    val[2] += val[3];

    val[0] += val[2];

    return val[0];
#else
    return 0.0;
#endif
}   



inline double dotp_avx512(double const * xx, double const * yy, size_t const & count) {
#ifdef __AVX512F__  
    double val[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

    // has FMA, use it.

    __m512d a1, a2, a3, a4, b1, b2, b3, b4;
    __m512d acc1 = _mm512_setzero_pd();
    __m512d acc2 = acc1;
    __m512d acc3 = acc1;
    __m512d acc4 = acc1;

    // compute the bulk
    size_t max = count & 0xFFFFFFFFFFFFFFE0;
    size_t k = 0;
    for (; k < max; k += 32) {
        a1 = _mm512_loadu_pd(xx + k);
        b1 = _mm512_loadu_pd(yy + k);
        acc1 = _mm512_fmadd_pd(a1, b1, acc1);

        a2 = _mm512_loadu_pd(xx + k + 8);
        b2 = _mm512_loadu_pd(yy + k + 8);
        acc2 = _mm512_fmadd_pd(a2, b2, acc2);

        a3 = _mm512_loadu_pd(xx + k + 16);
        b3 = _mm512_loadu_pd(yy + k + 16);
        acc3 = _mm512_fmadd_pd(a3, b3, acc3);

        a4 = _mm512_loadu_pd(xx + k + 24);
        b4 = _mm512_loadu_pd(yy + k + 24);
        acc4 = _mm512_fmadd_pd(a4, b4, acc4);
    }

    // compute the remaining.
    max = (count - k) >> 3;
    switch (max) {
        case 3:  
            a3 = _mm512_loadu_pd(xx + k + 16);
            b3 = _mm512_loadu_pd(yy + k + 16);
            acc3 = _mm512_fmadd_pd(a3, b3, acc3);
        case 2:
            a2 = _mm512_loadu_pd(xx + k + 8);
            b2 = _mm512_loadu_pd(yy + k + 8);
            acc2 = _mm512_fmadd_pd(a2, b2, acc2);
        case 1:
            a1 = _mm512_loadu_pd(xx + k);
            b1 = _mm512_loadu_pd(yy + k);
            acc1 = _mm512_fmadd_pd(a1, b1, acc1);
        default: break;
    }
    k += (max << 3);

    // handle accumulators and extract
    acc4 = _mm512_add_pd(acc4, acc3);
    acc2 = _mm512_add_pd(acc2, acc1);
    acc1 = _mm512_add_pd(acc2, acc4);
    _mm512_storeu_pd(val, acc1);
 
    // last ones (up to 7) if there.
    for (size_t i = 0; k < count; ++i, ++k) {
        val[i] += (xx[k] * yy[k]);
    }

    val[0] += val[1];
    val[2] += val[3];
    val[4] += val[5];
    val[6] += val[7];

    val[0] += val[2];
    val[4] += val[6];

    val[0] += val[4];

    return val[0];
#else
    return 0.0;
#endif
}   



template<typename IT, typename OT>
class DotProductKernel : public splash::kernel::inner_product<IT, OT, splash::kernel::DEGREE::VECTOR> {

	public:
		using InputType = IT;
		using OutputType = OT;
		using FT = splash::utils::widened<OT>;

		inline virtual OT operator()(IT const * first, IT const * second, size_t const & count) const  {
			FT prod = 0;
#if defined(__AVX512F__)
            prod = dotp_avx512(first, second, count);
#elif defined(__AVX__)
            prod = dotp_avx(first, second, count);
#elif defined(__SSE2__)
            prod = dotp_sse(first, second, count);
#else
			// MAJORITY OF TIME HERE.  slowdown: O3 + omp simd (1x).  O3 (5x).  sanitizer + omp simd (11x), sanitizer (5.5x) 
            prod = dotp_omp(first, second, count);
#endif
			return prod;
		};
};




}}