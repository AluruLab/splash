//
// Simple and fast atof (ascii to float) function.
//
// - Executes about 5x faster than standard MSCRT library atof().
// - An attractive alternative if the number of calls is in the millions.
// - Assumes input is a proper integer, fraction, or scientific format.
// - Matches library atof() to 15 digits (except at extreme exponents).
// - Follows atof() precedent of essentially no error checking.
//
// 09-May-2009 Tom Van Baak (tvb) www.LeapSecond.com
// http://www.leapsecond.com/tools/fast_atof.c

// modified by Tony Pan, Aug 20, 2020

#pragma once

#include "utils/precise_float.hpp"

namespace splash { namespace utils {

#define white_space(c) ((c) == ' ' || (c) == '\t')
#define valid_digit(c) ((c) >= '0' && (c) <= '9')

// not precise.  can potentially cause problems downstream.
inline double atof (const char *p)
{
    using wide = typename splash::utils::widened_type<double>::type;
    bool frac;
    wide sign, value;  // modified here

    // Skip leading white space, if any.

    while (white_space(*p) ) {
        ++p;
    }

    // Get sign, if any.

    sign = static_cast<wide>(1.0);
    if (*p == '-') {
        sign = static_cast<wide>(-1.0);
        ++p;

    } else if (*p == '+') {
        ++p;
    }

    // Get digits before decimal point or exponent, if any.

    for (value = 0.0; valid_digit(*p); ++p) {
        value = value * static_cast<wide>(10.0) + static_cast<wide>(*p - '0');
    }

    // Get digits after decimal point, if any.

    if (*p == '.') {
        wide pow10 = static_cast<wide>(0.1);  // modified here
        ++p;
        while (valid_digit(*p)) {
            value += static_cast<wide>(*p - '0') * pow10;  // modified here
            pow10 *= static_cast<wide>(0.1);  // modified here
            ++p;
        }
    }

    // Handle exponent, if any.

    frac = false;
    if ((*p == 'e') || (*p == 'E')) {
        unsigned int expon;

        // Get sign of exponent, if any.

        ++p;
        if (*p == '-') {
            frac = true;
            ++p;

        } else if (*p == '+') {
            ++p;
        }

        // Get digits of exponent, if any.

        for (expon = 0; valid_digit(*p); ++p) {
            expon = expon * 10 + (*p - '0');
        }
        if (expon > 308) expon = 308;

        // Calculate scaling factor.
		// modified here, branch by frac.
		if (frac) {
			while (expon >= 50) { sign *= static_cast<wide>(1E-50); expon -= 50; }
			while (expon >=  8) { sign *= static_cast<wide>(1E-8);  expon -=  8; }
			while (expon >   0) { sign *= static_cast<wide>(0.1);   expon -=  1; }
		} else {
			while (expon >= 50) { sign *= static_cast<wide>(1E50);  expon -= 50; }
			while (expon >=  8) { sign *= static_cast<wide>(1E8);   expon -=  8; }
			while (expon >   0) { sign *= static_cast<wide>(10.0);  expon -=  1; }
		}
    }

    // Return signed and scaled floating point result.
    // return sign * (frac ? (value / scale) : (value * scale));
    return sign * value;
}

#define LOG2_10 3.3219280948873623478703194294894L  // long double precision, instead of doing the math.

// TCP: hopefully more precise version.  using integers internally. NOT directly modifying the bits in output. 
inline double p_atof(const char *p)
{
    size_t whole, frac;
    ssize_t frac_digits, expon;
    bool neg = false;

    // Skip leading white space, if any.

    while (white_space(*p) ) {
        ++p;
    }

    // Get sign, if any.

    if (*p == '-') {
        neg = true;
        ++p;

    } else if (*p == '+') {
        ++p;
    }

    // Get digits before decimal point or exponent, if any.
    whole = 0;
    for (; valid_digit(*p); ++p) {
        whole = whole * 10 + (*p - '0');
    }

    // Get digits after decimal point, if any.
    frac = 0;
    frac_digits = 0;
    if (*p == '.') {
        ++p;
        for (; valid_digit(*p); --frac_digits, ++p) {
            frac = frac * 10 + (*p - '0');  
        }
    }

    // Handle exponent, if any.

    expon = 0;
    if ((*p == 'e') || (*p == 'E')) {
        bool neg_exp = false;

        ++p;
        // Get sign of exponent, if any.
        if (*p == '-') {
            neg_exp = true;
            ++p;
        } else if (*p == '+') {
            ++p;
        }

        // Get digits of exponent, if any.
        for (; valid_digit(*p); ++p) {
            expon = expon * 10 + (*p - '0');
        }
        if (expon > 308) expon = 308;

        if (neg_exp) expon = -expon;
    }

    using wide = typename splash::utils::widened_type<double>::type;
    // printf("value = %lu.%lue%ld, frac expon = %ld.  frac as double %Lf\n", whole, frac, expon, frac_digits, static_cast<wide>(frac));

    // Return signed and scaled floating point result.
    // wide f = static_cast<wide>(frac_digits) * LOG2_10;  // this is 
    // wide e = static_cast<wide>(expon) * LOG2_10;
    double value = (static_cast<wide>(whole) + 
        static_cast<wide>(frac) * exp10l(static_cast<wide>(frac_digits))) * 
        exp10l(static_cast<wide>(expon));
    return neg ? -value : value;
}


}}
