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

namespace splash { namespace utils {

#define white_space(c) ((c) == ' ' || (c) == '\t')
#define valid_digit(c) ((c) >= '0' && (c) <= '9')

double atof (const char *p)
{
    int frac;
    double sign, value;  // modified here

    // Skip leading white space, if any.

    while (white_space(*p) ) {
        p += 1;
    }

    // Get sign, if any.

    sign = 1.0;
    if (*p == '-') {
        sign = -1.0;
        p += 1;

    } else if (*p == '+') {
        p += 1;
    }

    // Get digits before decimal point or exponent, if any.

    for (value = 0.0; valid_digit(*p); p += 1) {
        value = value * 10.0 + (*p - '0');
    }

    // Get digits after decimal point, if any.

    if (*p == '.') {
        double pow10 = 0.1;  // modified here
        p += 1;
        while (valid_digit(*p)) {
            value += (*p - '0') * pow10;  // modified here
            pow10 *= 0.1;  // modified here
            p += 1;
        }
    }

    // Handle exponent, if any.

    frac = 0;
    if ((*p == 'e') || (*p == 'E')) {
        unsigned int expon;

        // Get sign of exponent, if any.

        p += 1;
        if (*p == '-') {
            frac = 1;
            p += 1;

        } else if (*p == '+') {
            p += 1;
        }

        // Get digits of exponent, if any.

        for (expon = 0; valid_digit(*p); p += 1) {
            expon = expon * 10 + (*p - '0');
        }
        if (expon > 308) expon = 308;

        // Calculate scaling factor.
		// modified here, branch by frac.
		if (frac) {
			while (expon >= 50) { sign *= 1E-50; expon -= 50; }
			while (expon >=  8) { sign *= 1E-8;  expon -=  8; }
			while (expon >   0) { sign *= 0.1;   expon -=  1; }
		} else {
			while (expon >= 50) { sign *= 1E50;  expon -= 50; }
			while (expon >=  8) { sign *= 1E8;   expon -=  8; }
			while (expon >   0) { sign *= 10.0;  expon -=  1; }
		}
    }

    // Return signed and scaled floating point result.
    // return sign * (frac ? (value / scale) : (value * scale));
    return sign * value;
}


}}
