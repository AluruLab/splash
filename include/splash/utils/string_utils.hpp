/*
 * Copyright 2020 Georgia Tech Research Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 * Author(s): Tony C. Pan
 */

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
// revised by Tony Pan, Aug 27, 2020.  parse integer, fraction, and exponent parts as ints so exact
//   avoid exp10 and exp2.  iterate over expon, multiply via literal constant.  also use lookup tables.
//   this version is now slightly faster than the naive optimized, and significantly faster than atof.
//   using long double, allows double precision to be exact. (unlike the Ven Baak version)

#pragma once

#include "splash/utils/precise_float.hpp"
#if __cplusplus >= 201703L // C++17 and later 
#include <string_view>
#else
#include <string>
#endif

namespace splash { namespace utils {

std::string trim(const std::string& line)
{
    const char* WhiteSpace = " \t\v\r\n";
    std::size_t start = line.find_first_not_of(WhiteSpace);
    std::size_t end = line.find_last_not_of(WhiteSpace);
    return start == end ? std::string() : line.substr(start, end - start + 1);
}


#define white_space(c) ((c) == ' ' || (c) == '\t')
#define valid_digit(c) ((c) >= '0' && (c) <= '9')

#define LOG2_10 3.3219280948873623478703194294894L  // long double precision, instead of doing the math.

// define static global lookup table.
static const typename splash::utils::widened_type<double>::type pow10_neg[25] = {
    1.0L,
    1E-1L,
    1E-2L,
    1E-3L,
    1E-4L,
    1E-5L,
    1E-6L,
    1E-7L,
    1E-8L,
    1E-9L,
    1E-10L,
    1E-11L,
    1E-12L,
    1E-13L,
    1E-14L,
    1E-15L,
    1E-16L,
    1E-17L,
    1E-18L,
    1E-19L,
    1E-20L,
    1E-21L,
    1E-22L,
    1E-23L,
    1E-24L
};
static const typename splash::utils::widened_type<double>::type pow10_pos[25] = {
    1.0L,
    1E1L,
    1E2L,
    1E3L,
    1E4L,
    1E5L,
    1E6L,
    1E7L,
    1E8L,
    1E9L,
    1E10L,
    1E11L,
    1E12L,
    1E13L,
    1E14L,
    1E15L,
    1E16L,
    1E17L,
    1E18L,
    1E19L,
    1E20L,
    1E21L,
    1E22L,
    1E23L,
    1E24L
};

// TCP: hopefully more precise version.  using integers internally. NOT directly modifying the bits in output. 
inline double atof(const char *p)
{
    using wide = typename splash::utils::widened_type<double>::type;

    size_t whole, frac;
    ssize_t frac_digits;
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
        whole = (whole << 3) + (whole << 1) + (*p - '0');
    }

    // Get digits after decimal point, if any.
    frac = 0;
    frac_digits = 0;
    if (*p == '.') {
        ++p;
        for (; valid_digit(*p); ++frac_digits, ++p) {
            frac = (frac << 3)  + (frac << 1) + (*p - '0');
        }
    }

    // Handle exponent, if any.
    wide scale = 1.0L;
    if ((*p == 'e') || (*p == 'E')) {
        bool neg_exp = false;
        size_t expon = 0;

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
            expon = (expon << 3) + (expon << 1) + (*p - '0');
        }
        if (expon > 308) expon = 308;

        if (neg_exp) {
            // 25 * (1 + x + x^2) >= 308. -> x = 3
            while (expon >= 100) { scale *= 1E-100L; expon -= 100; }
            while (expon >= 25) { scale *= 1E-25L; expon -= 25; }
            scale *= pow10_neg[expon];
        } else {
            while (expon >= 100) { scale *= 1E100L; expon -= 100; }
            while (expon >= 25) { scale *= 1E25L; expon -= 25; }
            scale *= pow10_pos[expon];
        }

    }

    // using exp10l vs exp10 has a performance impact.  precision is lost with exp10.
    // both frac and exp results need to be wide to be precise, but again, there is a cost for this.
    // may be able to use a lookup table for frac_digits.
    // exp10 is expensive.  exp2 is a little less, lookup is faster.

    // Return signed and scaled floating point result.
    double value = (static_cast<wide>(whole) + 
        static_cast<wide>(frac) * pow10_neg[frac_digits]) * 
        scale;
        // exp2(static_cast<wide>(expon) * LOG2_10);  // exp2 is about 20% faster than exp10.
    
    // alternative - uses exp10l (or exp2, same idea)
    // double value = (static_cast<wide>(whole) + 
    //     static_cast<wide>(frac) * exp10l(static_cast<wide>(-frac_digits))) * 
    //     exp10l(static_cast<wide>(expon));
    return neg ? -value : value;
}

/** multiply has 1 cycle throughput but 2 cycle latency even on coffee lake.
 * dec 10 is bin 1010.  10x + y = x << 3 + x << 1 + y
*/


//========= Suffix/Prefix matching.
// https://stackoverflow.com/questions/874134/find-out-if-string-ends-with-another-string-in-c

#if __cplusplus >= 201703L // C++17 and later 
#include <string_view>

static bool endsWith(std::string_view str, std::string_view suffix)
{
    return str.size() >= suffix.size() && 0 == str.compare(str.size()-suffix.size(), suffix.size(), suffix);
}

// static bool startsWith(std::string_view str, std::string_view prefix)
// {
//     return str.size() >= prefix.size() && 0 == str.compare(0, prefix.size(), prefix);
// }
#else  // C++ 14 and earlier.

// static bool endsWith(const std::string& str, const std::string& suffix)
// {
//     return str.size() >= suffix.size() && 0 == str.compare(str.size()-suffix.size(), suffix.size(), suffix);
// }

// static bool startsWith(const std::string& str, const std::string& prefix)
// {
//     return str.size() >= prefix.size() && 0 == str.compare(0, prefix.size(), prefix);
// }

static bool endsWith(const std::string& str, const char* suffix, unsigned suffixLen)
{
    return str.size() >= suffixLen && 0 == str.compare(str.size()-suffixLen, suffixLen, suffix, suffixLen);
}

static bool endsWith(const std::string& str, const char* suffix)
{
    return endsWith(str, suffix, std::string::traits_type::length(suffix));
}

// static bool startsWith(const std::string& str, const char* prefix, unsigned prefixLen)
// {
//     return str.size() >= prefixLen && 0 == str.compare(0, prefixLen, prefix, prefixLen);
// }

// static bool startsWith(const std::string& str, const char* prefix)
// {
//     return startsWith(str, prefix, std::string::traits_type::length(prefix));
// }
#endif

}}
