/*
 *  benchmark.hpp
 *
 *  Created on: June 12, 2020
 *  Author: Tony C. Pan
 *  Affiliation: School of Computational Science & Engineering
 *  						Georgia Institute of Technology, Atlanta, GA 30332
 */

#pragma once

#include <chrono> // steady_clock, duration cast

#define getSysTime() std::chrono::steady_clock::now()

template <class TimePoint>
inline double get_duration_s(TimePoint const & start, TimePoint const & end) {
    return std::chrono::duration<double>(end - start).count();
}

