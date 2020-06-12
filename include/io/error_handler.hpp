/*
 *  error_handler.hpp
 *
 *  Created on: June 11, 2020
 *  Author: Tony Pan
 *  Affiliation: Institute for Data Engineering and Science
 *  			Georgia Institute of Technology, Atlanta, GA 30332
 */

#pragma once

#include <cstdio>

namespace splash{ namespace io {

template <class... Types>
void print_err(Types... args) {
    fprintf(stderr, args...);
}



} }

