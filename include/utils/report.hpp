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

template <class... Types>
void print(Types... args) {
#ifdef WITH_MPI
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) fprintf(stdout, args...);
#else
    fprintf(stdout, args...);
#endif
}



} }

