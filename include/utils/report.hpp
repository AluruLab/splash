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

#ifdef USE_MPI
#include <mpi.h>
#endif

#define PRINT_ERR(...)  fprintf(stderr, __VA_ARGS__)
#ifdef USE_MPI
#define PRINT(...) do {\
    int rank; \
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); \
    if (rank == 0) fprintf(stdout, __VA_ARGS__); \
} while (false)
#else
#define PRINT(...)  fprintf(stdout, __VA_ARGS__)
#endif

