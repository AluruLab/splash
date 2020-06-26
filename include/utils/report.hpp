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

// has to use macro.  if function, format string will become variable instead of literal.

#define PRINT_ERR(...)  fprintf(stderr, __VA_ARGS__)

#ifdef USE_MPI
#define PRINT_RANk() do {\
    int rank; \
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); \
    fprintf(stdout, "RANK %d: ", rank); \
} while (false)
#else  
#define PRINT_RANK() 
#endif

#ifdef USE_MPI
#define PRINT_MPI_ROOT(...) do {\
    int rank; \
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); \
    if (rank == 0) { \
        fprintf(stdout, "RANK %d: ", rank); \
        fprintf(stdout, __VA_ARGS__); \
    }\
} while (false)
#else
#define PRINT_MPI_ROOT(...)  fprintf(stdout, __VA_ARGS__)
#endif

#ifdef USE_MPI
#define PRINT_MPI(...) do {\
    int rank; \
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); \
    fprintf(stdout, "RANK %d: ", rank); \
    fprintf(stdout, __VA_ARGS__); \
} while (false)
#else
#define PRINT_MPI(...) fprintf(stdout, __VA_ARGS__)
#endif

#define PRINT(...) fprintf(stdout, __VA_ARGS__)

#define FLUSH() do { fflush(stdout); fflush(stderr); } while(false)