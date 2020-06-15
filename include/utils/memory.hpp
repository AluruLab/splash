/*
 *  memory.hpp
 *
 *  Created on: June 12, 2020
 *  Author: Tony C. Pan
 *  Affiliation: School of Computational Science & Engineering
 *  						Georgia Institute of Technology, Atlanta, GA 30332
 */

#pragma once

#include <unistd.h>  // sysconf, for getting cacheline size.
#include <stdlib.h>  // aligned_alloc
#include <new>       // bad_alloc
#include <exception>
#include "utils/error_handler.hpp"


namespace splash { namespace utils { 

inline size_t get_cacheline_size() {
    long val = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    if (val < 1) {
        splash::utils::print_err("%s:%d: Unable to detect cacheline size.  using default of 64\n", __FILE__, __LINE__);
        return 64UL;
    } else 
        return val;
}

inline size_t get_aligned_size(size_t const & bytes, size_t const & align) {
    if ((align & (align - 1UL)) != 0) throw std::domain_error("Alignment is not a power of 2.\n");
    return (bytes + align - 1UL) & ~(align - 1UL);
}

inline size_t get_aligned_size(size_t const & bytes) {
    return splash::utils::get_aligned_size(bytes, splash::utils::get_cacheline_size());
}

// default is cache line size.
inline void* aligned_alloc(size_t const & bytes, size_t const & alignment) {
    void* data = ::std::aligned_alloc(alignment, splash::utils::get_aligned_size(bytes, alignment));

    if (!data) {
        splash::utils::print_err("%s:%d: Memory allocation failed\n", __FILE__, __LINE__);
        throw std::bad_alloc();
    }

    return data;
}

inline void* aligned_alloc(size_t const & bytes) {
    return splash::utils::aligned_alloc(bytes, splash::utils::get_cacheline_size());
}

inline void free(void* data) {
    if (data) {
        free(data);
        data = nullptr;
    }
}



}}