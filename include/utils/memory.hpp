/*
 *  memory.hpp
 *
 *  Created on: June 12, 2020
 *  Author: Tony C. Pan
 *  Affiliation: School of Computational Science & Engineering
 *  						Georgia Institute of Technology, Atlanta, GA 30332
 */

#pragma once

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <unistd.h>  // sysconf, for getting cacheline size.
#endif

#include <stdlib.h>  // aligned_alloc, posix_memalign
#include <new>       // bad_alloc
#include <exception>
#include "utils/report.hpp"
#include <cstring>


/*
 * NOTE:  mvapich intercepts memory allocation and free operations for registration with infiniband:  malloc, posix_memalign, and free.
 *          mvapich2 2.3.3 and earlier does not intercept std c lib aligned_alloc.   but the free is intercepted.
 *        the result is on free, we get a segv.
 *        not clear if _mm_malloc and _mm_free is supported.
 * 
 *        OpenMPI does not perform this registration, and likely intel MPI does not.  both of which do not segv.
 * 
 * See https://github.com/QMCPACK/qmcpack/issues/1703
 * 
 * Proposed solutions include 
 *      1. use compiler macro MVAPICH2_VERSION to change to using posix_memalign when necessary.
 *      2. loading libc before libmpi (prevents interception).  This probably reduce performance.
 *      3. switch to using posix_memalign everywhere.
 * mvapich2 2.3.4 fixes this by including interceptor for aligned_alloc.  
 *  http://mvapich.cse.ohio-state.edu/static/media/mvapich/MV2_CHANGELOG-2.3.4.txt
 * 
 * it's not clear that there are performance benefits for aligned_alloc or posix_memalign, so for mvapich2 we can just globally use posix_memalign.
 * the concern is for non-posix systems.  2 methods are potentially universal:  aligned_alloc, and _mm_malloc.  so these are what we'll use.
 * 
 */
#ifdef USE_MPI
#include <mpi.h>   // need to see if we are using MVAPICH2. 

#ifdef MVAPICH2_VERSION 
#include <xmmintrin.h>   // if MVAPICH2, then use _mm_malloc, _mm_free
#endif

#endif


namespace splash { namespace utils { 

inline size_t get_cacheline_size() {
    long val = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
    if (val < 1) {
        PRINT_ERR("%s:%d: Unable to detect cacheline size.  using default of 64\n", __FILE__, __LINE__);
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



inline void afree(void* data) {
    if (data) {
#if defined(MVAPICH2_VERSION)
        // MVAPICH2 2.3.3 and earlier does not support aligned_alloc.  for cross platform, use _mm_free.
        _mm_free(data);
#else
        // not using MVAPICH2 or not using MPI.  so use free
        free(data);
#endif
    }
}


// ---------------- alloc, free, realloc.  
// no we are disabling bytes size of 0.

// like malloc, but aligned. default is cache line size.
inline void* aalloc(size_t const & bytes, size_t const & alignment) {
    if (bytes == 0) {
        PRINT_ERR("%s:%d: aligned alloc with size == 0\n", __FILE__, __LINE__);
        return nullptr;
        // throw std::logic_error("aalloc with size == 0");
    }

    void * data;
#if defined(MVAPICH2_VERSION)
    // MVAPICH2 2.3.3 and earlier does not support aligned_alloc.  for cross platform, use _mm_malloc.
    data = _mm_malloc(splash::utils::get_aligned_size(bytes, alignment), alignment);
#else
    // not using MVAPICH2 or not using MPI.  so use aligned_alloc.
    data = aligned_alloc(alignment, splash::utils::get_aligned_size(bytes, alignment));
#endif
    if (!data) {
        PRINT_ERR("%s:%d: Memory allocation failed\n", __FILE__, __LINE__);
        throw std::bad_alloc();
    }
    return data;
}

inline void* aalloc(size_t const & bytes) {
    return splash::utils::aalloc(bytes, splash::utils::get_cacheline_size());
}

// like calloc
inline void* acalloc(size_t const & bytes, size_t const & alignment) {
    void *data = splash::utils::aalloc(bytes, alignment);
    memset(data, 0, bytes);
    return data;
}

inline void* acalloc(size_t const & bytes) {
    return splash::utils::acalloc(bytes, splash::utils::get_cacheline_size());
}

template <typename T>
inline T* aalloc(size_t const & count, size_t const & alignment) {
    return reinterpret_cast<T*>(splash::utils::aalloc(count * sizeof(T), alignment));
}
template <typename T>
inline T* aalloc(size_t const & count) {
    return splash::utils::aalloc<T>(count, splash::utils::get_cacheline_size());
}
template <typename T>
inline T* acalloc(size_t const & count, size_t const & alignment) {
    return reinterpret_cast<T*>(splash::utils::acalloc(count * sizeof(T), alignment));
}
template <typename T>
inline T* acalloc(size_t const & count) {
    return splash::utils::acalloc<T>(count, splash::utils::get_cacheline_size());
}




// resize.  no copy of old data..  return actual size of buffer.
template <typename T>
inline std::pair<T *, size_t> aresize(T * data, size_t const & old_count, size_t const & count, size_t const & alignment) {
   // does not resize down.
    bool allocate = (data == nullptr) || (old_count < count);
    if (allocate) {
        if (data) splash::utils::afree(data);
        data = reinterpret_cast<T*>(splash::utils::aalloc(count * sizeof(T), alignment));  // if bytes == 0 here, that means old data==nullptr..  we return nullptr and 0 bytes.
        return std::make_pair(data, count);
    }  else
        return std::make_pair(data, old_count);
}
template <typename T>
inline std::pair<T *, size_t> aresize(T * data, size_t const & old_count, size_t const & count) {
    return splash::utils::aresize(data, old_count, count, splash::utils::get_cacheline_size());
}
template <typename T>
inline std::pair<T *, size_t> acresize(T * data, size_t const & old_count, size_t const & count, size_t const & alignment) {
    // does not resize down.
    bool allocate = (data == nullptr) || (old_count < count);
    if (allocate) {
        if (data) splash::utils::afree(data);
        data = reinterpret_cast<T*>(splash::utils::acalloc(count * sizeof(T), alignment));  // if bytes == 0 here, that means old data==nullptr..  we return nullptr and 0 bytes.
        return std::make_pair(data, count);
    }  else {
        memset(data, 0, old_count * sizeof(T));
        return std::make_pair(data, old_count);
    }
}
template <typename T>
inline std::pair<T *, size_t> acresize(T * data, size_t const & old_count, size_t const & count) {
    return splash::utils::acresize(data, old_count, count, splash::utils::get_cacheline_size());
}

template <>
inline std::pair<void *, size_t> aresize(void * data, size_t const & old_bytes, size_t const & bytes, size_t const & alignment) {
    // does not resize down.
    bool allocate = (data == nullptr) || (old_bytes < bytes);
    if (allocate) {
        if (data) splash::utils::afree(data);
        data = splash::utils::aalloc(bytes, alignment);  // if bytes == 0 here, that means old data==nullptr..  we return nullptr and 0 bytes.
        return std::make_pair(data, bytes);
    }  else
        return std::make_pair(data, old_bytes);
}
template <>
inline std::pair<void *, size_t> aresize(void * data, size_t const & old_bytes, size_t const & bytes) {
    return splash::utils::aresize(data, old_bytes, bytes, splash::utils::get_cacheline_size());
}

// resize.  no copy of old data.
template <>
inline std::pair<void *, size_t> acresize(void * data, size_t const & old_bytes, size_t const & bytes, size_t const & alignment) {
    // does not resize down.
    bool allocate = (data == nullptr) || (old_bytes < bytes);
    if (allocate) {
        if (data) splash::utils::afree(data);
        data = splash::utils::acalloc(bytes, alignment);  // if bytes == 0 here, that means old data==nullptr..  we return nullptr and 0 bytes.
        return std::make_pair(data, bytes);
    }  else {
        memset(data, 0, old_bytes);
        return std::make_pair(data, old_bytes);
    }
}
template <>
inline std::pair<void *, size_t> acresize(void * data, size_t const & old_bytes, size_t const & bytes) {
    return splash::utils::acresize(data, old_bytes, bytes, splash::utils::get_cacheline_size());
}



// like realloc.  does not scale down.  want bytes==0 to generate an exception.  old pointer may be deallocated, but always set to null
template <typename T>
inline std::pair<T *, size_t> arealloc(T * data, size_t const & old_count, size_t const & count, size_t const & alignment) {
    // does not resize down.
    bool allocate = (data == nullptr) || (old_count < count);
    if (allocate) {
        // all other cases, alloc.  
        T * output = reinterpret_cast<T*>(splash::utils::aalloc(count * sizeof(T), alignment));
        if (data)  {
            memcpy(output, data, old_count * sizeof(T));  // including when old_bytes == 0
            splash::utils::afree(data);   // free the old data.
        }
        return std::make_pair(output, count);
    } else {
        return std::make_pair(data, old_count);
    }
}
template <typename T>
inline std::pair<T *, size_t> arealloc(T * data, size_t const & old_count, size_t const & count) {
    return splash::utils::arealloc(data, old_count, count, splash::utils::get_cacheline_size());
}
template <typename T>
inline std::pair<T *, size_t> arecalloc(T * data, size_t const & old_count, size_t const & count, size_t const & alignment) {
    bool allocate = (data == nullptr) || (old_count < count);
    if (allocate) {
        // all other cases, alloc.  
        T * output = reinterpret_cast<T*>(splash::utils::acalloc(count * sizeof(T), alignment));
        if (data)  {
            memcpy(output, data, old_count * sizeof(T));  // including when old_count == 0
            splash::utils::afree(data);   // free the old data.
        }
        return std::make_pair(output, count);
    } else {
        memset(data + count, 0, (old_count - count) * sizeof(T));
        return std::make_pair(data, old_count);
    }
}
template <typename T>
inline std::pair<T *, size_t> arecalloc(T * data, size_t const & old_count, size_t const & count) {
    return splash::utils::arecalloc(data, old_count, count, splash::utils::get_cacheline_size());
}


template <>
inline std::pair<void *, size_t> arealloc(void * data, size_t const & old_bytes, size_t const & bytes, size_t const & alignment) {
    // does not resize down.
    unsigned char* d;
    size_t s;
    std::tie(d, s) = arealloc(reinterpret_cast<unsigned char*>(data), old_bytes, bytes, alignment);
    return std::make_pair(reinterpret_cast<void *>(d), s);
}
template <>
inline std::pair<void *, size_t> arealloc(void * data, size_t const & old_bytes, size_t const & bytes) {
    return splash::utils::arealloc(data, old_bytes, bytes, splash::utils::get_cacheline_size());
}

// like realloc, but also init remaining to 0
template <>
inline std::pair<void *, size_t> arecalloc(void * data, size_t const & old_bytes, size_t const & bytes, size_t const & alignment) {
    // does not resize down.
    unsigned char* d;
    size_t s;
    std::tie(d, s) = arecalloc(reinterpret_cast<unsigned char*>(data), old_bytes, bytes, alignment);
    return std::make_pair(reinterpret_cast<void *>(d), s);
}
// like realloc, but also init remaining to 0
template <>
inline std::pair<void *, size_t> arecalloc(void * data, size_t const & old_bytes, size_t const & bytes) {
    return splash::utils::arecalloc(data, old_bytes, bytes, splash::utils::get_cacheline_size());
}




// ------- 2D version  ----------------

inline void* aalloc_2D(size_t const & rows, size_t const & col_bytes, size_t const & align) {
    return splash::utils::aalloc(rows * splash::utils::get_aligned_size(col_bytes, align), align);
}
inline void* aalloc_2D(size_t const & rows, size_t const & col_bytes) {
    return splash::utils::aalloc_2D(rows, col_bytes, splash::utils::get_cacheline_size());    
}

inline void* acalloc_2D(size_t const & rows, size_t const & col_bytes, size_t const & align) {
    return splash::utils::acalloc(rows * splash::utils::get_aligned_size(col_bytes, align), align);
}
inline void* acalloc_2D(size_t const & rows, size_t const & col_bytes) {
    return splash::utils::acalloc_2D(rows, col_bytes, splash::utils::get_cacheline_size());    
}


// resize.  no copy of old data.
inline std::pair<void *, size_t> aresize_2D(void * data, size_t const & old_rows, size_t const & old_col_bytes, 
    size_t const & rows, size_t const & col_bytes, size_t const & align) {
    return splash::utils::aresize(data, old_rows * splash::utils::get_aligned_size(old_col_bytes, align), 
        rows * splash::utils::get_aligned_size(col_bytes, align), align);
}
inline std::pair<void *, size_t> aresize_2D(void * data, size_t const & old_rows, size_t const & old_col_bytes, 
    size_t const & rows, size_t const & col_bytes) {
    return splash::utils::aresize_2D(data, old_rows, old_col_bytes, rows, col_bytes, splash::utils::get_cacheline_size());
}

inline std::pair<void *, size_t> acresize_2D(void * data, size_t const & old_rows, size_t const & old_col_bytes, 
    size_t const & rows, size_t const & col_bytes, size_t const & align) {
    return splash::utils::acresize(data, old_rows * splash::utils::get_aligned_size(old_col_bytes, align), 
        rows * splash::utils::get_aligned_size(col_bytes, align), align);
}
inline std::pair<void *, size_t> acresize_2D(void * data, size_t const & old_rows, size_t const & old_col_bytes, 
    size_t const & rows, size_t const & col_bytes) {
    return splash::utils::acresize_2D(data, old_rows, old_col_bytes, rows, col_bytes, splash::utils::get_cacheline_size());
}




inline void memmove_2D(void * src, void * dest, size_t const & rows, size_t const & src_col_bytes, size_t const & dest_col_bytes) {
    size_t min_col_bytes = std::min(src_col_bytes, dest_col_bytes);
    unsigned char * s = reinterpret_cast<unsigned char *>(src);
    unsigned char * d = reinterpret_cast<unsigned char *>(dest);
    for (size_t r = 0; r < rows; ++r,
        s += src_col_bytes, d += dest_col_bytes ) {
            memmove(d, s, min_col_bytes);
    }
}

inline std::pair<void *, size_t> arealloc_2D(void * data, size_t const & old_rows, size_t const & old_col_bytes, 
    size_t const & rows, size_t const & col_bytes, size_t const & align) {
    // col size aligned.
    size_t a_old_col_bytes = splash::utils::get_aligned_size(old_col_bytes, align);
    size_t a_col_bytes = splash::utils::get_aligned_size(col_bytes, align);
    // check total size:
    size_t old_bytes = old_rows * a_old_col_bytes;
    size_t bytes = rows * a_col_bytes;

    if (bytes == 0) {
        PRINT_ERR("%s:%d: aligned realloc_2D with row or col == 0\n", __FILE__, __LINE__);
        throw std::logic_error("arealloc with row or col == 0");
    }
    bool allocate = (data == nullptr) || (old_bytes < bytes) || (a_old_col_bytes < a_col_bytes);  // second condition requires memmove in reverse to expand. simpler to allocate.
    size_t min_rows = std::min(old_rows, rows);
    
    if (allocate) {
        void * output = splash::utils::aalloc(bytes, align);    
        // if has data to copy, do it.  (reuse existing only when allowed.)
        // now free data and set to null
        if (data) {
            memmove_2D(data, 
                output, 
                min_rows, a_old_col_bytes, a_col_bytes);
            splash::utils::afree(data);
        }  
        return std::make_pair(output, bytes);
    } else {     //   (data != nullptr) && (old_bytes >= bytes) && (a_old_col_bytes < a_col_bytes)
        // sufficient space, and shorter or equal column width (compacting or no copy), so don't need to allocate.
        if  (a_old_col_bytes > a_col_bytes) {  // no copy needed if col size same.
            memmove_2D(data, 
                data, 
                min_rows, a_old_col_bytes, a_col_bytes);
        } // if old col > new column, handled by memmove_2D below.
        return std::make_pair(data, old_bytes);
    }
}
inline std::pair<void *, size_t> arealloc_2D(void * data, size_t const & old_rows, size_t const & old_col_bytes, 
    size_t const & rows, size_t const & col_bytes) {
    return splash::utils::arealloc_2D(data, old_rows, old_col_bytes, rows, col_bytes, splash::utils::get_cacheline_size());
}

inline std::pair<void *, size_t> arecalloc_2D(void * data, size_t const & old_rows, size_t const & old_col_bytes, 
    size_t const & rows, size_t const & col_bytes, size_t const & align) {

    // col size aligned.
    size_t a_old_col_bytes = splash::utils::get_aligned_size(old_col_bytes, align);
    size_t a_col_bytes = splash::utils::get_aligned_size(col_bytes, align);
    // check total size:
    size_t old_bytes = old_rows * a_old_col_bytes;
    size_t bytes = rows * a_col_bytes;
    if (bytes == 0) {
        PRINT_ERR("%s:%d: aligned realloc_2D with row or col == 0\n", __FILE__, __LINE__);
        throw std::logic_error("arealloc with row or col == 0");
    }
    bool allocate = (data == nullptr) || (old_bytes < bytes) || (a_old_col_bytes < a_col_bytes);  // second condition requires memmove in reverse to expand. simpler to allocate.
    size_t min_rows = std::min(old_rows, rows);
    size_t data_bytes = 0;

    if (allocate) {
        void *output = splash::utils::acalloc(bytes, align);    
        // if has data to copy, do it.  (reuse existing only when allowed.)
        if (data) {
            memmove_2D(data, 
                output, 
                min_rows, a_old_col_bytes, a_col_bytes);
            splash::utils::afree(data);
        }
        return std::make_pair(output, bytes);
        // now free data and set to null
    } else {
        // sufficient space, and shorter or equal column width (compacting or no copy), so don't need to allocate.
        if  (a_old_col_bytes > a_col_bytes) {  // no copy needed if col size same.
            memmove_2D(data, 
                data, 
                min_rows, a_old_col_bytes, a_col_bytes);
        } // if old col > new column, handled by memmove_2D below.
        data_bytes = min_rows * a_col_bytes;
        memset(reinterpret_cast<unsigned char*>(data) + data_bytes, 0, old_bytes - data_bytes);
        return std::make_pair(data, old_bytes);
    }
    // output is never null in this setup.
}
inline std::pair<void *, size_t> arecalloc_2D(void * data, size_t const & old_rows, size_t const & old_col_bytes, 
    size_t const & rows, size_t const & col_bytes) {
    return splash::utils::arecalloc_2D(data, old_rows, old_col_bytes, rows, col_bytes, splash::utils::get_cacheline_size());    
}




}}