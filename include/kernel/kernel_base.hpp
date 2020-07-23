/*
 *  kernel_base.hpp
 *
 *  Created on: June 11, 2020
 *  Author: Tony Pan
 *  Affiliation: Institute for Data Engineering and Science
 *  			Georgia Institute of Technology, Atlanta, GA 30332
 */

#pragma once

#ifdef USE_OPENMP
#include <omp.h>
#endif

#include "utils/memory.hpp"

namespace splash { namespace kernel { 

enum DEGREE : int { SCALAR = 0, VECTOR = 1, MATRIX = 2};

/**
 * KERNEL BASE CLASSES.
 * Set up as virtual super classes - while these are core kernels, computation is per vector, which is assumed long, so cost of virtual function lookup MAY not be bad.
 *      allow enforcement of signature for public operator() and protected initialize()
 *      all class must have virtual destructor
 * 
 * THREAD SAFETY:
 * 1. kernels (these, and their subclasses) should be considered stateless.  I.e. no information is persisted over multiple calls.
 * 2. For kernels with instance level buffers (instead of allocating and freeing on every call), need separate buffer for each thread.
 *      A. each thread should have its own instance of kernel. -> Pattern classes should instantiate on demand, 
 *      B. kernel parameters should be copied. -> Pattern instance should be passed a "template" instance of kernel to be copied.
 *      C. kernel class must provide copy constructor at least.
 *      D. kernel class must provide "initialize()" to initialize internal buffer as needed.
 *    With B, kernel parameters are encapsulated from the Patterns class definition, which can remain largely generic.
 * 
 *   An unordered map cannot be used for parameter passing since the value types may not be uniform.
 */
// differs from patterns in that kernel is meant to be single thread, whereas patterns deal with multinode/core

// Rather than defining some all encompasing data structure for passing kernel parameters during copy,
// do:  define parameterized constructor, default constructor, and copy_parameters.
//    create template instance with the correct paramater (e.g. in main thread)
//    in thread, create with default constructor and copy_parameters.
class paramed_kernel {
    public:
        paramed_kernel() {}
        virtual ~paramed_kernel() {}
        void copy_parameters(paramed_kernel const & other)  {}
};



template <typename T, int NUM_BUFFERS = 1>
class buffered_kernel;

template <typename T>
class buffered_kernel<T, 2> {
    protected:
        mutable T * buffer;
        mutable T * aux;
        mutable size_t vecSize;

    public:
        buffered_kernel() : buffer(nullptr), aux(nullptr), vecSize(0) {
// #ifdef USE_OPENMP
// 			fprintf(stdout, "construct BUFFERED_KERNEL, thread %d\n", omp_get_thread_num());
// #endif  
		}
		virtual ~buffered_kernel() {
			if (buffer) {
				splash::utils::afree(buffer);
				buffer = nullptr;
			}
			if (aux) {
				splash::utils::afree(aux);
				aux = nullptr;
			}
            vecSize = 0;
		}
		buffered_kernel(buffered_kernel const & other) : vecSize(other.vecSize)  {
            buffer = reinterpret_cast<T*>(splash::utils::aalloc(vecSize * sizeof(T)));
            if (other.buffer && other.vecSize) {
                memcpy(buffer, other.buffer, other.vecSize * sizeof(T));
            }
            aux = reinterpret_cast<T*>(splash::utils::aalloc(vecSize * sizeof(T)));
            if (other.aux && other.vecSize) {
                memcpy(aux, other.aux, other.vecSize * sizeof(T));
            }
        }
		buffered_kernel& operator=(buffered_kernel const & other) {
            std::tie(buffer, vecSize) = splash::utils::acresize(buffer, vecSize, other.vecSize);
            std::tie(aux, vecSize) = splash::utils::acresize(aux, vecSize, other.vecSize);
            if (buffer && other.buffer) memcpy(buffer, other.buffer, vecSize * sizeof(T));
            if (aux && other.aux) memcpy(aux, other.aux, vecSize * sizeof(T));
        }
		buffered_kernel(buffered_kernel && other) : buffer(std::move(other.buffer)), aux(std::move(other.aux)), vecSize(other.vecSize) {
            other.buffer = nullptr;
            other.aux = nullptr;
            other.vecSize = 0;
        } 
		buffered_kernel& operator=(buffered_kernel && other) {
            if (buffer) splash::utils::afree(buffer);
            buffer = other.buffer; other.buffer = nullptr;
            if (aux) splash::utils::afree(aux);
            aux = other.aux; other.aux = nullptr;
            vecSize = other.vecSize; other.vecSize = 0;
        }

		virtual void resize(size_t const & size) const {
            std::tie(buffer, vecSize) = splash::utils::arecalloc(buffer, vecSize, size);
            std::tie(aux, vecSize) = splash::utils::arecalloc(aux, vecSize, size);
		}
};

// default constructor creates a null buffer.   allocate during run, by first calling resize.
template <typename T>
class buffered_kernel<T, 1> {
    protected:
        mutable T * buffer;
        mutable size_t vecSize;

    public:
        buffered_kernel() : buffer(nullptr), vecSize(0) {
// #ifdef USE_OPENMP
// 			fprintf(stdout, "construct BUFFERED_KERNEL, thread %d\n", omp_get_thread_num());
// #endif  
		}
		virtual ~buffered_kernel() {
			if (buffer) {
				splash::utils::afree(buffer);
				buffer = nullptr;
                vecSize = 0;
			}
		}
		buffered_kernel(buffered_kernel const & other) : vecSize(other.vecSize)  {
            buffer = reinterpret_cast<T*>(splash::utils::aalloc(vecSize * sizeof(T)));
            if (other.buffer && other.vecSize) {
                memcpy(buffer, other.buffer, other.vecSize * sizeof(T));
            }
        }
		buffered_kernel& operator=(buffered_kernel const & other) {
            std::tie(buffer, vecSize) = splash::utils::acresize(buffer, vecSize, other.vecSize);
            if (buffer && other.buffer)
                memcpy(buffer, other.buffer, vecSize * sizeof(T));
        }
		buffered_kernel(buffered_kernel && other) : buffer(std::move(other.buffer)), vecSize(other.vecSize) {
            other.buffer = nullptr;
            other.vecSize = 0;
        } 
		buffered_kernel& operator=(buffered_kernel && other) {
            if (buffer) splash::utils::afree(buffer);
            buffer = other.buffer; other.buffer = nullptr;
            vecSize = other.vecSize; other.vecSize = 0;
        }

		virtual void resize(size_t const & size) const {
            std::tie(buffer, vecSize) = splash::utils::arecalloc(buffer, vecSize, size);
		}
};



// vector + vector -> scalar operator.
template <typename IT, typename OT, int DEG>
class inner_product;

template <typename IT, typename OT>
class inner_product<IT, OT, DEGREE::VECTOR> : public paramed_kernel {
    public:
        
        using InputType = IT;
        using OutputType = OT;

        virtual ~inner_product() {};
        inline virtual OT operator()(IT const * first, IT const * second, size_t const & count) const = 0;
    protected:
        inline virtual void initialize(size_t const & count) {};
};

/*
 * TODO: COMMENTED OUT FOR NOW BECAUSE NO GREAT WAY TO  PATTERN THIS YET.
// // (vector, scalar) + (vector, scalar) -> scalar operator.
// template <typename IT, typename OT>
// class VSVS2SOp {
//     public:
//         using InputType = IT;
//         using OutputType = OT;
//         inline OT operator()(IT const * first, IT const & first_aux,
//              IT const * second, IT const & second_aux, 
//              size_t const & count) const {};
// };
*/


// Vector generator.
template <typename OT, int DEG>
class generate;

template <typename OT>
class generate<OT, DEGREE::VECTOR>  : public paramed_kernel {
    public:
        using InputType = void;
        using OutputType = OT;
        virtual ~generate() {};
        inline virtual void operator()(size_t const & count, OT * out_vector) const = 0;
    protected:
        inline virtual void initialize(size_t const & count) {};
};

// Matrix generator.
template <typename OT>
class generate<OT, DEGREE::MATRIX>  : public paramed_kernel {
    public:
        using InputType = void;
        using OutputType = OT;
        virtual ~generate() {};
        inline virtual void operator()(
            size_t const & rows, size_t const & cols, size_t const & stride_bytes,
            OT * out_matrix) const = 0;
    protected:
        inline virtual void initialize(size_t const & cols) {};

};


// vector -> vector operator.
template <typename IT, typename OT, int DEG>
class transform;

template <typename IT, typename OT>
class transform<IT, OT, DEGREE::VECTOR>  : public paramed_kernel {
    public:
        using InputType = IT;
        using OutputType = OT;
        virtual ~transform() {};
        inline virtual void operator()(IT const * in_vector, 
            size_t const & count,
            OT * out_vector) const = 0;
    protected:
        inline virtual void initialize(size_t const & count) {};

};

// matrix -> matrix operator.
template <typename IT, typename OT>
class transform<IT, OT, DEGREE::MATRIX>  : public paramed_kernel {
    public:
        using InputType = IT;
        using OutputType = OT;
        virtual ~transform() {};
        inline virtual void operator()(IT const * in_matrix, 
            size_t const & rows, size_t const & cols, size_t const & stride_bytes,
            OT * out_matrix) const = 0;
    protected:
        inline virtual void initialize(size_t const & cols) {};

};

// vector -> scalar operator
template <typename IT, typename OT, int DEG>
class reduce; 

template <typename IT, typename OT>
class reduce<IT, OT, DEGREE::VECTOR>  : public paramed_kernel {
    public:
        using InputType = IT;
        using OutputType = OT;
         virtual ~reduce() {};
       inline virtual OT operator()(IT const * in_vector, 
            size_t const & count) const = 0;
    protected:
        inline virtual void initialize(size_t const & count) {};
};





}}