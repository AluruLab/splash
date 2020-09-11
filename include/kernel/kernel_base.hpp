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
enum DIM    : int { ROW = 0, COLUMN = 1};

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
class kernel_base {
    public:
        kernel_base() {}
        virtual ~kernel_base() {}
        void copy_parameters(kernel_base const & other)  {}
};

// vector + vector -> scalar operator.
template <typename IT, typename OT, int DEG1, int DEG2 = DEG1>
class inner_product;

template <typename IT, typename OT>
class inner_product<IT, OT, DEGREE::VECTOR, DEGREE::VECTOR> : public kernel_base {
    public:
        
        using InputType = IT;
        using OutputType = OT;

        virtual ~inner_product() {};
        inline virtual OT operator()(IT const * first, IT const * second, size_t const & count) const = 0;
    protected:
        inline virtual void initialize(size_t const & count) {};
};

// vector + vector -> scalar operator.
template <typename IT, typename OT, int DEG1, int DEG2 = DEG1>
class inner_product_pos;

template <typename IT, typename OT>
class inner_product_pos<IT, OT, DEGREE::VECTOR, DEGREE::VECTOR> : public kernel_base {
    public:
        
        using InputType = IT;
        using OutputType = OT;

        virtual ~inner_product_pos() {};
        inline virtual OT operator()(size_t const & r, size_t const & c, IT const * first, IT const * second, size_t const & count) const = 0;
    protected:
        inline virtual void initialize(size_t const & count) {};
};


// template <typename IT, typename OT>
// class inner_product<IT, OT, DEGREE::MATRIX, DEGREE::VECTOR> : public kernel_base {
//     public:
        
//         using InputType = IT;
//         using OutputType = OT;

//         virtual ~inner_product() {};
//         inline virtual void operator()(IT const * first, IT const * second, size_t const & row, size_t const & col,
//             size_t const & stride_bytes1, OT * out) const = 0;
//     protected:
//         inline virtual void initialize(size_t const & row, size_t const & col) {};
// };


// template <typename IT, typename OT>
// class inner_product<IT, OT, DEGREE::MATRIX, DEGREE::MATRIX> : public kernel_base {
//     public:
        
//         using InputType = IT;
//         using OutputType = OT;

//         virtual ~inner_product() {};
//         inline virtual void operator()(IT const * first, IT const * second, size_t const & row, size_t const & rc, size_t const & col,
//             size_t const & stride_bytes1, size_t const & stride_bytes2, OT * out) const = 0;
//     protected:
//         inline virtual void initialize(size_t const & row, size_t const & rc, size_t const & col) {};
// };


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
class generate<OT, DEGREE::VECTOR>  : public kernel_base {
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
class generate<OT, DEGREE::MATRIX>  : public kernel_base {
    public:
        using InputType = void;
        using OutputType = OT;
        virtual ~generate() {};
        inline virtual void operator()(
            size_t const & rows, size_t const & cols, size_t const & stride_bytes,
            OT * out_matrix) const = 0;
    protected:
        inline virtual void initialize(size_t const & rows, size_t const & cols) {};

};

// binary operations
template <typename IT, typename IT1, typename OT, int DEG>
class binary_op;

template <typename IT, typename IT1, typename OT>
class binary_op<IT, IT1, OT, DEGREE::VECTOR> : public kernel_base {
    public:
        using OutputType = OT;
        virtual ~binary_op() {};
        inline virtual void operator()(IT const * in, IT1 const * aux1, size_t const & count, OT * out) const = 0;
};


// binary operations
template <typename IT, typename IT1, typename IT2, typename OT, int DEG>
class ternary_op;

template <typename IT, typename IT1, typename IT2, typename OT>
class ternary_op<IT, IT1, IT2, OT, DEGREE::SCALAR> : public kernel_base {
    public:
        using OutputType = OT;
        virtual ~ternary_op() {};
        inline virtual OT operator()(IT const & in, IT1 const & aux1, IT2 const & aux2) const = 0;
};

// vector -> vector operator.
template <typename IT, typename OT, int DEG>
class transform;


template <typename IT, typename OT>
class transform<IT, OT, DEGREE::SCALAR>  : public kernel_base {
    public:
        using InputType = IT;
        using OutputType = OT;
        virtual ~transform() {};
        inline virtual OT operator()(IT const & in) const = 0;
};

template <typename IT, typename OT>
class transform<IT, OT, DEGREE::VECTOR>  : public kernel_base {
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
class transform<IT, OT, DEGREE::MATRIX>  : public kernel_base {
    public:
        using InputType = IT;
        using OutputType = OT;
        virtual ~transform() {};
        inline virtual void operator()(IT const * in_matrix, 
            size_t const & rows, size_t const & cols, size_t const & stride_bytes,
            OT * out_matrix) const = 0;
    protected:
        inline virtual void initialize(size_t const & rows, size_t const & cols) {};

};

// general reduction template
// REDUC_PER indicates which dimenion to preserve.  reduction is done across the other dims.
template <typename IT, typename OT, int IN_DEG, int REDUC_PER = DIM::ROW>
class reduce; 

template <typename IT, typename OT>
class reduce<IT, OT, DEGREE::VECTOR, DIM::ROW>  : public kernel_base {
    public:
        using InputType = IT;
        using OutputType = OT;
         virtual ~reduce() {};
       inline virtual OT operator()(IT const * in_vector, 
            size_t const & count) const = 0;
    protected:
        inline virtual void initialize(size_t const & count) {};
};

template <typename IT, typename OT>
class reduce<IT, OT, DEGREE::VECTOR, DIM::COLUMN>  : public kernel_base {
    public:
        using InputType = IT;
        using OutputType = OT;
         virtual ~reduce() {};
        inline virtual void operator()(IT const * in, 
            size_t const & count,
            OT * aux) const = 0;
    protected:
        inline virtual void initialize(size_t const & count) {};
};


// // this will always be row-wise reduction.
// template <typename IT, typename OT>
// class reduce<IT, OT, DEGREE::MATRIX, DEGREE::VECTOR>  : public kernel_base {
//     public:
//         using InputType = IT;
//         using OutputType = OT;
//          virtual ~reduce() {};
//        inline virtual void operator()(IT const * in, 
//             size_t const & row, size_t const & col, OT * out) const = 0;
//     protected:
//         inline virtual void initialize(size_t const & row, size_t const & col) {};
// };


// template <typename IT, typename OT>
// class reduce<IT, OT, DEGREE::MATRIX, DEGREE::SCALAR>  : public kernel_base {
//     public:
//         using InputType = IT;
//         using OutputType = OT;
//          virtual ~reduce() {};
//        inline virtual OT operator()(IT const * in_vector, 
//             size_t const & row, size_t const & col) const = 0;
//     protected:
//         inline virtual void initialize(size_t const & row, size_t const & col) {};
// };




}}