/*
 *  kernel_base.hpp
 *
 *  Created on: June 11, 2020
 *  Author: Tony Pan
 *  Affiliation: Institute for Data Engineering and Science
 *  			Georgia Institute of Technology, Atlanta, GA 30332
 */

#pragma once

namespace splash { namespace kernel { 

enum DEGREE : int { SCALAR = 0, VECTOR = 1, MATRIX = 2};

// differs from patterns in that kernel is meant to be single thread, whereas patterns deal with multinode/core

// vector + vector -> scalar operator.
template <typename IT, typename OT, int DEG>
class inner_product;

template <typename IT, typename OT>
class inner_product<IT, OT, DEGREE::VECTOR> {
    public:
        using InputType = IT;
        using OutputType = OT;
        inline OT operator()(IT const * first, IT const * second, size_t const & count) const {};
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
class generate<OT, DEGREE::VECTOR> {
    public:
        using InputType = void;
        using OutputType = OT;
        inline void operator()(size_t const & count,
            OT * out_vector) const {};
};

// Matrix generator.
template <typename OT>
class generate<OT, DEGREE::MATRIX> {
    public:
        using InputType = void;
        using OutputType = OT;
        inline void operator()(
            size_t const & rows, size_t const & cols, size_t const & stride_bytes,
            OT * out_matrix) const {};
};


// vector -> vector operator.
template <typename IT, typename OT, int DEG>
class transform;

template <typename IT, typename OT>
class transform<IT, OT, DEGREE::VECTOR> {
    public:
        using InputType = IT;
        using OutputType = OT;
        inline void operator()(IT const * in_vector, 
            size_t const & count,
            OT * out_vector) const {};
};

// matrix -> matrix operator.
template <typename IT, typename OT>
class transform<IT, OT, DEGREE::MATRIX> {
    public:
        using InputType = IT;
        using OutputType = OT;
        inline void operator()(IT const * in_matrix, 
            size_t const & rows, size_t const & cols, size_t const & stride_bytes,
            OT * out_matrix) const {};
};

// vector -> scalar operator
template <typename IT, typename OT, int DEG>
class reduce; 

template <typename IT, typename OT>
class reduce<IT, OT, DEGREE::VECTOR> {
    public:
        using InputType = IT;
        using OutputType = OT;
        inline OT operator()(IT const * in_vector, 
            size_t const & count) const {};
};





}}