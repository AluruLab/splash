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

// vector + vector -> scalar operator.
template <typename IT, typename OT>
class VV2SOp {
    public:
        using InputType = IT;
        using OutputType = OT;
        inline OT operator()(IT const * first, IT const * second, size_t const & count) const {};
};

// (vector, scalar) + (vector, scalar) -> scalar operator.
template <typename IT, typename OT>
class VSVS2SOp {
    public:
        using InputType = IT;
        using OutputType = OT;
        inline OT operator()(IT const * first, IT const & first_aux,
             IT const * second, IT const & second_aux, 
             size_t const & count) const {};
};



// Vector generator.
template <typename OT>
class N2VOp {
    public:
        using InputType = void;
        using OutputType = OT;
        inline void operator()(size_t const & count,
            OT * out_vector) const {};
};



// vector -> vector operator.
template <typename IT, typename OT>
class V2VOp {
    public:
        using InputType = IT;
        using OutputType = OT;
        inline void operator()(IT const * in_vector, 
            size_t const & count,
            OT * out_vector) const {};
};

// vector -> scalar operator
template <typename IT, typename OT>
class V2SOp {
    public:
        using InputType = IT;
        using OutputType = OT;
        inline OT operator()(IT const * in_vector, 
            size_t const & count) const {};
};

// Matrix generator.
template <typename OT>
class N2MOp {
    public:
        using InputType = void;
        using OutputType = OT;
        inline void operator()(
            size_t const & rows, size_t const & cols, size_t const & stride_bytes,
            OT * out_matrix) const {};
};

// matrix -> matrix operator.
template <typename IT, typename OT>
class M2MOp {
    public:
        using InputType = IT;
        using OutputType = OT;
        inline void operator()(IT const * in_matrix, 
            size_t const & rows, size_t const & cols, size_t const & stride_bytes,
            OT * out_matrix) const {};
};


}}