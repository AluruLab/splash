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
        inline OT operator()(IT const * first, IT const * second, size_t const & count) {};
};


// Vector generator.
template <typename OT>
class N2VOp {
    public:
        inline void operator()(size_t const & count,
            OT * out_vector) {};
};



// vector -> vector operator.
template <typename IT, typename OT>
class V2VOp {
    public:
        inline void operator()(IT const * in_vector, 
            size_t const & count,
            OT * out_vector) {};
};

// Matrix generator.
template <typename OT>
class N2MOp {
    public:
        inline void operator()(
            size_t const & rows, size_t const & cols, size_t const & stride_bytes,
            OT * out_matrix) {};
};

// matrix -> matrix operator.
template <typename IT, typename OT>
class M2MOp {
    public:
        inline void operator()(IT const * in_matrix, 
            size_t const & rows, size_t const & cols, size_t const & stride_bytes,
            OT * out_matrix) {};
};


}}