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

template <typename IT, typename OT>
class BinaryVectorOpBase {
    public:
        inline OT operator()(IT const * first, IT const * second, size_t const & count) {};
};



// for operation on a single matrix
template <typename IT, typename OT>
class UnaryMatrixOpBase {
    public:
        inline void operator()(IT const * in_matrix, 
            size_t const & rows, size_t const & cols, size_t const & row_stride,
            OT * out_matrix) {};
};


// for operation on a single matrix
template <typename IT, typename OT>
class UnaryVectorOpBase {
    public:
        inline void operator()(IT const * in_vector, 
            size_t const & count,
            OT * out_vector) {};
};


}}