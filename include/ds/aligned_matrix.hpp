/*
 *  aligned_matrix.hpp
 *
 *  Created on: June 11, 2020
 *  Author: Tony Pan
 *  Affiliation: Institute for Data Engineering and Science
 *  			Georgia Institute of Technology, Atlanta, GA 30332
 */

#pragma once  // instead of #ifndef ...

#include <stdlib.h> // aligned_alloc
#include <new>      // bad_alloc

#include "ds/common.hpp"
#include "io/error_handler.hpp"

namespace splash { namespace ds { 


template<typename FloatType>
class aligned_matrix {
    public:
        using reference = FloatType &;
        using const_reference = FloatType const &;
        using pointer = FloatType *;
        using const_pointer = FloatType const *;
        using iterator = pointer;
        using const_iterator = const_pointer;

    protected:
        void* _data; // internal pointer.
        size_t _rows;   // number of vectors in the data
        size_t _cols;   // vector length
        size_t _align;  // alignment
        size_t _bytes_per_row;   // assume row major
        bool manage;   // owning _data or not.

        inline pointer _get_row(size_t const & row = 0) {
            return reinterpret_cast<pointer>(_data + row * _bytes_per_row);
        }
        inline const_pointer _get_row(size_t const & row = 0) const {
            return reinterpret_cast<const_pointer>(_data + row * _bytes_per_row);
        }

    public:
        aligned_matrix() : 
            _data(nullptr), _rows(0), _cols(0), _align(SPL_CACHELINE_WIDTH), _bytes_per_row(0), manage(true) {}
        
        // construct, optionally with allocated data.
        aligned_matrix(size_t const & rows, size_t const & cols, size_t const & align = SPL_CACHELINE_WIDTH,
                        void* data = nullptr, bool const & copy=true) :
            _rows(rows), _cols(cols), _align(align),
            _bytes_per_row((cols * sizeof(FloatType) + static_cast<size_t>(align - 1)) & ~(static_cast<size_t>(align - 1))),
            manage(copy)
        {
            if (manage) {
                _data = aligned_alloc(_align, _bytes_per_row * _rows);  // total size is multiple of alignment.

                if (!_data) {
                    splash::io::print_err("Memory allocation failed at line %d in file %s\n", __LINE__, __FILE__);
                    throw std::bad_alloc();
                }
            }
            if (data)
                if (copy)
                   memcpy(_data, data, _rows * _bytes_per_row);
                else _data = data;
        }
        // copy constructor and assignment.  deep copy.
        aligned_matrix(aligned_matrix const & other) : 
            aligned_matrix(other._rows, other._cols, other._align, other._data, other.manage) {}
        aligned_matrix & operator=(aligned_matrix const & other) {
            if (!other.manage) 
                _data = other._data;
            else {
                // not same size.  free and reallocate.
                if (allocated() != other.allocated()) {
                    if (_data) free(_data);
                    _data = aligned_alloc(other._align, other.allocated());  // total size is multiple of alignment.
                    if (!_data) {
                        splash::io::print_err("Memory allocation failed at line %d in file %s\n", __LINE__, __FILE__);
                        throw std::bad_alloc();
                    }
                memcpy(_data, other._data, other.allocated());
            }
            _rows = other._rows;
            _cols = other._cols;
            _align = other._align;
            _bytes_per_row = other._bytes_per_row;
            manage = other.manage;
        }
        // move constructor.  take ownership.
        aligned_matrix(aligned_matrix && other) : aligned_matrix() {
            std::swap(_data, other._data);
            std::swap(_rows, other._rows);
            std::swap(_cols, other._cols);
            std::swap(_align, other._align);
            std::swap(_bytes_per_row, other._bytes_per_row);
            std::swap(manage, other.manage);
        }
        aligned_matrix & operator=(aligned_matrix && other) {
            std::swap(_data, other._data);
            std::swap(_rows, other._rows);
            std::swap(_cols, other._cols);
            std::swap(_align, other._align);
            std::swap(_bytes_per_row, other._bytes_per_row);
            std::swap(manage, other.manage);
        }


        ~aligned_matrix() {
            if (_data && manage) {
                free(_data);
            }
            _data = nullptr;
        }

        inline size_t size() const {  return _rows * _cols; }
        inline size_t allocated() const {
            return _rows * _bytes_per_row;
        }

        inline size_t rows() const {  return _rows; }
        inline size_t columns() const {  return _cols; }
        inline size_t row_allocated() const {
            return _bytes_per_row;
        }


        inline pointer data(size_t const & row = 0, size_t const & col = 0) noexcept { 
            return _get_row(row) + col; 
        }
        inline const_pointer data(size_t const & row = 0, size_t const & col = 0) const noexcept {
            return _get_row(row) + col; 
        }

        // data value accessor
        inline reference operator()(size_t const & row = 0, size_t const & col = 0) { 
            return _get_row(row)[col]; 
        }
        inline const_reference operator()(size_t const & row = 0, size_t const & col = 0) const { 
            return _get_row(row)[col]; 
        }

        inline reference at(size_t const & row = 0, size_t const & col = 0) { 
            return _get_row(row)[col]; 
        }
        inline const_reference at(size_t const & row = 0, size_t const & col = 0) const { 
            return _get_row(row)[col]; 
        }

        inline explicit operator FloatType*() { return reinterpret_cast<pointer>(_data); }
        inline explicit operator FloatType*() const { return reinterpret_cast<const_pointer>(_data); }


	    /*transpose input matrix*/
	    aligned_matrix<FloatType> transpose() {
            aligned_matrix<FloatType> output(_cols, _rows, _align);
        
            FloatType *in;
            /*transpose the matrix*/
            for(size_t row = 0; row < _rows; ++row){
                in = this->_get_row(row);
                for(size_t col = 0; col < _cols; ++col) {
                    output(col, row) = in[col];
                }
            }

        }
        // shallow copy?
        aligned_matrix<FloatType> deep_copy() {
            aligned_matrix<FloatType> output(_rows, _cols, _align);
            memcpy(output._data, _data, allocated());
        }

        void print() {
            FloatType * d;
            for (size_t row = 0; row < _rows; ++row){
                d = this->_get_row(row);
                for (size_t col = 0; col < _cols; ++col) {
                    printf("%f ", d[col]);
                }
                printf("\n");
            }
        }

};



}}
