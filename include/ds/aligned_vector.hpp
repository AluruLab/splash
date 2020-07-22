/*
 *  aligned_vector.hpp
 *
 *  Created on: June 11, 2020
 *  Author: Tony Pan
 *  Affiliation: Institute for Data Engineering and Science
 *  			Georgia Institute of Technology, Atlanta, GA 30332
 */
/* TODO:
 * [ ] use a partition object as dimension descriptor
 */
#pragma once

#include "utils/memory.hpp"
#include <cstring>  // memset, memcpy

// TODO 
// [ ] MPI gather, scatter, allgather, etc.

namespace splash { namespace ds { 

// NOTE: this is a STRICTLY LOCAL data structure.
//       It is NOT thread-safe.
template<typename FloatType>
class aligned_vector {
    public:
        using reference = FloatType &;
        using const_reference = FloatType const &;
        using pointer = FloatType *;
        using const_pointer = FloatType const *;
        using iterator = pointer;
        using const_iterator = const_pointer;

        using data_type = FloatType;
        using size_type = size_t;

    protected:
        unsigned char* _data;
        size_type _cols;   // vector length
        size_t _align;  // alignment
        bool manage;   // owning _data or not.

    private:
        size_type bytes;

    public:
        aligned_vector() : 
            _data(nullptr), _cols(0), _align(splash::utils::get_cacheline_size()), manage(true), bytes(0) {}

        // construct, optionally with allocated data.
        // alignment of 0 indicates: use system's cacheline size.
        aligned_vector(size_type const & cols, size_t const & align = 0,
                        void* data = nullptr, bool const & copy=true) :
            _cols(cols), _align(align == 0 ? splash::utils::get_cacheline_size() : align),
            manage(copy),
            bytes(splash::utils::get_aligned_size(cols * sizeof(FloatType), _align))
        {
            if (manage) {
                _data = reinterpret_cast<unsigned char*>(splash::utils::aalloc(bytes, _align));  // total size is multiple of alignment.
                memset(_data, 0, bytes);
            }
            if (data) {
                if (copy)
                   memcpy(_data, data, bytes);
                else _data = reinterpret_cast<unsigned char*>(data);
            }
        }
        // copy constructor and assignment.  deep copy.
        aligned_vector(aligned_vector const & other) : 
            aligned_vector(other._cols, other._align, other._data, other.manage) {}
        aligned_vector & operator=(aligned_vector const & other) {
            if (!other.manage) 
                _data = other._data;
            else {
                // not same size.  free and reallocate.
                if (allocated() != other.allocated()) {
                    if (_data) splash::utils::afree(_data);
                    _data = reinterpret_cast<unsigned char*>(splash::utils::aalloc(other.allocated(), other._align));  // total size is multiple of alignment.
                }
                memcpy(_data, other._data, other.allocated());
            }
            _cols = other._cols;
            _align = other._align;
            bytes = other.bytes;
            manage = other.manage;

            return *this;
        }
        // move constructor.  take ownership.
        aligned_vector(aligned_vector && other) : _data(other._data),
            _cols(other._cols), _align(other._align), manage(other.manage), bytes(other.bytes) {
                other._data = nullptr;
                other._cols = 0;
                other.bytes = 0;
                other.manage = false;
        }
        aligned_vector & operator=(aligned_vector && other) {
            if (_data && manage) {
                splash::utils::afree(_data);
            }
            _data = other._data;
            manage = other.manage;
            other._data = nullptr;
            other.manage = false;

            _cols = other._cols;
            other._cols = 0;
            
            _align = other._align;
            bytes = other.bytes;
            other.bytes = 0;
            
            return *this;
        }


        ~aligned_vector() {
            if (_data && manage) {
                splash::utils::afree(_data);
            }
            _data = nullptr;
        }

        // resizes. copy data.
        inline void resize(size_type const  & cols)  {
            if (cols == _cols) return;
            // allocate new data.
            unsigned char* data = nullptr;
            size_type b = splash::utils::get_aligned_size(cols * sizeof(FloatType), _align);
            if (cols > 0) {
                data = reinterpret_cast<unsigned char*>(splash::utils::aalloc(b, _align));  // total size is multiple of alignment.
                memset(data, 0, b);
            }
            // copy data if any.
            if (_data) {
                size_type bmin = std::min(b, bytes);
                memcpy(data, _data, bmin);
            }
  
            // swap.
            std::swap(_data, data);
            _cols = cols;
            bytes = b;

            // if not managing, then okay to replace the pointer..  else need to free old data.
            if (data && manage) {
                splash::utils::afree(data);
            }
            // if was not managing, now is.
            manage = true;
        }



        inline size_type size() const { return _cols; }
        inline size_type allocated() const { return bytes; }

        inline pointer data(size_type const & idx = 0) noexcept { return reinterpret_cast<pointer>(_data) + idx; }
        inline const_pointer data(size_type const & idx = 0) const noexcept { return reinterpret_cast<const_pointer>(_data) + idx; }

        // data value accessor
        inline reference operator[](size_type idx) { return *(data(idx)); }
        inline const_reference operator[](size_type idx) const { return *(data(idx)); }
        inline reference operator()(size_type idx) { return *(data(idx)); }
        inline const_reference operator()(size_type idx) const { return *(data(idx)); }

        inline reference at(size_type idx) { return *(data(idx)); }
        inline const_reference at(size_type idx) const { return *(data(idx)); }

        // inline explicit operator FloatType*() { return reinterpret_cast<pointer>(_data); }
        // inline explicit operator const FloatType*() const { return reinterpret_cast<const_pointer>(_data); }

        // shallow copy?
        aligned_vector<FloatType> deep_copy() {
            aligned_vector<FloatType> output(_cols, _align);
            memcpy(output._data, _data, allocated());
        }


        void print(const char* prefix) {
            PRINT_RT("%s ", prefix);
            for (size_type i = 0; i < _cols; ++i){
                PRINT("%f ", data(i));
            }
            PRINT("\n");
        }


};

}}



