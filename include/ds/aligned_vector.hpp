/*
 *  aligned_vector.hpp
 *
 *  Created on: June 11, 2020
 *  Author: Tony Pan
 *  Affiliation: Institute for Data Engineering and Science
 *  			Georgia Institute of Technology, Atlanta, GA 30332
 */

#pragma once

#include "utils/memory.hpp"

namespace splash { namespace ds { 

template<typename FloatType>
class aligned_vector {
    public:
        using reference = FloatType &;
        using const_reference = FloatType const &;
        using pointer = FloatType *;
        using const_pointer = FloatType const *;
        using iterator = pointer;
        using const_iterator = const_pointer;

    protected:
        void* _data;
        size_t _cols;   // vector length
        size_t _align;  // alignment
        bool manage;   // owning _data or not.

    private:
        size_t bytes;

    public:
        aligned_vector() : 
            _data(nullptr), _cols(0), _align(splash::utils::get_cacheline_size()), manage(true) {}

        // construct, optionally with allocated data.
        // alignment of 0 indicates: use system's cacheline size.
        aligned_vector(size_t const & cols, size_t const & align = 0,
                        void* data = nullptr, bool const & copy=true) :
            _cols(cols), _align(align == 0 ? splash::utils::get_cacheline_size() : align),
            bytes(splash::utils::get_aligned_size(cols * sizeof(FloatType), _align)),
            manage(copy)
        {
            if (manage) {
                _data = splash::utils::aligned_alloc(bytes, _align);  // total size is multiple of alignment.
            }
            if (data)
                if (copy)
                   memcpy(_data, data, bytes);
                else _data = data;
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
                    if (_data) splash::utils::aligned_free(_data);
                    _data = splash::utils::aligned_alloc(other.allocated(), other._align);  // total size is multiple of alignment.
                memcpy(_data, other._data, other.allocated());
            }
            _cols = other._cols;
            _align = other._align;
            bytes = other.bytes;
            manage = other.manage;
        }
        // move constructor.  take ownership.
        aligned_vector(aligned_vector && other) : aligned_vector() {
            std::swap(_data, other._data);
            std::swap(_cols, other._cols);
            std::swap(_align, other._align);
            std::swap(bytes, other.bytes);
            std::swap(manage, other.manage);
        }
        aligned_vector & operator=(aligned_vector && other) {
            std::swap(_data, other._data);
            std::swap(_cols, other._cols);
            std::swap(_align, other._align);
            std::swap(bytes, other.bytes);
            std::swap(manage, other.manage);
        }


        ~aligned_vector() {
            if (_data && manage) {
                splash::utils::aligned_free(_data);
            }
            _data = nullptr;
        }


        inline size_t size() const { return _cols; }
        inline size_t allocated() const { return bytes; }

        inline pointer data(size_t const & idx = 0) noexcept { return reinterpret_cast<pointer>(_data) + idx; }
        inline const_pointer data(size_t const & idx = 0) const noexcept { return reinterpret_cast<pointer>(_data) + idx; }

        // data value accessor
        inline reference operator[](size_t idx) { return *(data(idx)); }
        inline const_reference operator[](size_t idx) const { return *(data(idx)); }
        inline reference operator()(size_t idx) { return *(data(idx)); }
        inline const_reference operator()(size_t idx) const { return *(data(idx)); }

        inline reference at(size_t idx) { return *(data(idx)); }
        inline const_reference at(size_t idx) const { return *(data(idx)); }

        inline explicit operator FloatType*() { return reinterpret_cast<pointer>(_data); }
        inline explicit operator FloatType*() const { return reinterpret_cast<pointer>(_data); }

        // shallow copy?
        aligned_vector<FloatType> deep_copy() {
            aligned_vector<FloatType> output(_cols, _align);
            memcpy(output._data, _data, allocated());
        }


        void print() {
            for (size_t i = 0; i < _count; ++i){
                printf("%f ", _data[i]);
            }
            printf("\n");
        }


};



}}