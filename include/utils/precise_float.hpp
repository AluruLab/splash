#pragma once 

#include <cstdint>
#include <type_traits>

/*
 * this file contain some convenience classes for high precision math
 * 
 * these can be used for prefix scan operations
 * 
 */

namespace splash { namespace utils {

template <typename AT>  struct widened_type;
template <>  struct widened_type<float> { using type = double; };
template <>  struct widened_type<double> { using type = long double; };
template <>  struct widened_type<int8_t> { using type = int16_t; };
template <>  struct widened_type<int16_t> { using type = int32_t; };
template <>  struct widened_type<int32_t> { using type = int64_t; };
template <>  struct widened_type<int64_t> { using type = long long; };
template <>  struct widened_type<uint8_t> { using type = uint16_t; };
template <>  struct widened_type<uint16_t> { using type = uint32_t; };
template <>  struct widened_type<uint32_t> { using type = uint64_t; };
template <>  struct widened_type<uint64_t> { using type = unsigned long long; };

template <typename AT>
using widened = typename widened_type<AT>::type;


// This is a precise accumulator that can better handle addition and subtraction of different magnitude values.
// this is useful if the count of numbers to accumulate is much larger than the capacity (MAX_FLOAT / average value magnitude)
// for the most part, using widened would be easier (can use built-in casting), use the same space, and uses fewer instructions
template <typename FT>
class precise_float {
        static_assert(::std::is_floating_point<FT>::value, "only floating point types are allowed for precise_float...");

    public:
        FT value;
        FT excess;

        precise_float() = default;
        precise_float(FT const & val) : value(val) {}
        precise_float(FT const & _value, FT const & _excess) : value(_value), excess(_excess) {}

        // copy constructor
        precise_float(precise_float const & other) = default;
        precise_float& operator=(precise_float const & other) = default;

        // move constructor
        precise_float(precise_float && other) = default; 
        precise_float& operator=(precise_float && other) = default; 

        // cast operator
        inline operator FT() const { return value; }

        // ----------- ADDITION -------------------
        // addition.  precise_float + FT
        inline precise_float operator+(FT const & right) {
            FT y = right - this->excess;
            FT t = this->value + y;
            return precise_float(t, (t - this->value) - y);
        }

        inline precise_float & operator+=(FT const & right) {
            FT y = right - this->excess;
            FT t = this->value + y;
            this->excess = (t - this->value) - y;
            this->value = t;
            return *this;
        }

        // addition.  precise_float + precise_float.  correct way to do this?
        inline precise_float& operator+=(precise_float const & right) {
            // first do value
            *this += right.value;
            // then do excess.
            *this += right.excess;
            return *this;
        }
        inline precise_float operator+(precise_float const & right) {
            return precise_float(*this) += right;
        }

        // ----------- negation ----------------
        inline precise_float operator-() {
            return precise_float(-value, -excess);
        }

        // -------------- subtraction -------------
        inline precise_float operator-(FT const & right) {
            return this->operator+(-right);
        }

        inline precise_float & operator-=(FT const & right) {
            return this->operator+=(-right);
        }

        // addition.  precise_float + precise_float.  correct way to do this?
        inline precise_float operator-(precise_float const & right) {
            return this->operator+(-right);
        }

        inline precise_float& operator-=(precise_float const & right) {
            return this->operator+=(-right);
        }

};

// addition.  precise_float + FT
template <typename FT, typename std::enable_if<std::is_floating_point<FT>::value, int>::type = 0>
inline FT operator+(FT const & left, splash::utils::precise_float<FT> const & right) {
    return static_cast<FT>(right + left);
}
// addition.  precise_float + FT
template <typename FT, typename std::enable_if<std::is_floating_point<FT>::value, int>::type = 0>
inline FT operator-(FT const & left, splash::utils::precise_float<FT> const & right) {
    return -(static_cast<FT>(right - left));
}


}}