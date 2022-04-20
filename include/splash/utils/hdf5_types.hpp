#pragma once

#include <bitset>
#include <type_traits>
#include <hdf5.h>

namespace splash { namespace utils { namespace hdf5 {
    


template <typename T, bool BUILTIN = std::is_arithmetic<T>::value>
struct datatype;

// see https://support.hdfgroup.org/HDF5/doc/RM/PredefDTypes.html

// template specialized structs.  
template <typename TT> struct datatype<TT*, false> {  hid_t value { H5T_NATIVE_HADDR }; };

// duplicate uint64_t and int64_t
// template <> struct datatype<size_t, true> {  hid_t value { H5T_NATIVE_HSIZE }; };
// template <> struct datatype<ssize_t, true> {  hid_t value { H5T_NATIVE_HSSIZE }; };

template <> struct datatype<bool, true> {  hid_t value { H5T_NATIVE_HBOOL }; };

template <> struct datatype<char, true> {  hid_t value { H5T_NATIVE_CHAR }; };
template <> struct datatype<signed char, true> {  hid_t value { H5T_NATIVE_SCHAR }; };
template <> struct datatype<unsigned char, true> {  hid_t value { H5T_NATIVE_UCHAR }; };

template <> struct datatype<wchar_t, true> {  hid_t value { H5T_NATIVE_SHORT }; };
template <> struct datatype<short, true> {  hid_t value { H5T_NATIVE_SHORT }; };
template <> struct datatype<unsigned short, true> {  hid_t value { H5T_NATIVE_USHORT }; };

template <> struct datatype<int, true> {  hid_t value { H5T_NATIVE_INT }; };
template <> struct datatype<unsigned int, true> {  hid_t value { H5T_NATIVE_UINT}; };

template <> struct datatype<long, true> {  hid_t value { H5T_NATIVE_LONG }; };
template <> struct datatype<unsigned long, true> {  hid_t value { H5T_NATIVE_ULONG }; };
template <> struct datatype<long long, true> {  hid_t value { H5T_NATIVE_LLONG }; };
template <> struct datatype<unsigned long long, true> {  hid_t value { H5T_NATIVE_ULLONG }; };

template <> struct datatype<float, true> {  hid_t value { H5T_NATIVE_FLOAT }; };
template <> struct datatype<double, true> {  hid_t value { H5T_NATIVE_DOUBLE }; };
template <> struct datatype<long double, true> {  hid_t value { H5T_NATIVE_LDOUBLE }; };

template <> struct datatype<std::bitset<8>, false> { hid_t value { H5T_NATIVE_B8 }; };
template <> struct datatype<std::bitset<16>, false> { hid_t value { H5T_NATIVE_B16 }; };
template <> struct datatype<std::bitset<32>, false> { hid_t value { H5T_NATIVE_B32 }; };
template <> struct datatype<std::bitset<64>, false> { hid_t value { H5T_NATIVE_B64 }; };

}}}
