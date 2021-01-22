#pragma once

namespace splash { namespace utils { namespace hdf5 {
    
#include <type_traits>
#include <hdf5.h>

template <typename T, bool BUILTIN = std::is_arithmetic<T>::value>
struct datatype;

// template specialized structs.  
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


}}}
