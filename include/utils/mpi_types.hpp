#pragma once

namespace splash { namespace utils { namespace mpi {
    
#include <type_traits>
#include <mpi.h>

template <typename T, bool BUILTIN = std::is_arithmetic<T>::value>
struct datatype;

// template specialized structs.  
template <> struct datatype<char, true> {  MPI_Datatype value { MPI_CHAR }; };
template <> struct datatype<signed char, true> {  MPI_Datatype value { MPI_SIGNED_CHAR }; };
template <> struct datatype<unsigned char, true> {  MPI_Datatype value { MPI_BYTE }; };
template <> struct datatype<wchar_t, true> {  MPI_Datatype value { MPI_WCHAR }; };
template <> struct datatype<short, true> {  MPI_Datatype value { MPI_SHORT }; };
template <> struct datatype<unsigned short, true> {  MPI_Datatype value { MPI_UNSIGNED_SHORT }; };
template <> struct datatype<int, true> {  MPI_Datatype value { MPI_INT }; };
template <> struct datatype<unsigned int, true> {  MPI_Datatype value { MPI_UNSIGNED }; };
template <> struct datatype<long, true> {  MPI_Datatype value { MPI_LONG }; };
template <> struct datatype<unsigned long, true> {  MPI_Datatype value { MPI_UNSIGNED_LONG }; };
template <> struct datatype<long long, true> {  MPI_Datatype value { MPI_LONG_LONG }; };
template <> struct datatype<unsigned long long, true> {  MPI_Datatype value { MPI_UNSIGNED_LONG_LONG }; };
template <> struct datatype<float, true> {  MPI_Datatype value { MPI_FLOAT }; };
template <> struct datatype<double, true> {  MPI_Datatype value { MPI_DOUBLE }; };
template <> struct datatype<long double, true> {  MPI_Datatype value { MPI_LONG_DOUBLE }; };
// template <> struct datatype<int8_t, true> {  MPI_Datatype value { MPI_INT8_T }; };
// template <> struct datatype<int16_t, true> {  MPI_Datatype value { MPI_INT16_T }; };
// template <> struct datatype<int32_t, true> {  MPI_Datatype value { MPI_INT32_T }; };
// template <> struct datatype<int64_t, true> {  MPI_Datatype value { MPI_INT64_T }; };
// template <> struct datatype<uint8_t, true> {  MPI_Datatype value { MPI_UINT8_T }; };
// template <> struct datatype<uint16_t, true> {  MPI_Datatype value { MPI_UINT16_T }; };
// template <> struct datatype<uint32_t, true> {  MPI_Datatype value { MPI_UINT32_T }; };
// template <> struct datatype<uint64_t, true> {  MPI_Datatype value { MPI_UINT64_T }; };


// array types.  use std::vector as the specialization.
template <typename T> 
struct datatype<std::vector<T>, false> {
    datatype(size_t const & count) {
        splash::utils::mpi::datatype<T> dt1;
        MPI_Type_contiguous(count, dt1.value, &value);
        MPI_Type_commit(&value);
    }
    ~datatype() {
        MPI_Type_free(&value);
    }
    MPI_Datatype value;
};


template <typename T1, typename T2> 
struct datatype<std::pair<T1, T2>, false> {
    datatype() {
        splash::utils::mpi::datatype<T1> dt1;
        splash::utils::mpi::datatype<T2> dt2;

        MPI_Datatype types[2] = {
            dt1.value,
            dt2.value
        };
        int blocklen[2] = {1, 1};
        std::pair<T1, T2> test {};
        MPI_Aint disp[2]; 
        disp[0] = reinterpret_cast<unsigned char *>(&test.first) - reinterpret_cast<unsigned char *>(&test);
        disp[1] = reinterpret_cast<unsigned char *>(&test.second) - reinterpret_cast<unsigned char *>(&test);
        MPI_Type_create_struct(2, blocklen, disp, types, &value);
        MPI_Type_commit(&value);
    }
    ~datatype() {
        MPI_Type_free(&value);
    }
    MPI_Datatype value;
};


//  using structs and method
// NOTE: cannot use static variables - they are initialized before program starts, so before MPI runtime is initialized.
// struct so that we can construct commit and release complex types, and allow reuse.
// method so that a saved data type can be reused.
// no singleton pattern - requires either struct or method static variable. 
// initialization may happen before MPI_init and destruction after MPI_Finalize
// should look like:
// template <typename T>
// struct datatype<T, false> {
//      datatype() {};
//      ~datatype() {};
//    MPI_Datatype value; 
// }

}}}
