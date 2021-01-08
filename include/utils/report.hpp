/*
 *  error_handler.hpp
 *
 *  Created on: June 11, 2020
 *  Author: Tony Pan
 *  Affiliation: Institute for Data Engineering and Science
 *  			Georgia Institute of Technology, Atlanta, GA 30332
 */

#pragma once

#include <unordered_map>
#include <cstdio>
#include <sstream>
#include <atomic>
#include <cstring>

// fmt lib requires the macro def.
#define FMT_HEADER_ONLY
#include "fmt/format.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef USE_OPENMP
#include <omp.h>
#endif


// has to use macro. format string will become variable instead of literal. 
#ifdef USE_MPI
#define GET_RANK()  int ___bp__rank;  MPI_Comm_rank(MPI_COMM_WORLD, &___bp__rank)
#else
#define GET_RANK() int ___bp__rank = 0
#endif

#ifdef USE_OPENMP
#define GET_THREAD_ID() int ___bp__tid = omp_get_thread_num()
#else
#define GET_THREAD_ID() int ___bp__tid = 0
#endif

// goal: want to keep per thread /processs output as separated as possible in the stdout or stderr.
// also want to print the rank and thread for each
// challenge.  can't concat format strings  to do this 
//      build a new string -> not literal, so compiler warning
//      c++ doesn't seem to allow string literal to concate without operator anymore.




// auto expanding print buffer, for printing multiple strings together.
class buffered_printf {
    private:
        // 2D.  thread id, then fileid (including stdout, stderr)
        // flush should be called on map elements during destruction
        static std::unordered_map<int, std::unordered_map<FILE *, buffered_printf>> instances;
        
    protected:
        char * _data;  // local per thread.
        size_t _capacity;
        std::atomic<size_t> _size;
        FILE * stream;
        
        // this is  per thread., and should have no contention.
        inline void resize(size_t const & new_size) {
            if (new_size <= this->_capacity)  return;  // sufficient space
            // copy
            char* new_data = reinterpret_cast<char*>(malloc(new_size));
            memcpy(new_data , this->_data, this->_size);
            memset(new_data + this->_size, 0, new_size - this->_size);
            this->_capacity = new_size;

            // clear old
            free(this->_data);
            std::swap(this->_data, new_data);
        }

    public:
        static buffered_printf& get_instance(int const & tid, FILE * stream) {
// data race likely since single data structure.  use omp single to initialize.
#ifdef USE_OPENMP
#pragma omp critical  // have to use critical instead of single.  Singles does not appear to block  the other threads  so may cause hang at fprintf line.
            {
#endif

            if (instances.empty()) {
#ifdef USE_OPENMP
                int max = omp_get_max_threads();
                // fprintf(stdout, "thread = %d, curr %d, empty %lu\n", max, omp_get_thread_num(), instances.size());  fflush(stdout);
#else
                int max = 1;
#endif
                for (int p = 0; p < max; ++p) {
                    instances.emplace(p, std::unordered_map<FILE*, buffered_printf>());
                }
                // fprintf(stdout, "thread = %d, result  %lu\n", max, instances.size());  fflush(stdout);
            }

#ifdef USE_OPENMP
            }
#endif
            // following should be per thread.
            if (instances[tid].find(stream) == instances[tid].end()) {
                instances[tid].emplace(stream, buffered_printf(stream));
            }
            return instances[tid][stream];
        }

        buffered_printf() : buffered_printf(stdout) {}

        buffered_printf(FILE * str, size_t initial = 1024) : _capacity(initial), _size(0), stream(str) {
            this->_data = reinterpret_cast<char*>(malloc(this->_capacity));
            memset(this->_data, 0, this->_capacity);
        }
        
        // destruction should be hidden from general calling as well. 
        ~buffered_printf() {
            this->flush();
            fflush(stream);
            free(this->_data);
            this->_data = nullptr;
        }

        buffered_printf(buffered_printf const & other) : _capacity(other._capacity), _size(other._size.load()), stream(other.stream) {
            this->_data = reinterpret_cast<char*>(malloc(this->_capacity));
            memcpy(this->_data, other._data, other._size);
        };
        buffered_printf(buffered_printf && other) : _capacity(other._capacity), _size(other._size.load()), stream(other.stream) {
            this->_data = other._data;
            other._data = nullptr;
            other._capacity = 0;
            other._size = 0;
        };
        buffered_printf& operator=(buffered_printf const & other) {
            if (this->_capacity < other._capacity) {
                free(this->_data);
                this->_capacity = other._capacity;
                this->_data = reinterpret_cast<char*>(malloc(this->_capacity));
            }
            this->_size = other._size.load();
            memcpy(this->_data, other._data, this->_size); 

            this->stream = other.stream;

            return *this;
        }
        buffered_printf& operator=(buffered_printf && other) {
            if ((this->_capacity > 0) && (this->_data != nullptr) )
                free(this->_data);

            this->_data = other._data;
            other._data = nullptr;
            
            this->_capacity = other._capacity;  other._capacity = 0;
            this->_size = other._size.load();  other._size = 0;
            this->stream = other.stream;

            return *this;
        }

        inline size_t reserve(int const & count) {
            if ((this->_size + count + 1) > this->_capacity) {
                this->resize((this->_size + count + 1) * 2);  // double, just in case.
            }
            return this->_size.fetch_add(count);
        }

        inline char* data(size_t const & pos = 0) {
            return this->_data + pos;
        }

        // per thread flush.
        inline void flush() {
            if (this->_size == 0) return;

#ifdef USE_OPENMP
#pragma omp critical
            {            
#endif
            fprintf(stream, "%s", this->_data);
 #ifdef USE_OPENMP
            }
#endif
           memset(this->_data, 0, this->_size);
            this->_size = 0;
        
        }

        inline size_t size() const { return _size; }
        inline size_t capacity() const { return _capacity; }
};
// #ifdef USE_OPENMP
std::unordered_map<int, std::unordered_map<FILE *, buffered_printf>> buffered_printf::instances = std::unordered_map<int, std::unordered_map<FILE *, buffered_printf>>();



// first snprintf call returns size of printed string.
#define BUFFERED_PRINT(___bp__tid, ___bp__stream, format, ...) do {\
    ssize_t ___bp__pos = snprintf(NULL, 0, format, ##__VA_ARGS__);  \
    auto ___bp__buf = buffered_printf::get_instance(___bp__tid, ___bp__stream); \
    ___bp__pos = ___bp__buf.reserve(___bp__pos); \
    ___bp__pos = sprintf(___bp__buf.data(___bp__pos), format, ##__VA_ARGS__); \
} while (0) 

#define BUFFERED_PRINT_RT(___bp__rank, ___bp__tid, ___bp__stream, format, ...) do {\
    ssize_t ___bp__pos = snprintf(NULL, 0, "[R%dT%d] ", ___bp__rank, ___bp__tid); \
    ___bp__pos += snprintf(NULL, 0, format, ##__VA_ARGS__);  \
    auto ___bp__buf = buffered_printf::get_instance(___bp__tid, ___bp__stream); \
    ___bp__pos = ___bp__buf.reserve(___bp__pos); \
    ___bp__pos = sprintf(___bp__buf.data(___bp__pos), "[R%dT%d] ", ___bp__rank, ___bp__tid); \
    ___bp__pos += sprintf(___bp__buf.data(___bp__pos), format, ##__VA_ARGS__); \
} while (0) 

#define C_PRINT_ERR(format, ...)  do {\
    GET_RANK(); \
    GET_THREAD_ID(); \
    BUFFERED_PRINT_RT(___bp__rank, ___bp__tid, stderr, format, ##__VA_ARGS__); \
    buffered_printf::get_instance(___bp__tid, stderr).flush(); \
} while(false)

#define C_PRINT_RT(format, ...)  do {\
    GET_RANK(); \
    GET_THREAD_ID(); \
    BUFFERED_PRINT_RT(___bp__rank, ___bp__tid, stdout, format, ##__VA_ARGS__); \
    buffered_printf::get_instance(___bp__tid, stdout).flush(); \
} while(false)

#define C_ROOT_PRINT(format, ...) do {\
    GET_RANK(); \
    if (___bp__rank == 0) {\
        GET_THREAD_ID(); \
        BUFFERED_PRINT(___bp__tid, stdout, format, ##__VA_ARGS__); \
        buffered_printf::get_instance(___bp__tid, stdout).flush(); \
    } \
} while(false)

#define C_ROOT_PRINT_RT(format, ...) do {\
    GET_RANK(); \
    if (___bp__rank == 0) {\
        GET_THREAD_ID(); \
        BUFFERED_PRINT_RT(___bp__rank, ___bp__tid, stdout, format, ##__VA_ARGS__); \
        buffered_printf::get_instance(___bp__tid, stdout).flush(); \
    } \
} while(false)

#define C_MIN_MAX_DOUBLE_PRINT(prefix, val) do {\
    double v = val; \
    double mx = val; \
    double mn = val; \
    MPI_Reduce(&v, &mx, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); \
    MPI_Reduce(&v, &mn, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD); \
    GET_RANK(); \
    if (___bp__rank == 0) {\
        GET_THREAD_ID(); \
        BUFFERED_PRINT(___bp__tid, stdout, "%s [%lf - %lf]\n", prefix, mn, mx); \
        buffered_printf::get_instance(___bp__tid, stdout).flush(); \
    } \
} while(false)


#define C_PRINT(format, ...) fprintf(stdout, format, ##__VA_ARGS__)

#define C_FLUSH() do { fflush(stdout); fflush(stderr); } while(false)







/////// USE FMT library.





// auto expanding print buffer, for printing multiple strings together.
class fmt_printf {
    private:
        // 2D.  thread id, then fileid (including stdout, stderr)
        // flush should be called on map elements during destruction
        static std::unordered_map<int, std::unordered_map<FILE *, fmt_printf>> instances;
        
    protected:
        fmt::memory_buffer _data;  // local per thread.
        FILE * stream;
        
        // this is  per thread., and should have no contention.
        inline void resize(size_t const & new_size) {
            this->_data.resize(new_size);
        }

    public:
        static fmt_printf& get_instance(int const & tid, FILE * stream) {
// data race likely since single data structure.  use omp single to initialize.
#ifdef USE_OPENMP
#pragma omp critical  // have to use critical instead of single.  Singles does not appear to block  the other threads  so may cause hang at fprintf line.
            {
#endif

            if (instances.empty()) {
#ifdef USE_OPENMP
                int max = omp_get_max_threads();
                // fprintf(stdout, "thread = %d, curr %d, empty %lu\n", max, omp_get_thread_num(), instances.size());  fflush(stdout);
#else
                int max = 1;
#endif
                for (int p = 0; p < max; ++p) {
                    instances.emplace(p, std::unordered_map<FILE*, fmt_printf>());
                }
                // fprintf(stdout, "thread = %d, result  %lu\n", max, instances.size());  fflush(stdout);
            }

#ifdef USE_OPENMP
            }
#endif
            // following should be per thread.
            if (instances[tid].find(stream) == instances[tid].end()) {
                instances[tid].emplace(stream, fmt_printf(stream));
            }
            return instances[tid][stream];
        }

        fmt_printf() : fmt_printf(stdout) {}

        fmt_printf(FILE * str) : stream(str) {}
        
        // destruction should be hidden from general calling as well. 
        ~fmt_printf() {
            this->flush();
            fflush(stream);
        }

        fmt_printf(fmt_printf const & other) : stream(other.stream) {
            _data.resize(other._data.size());
            _data.reserve(other._data.capacity());
            memcpy(_data.data(), other._data.data(), other._data.size());
        };

        fmt_printf(fmt_printf && other) : _data(std::move(other._data)), stream(other.stream) {};

        fmt_printf& operator=(fmt_printf const & other) {
            _data.resize(other._data.size());
            _data.reserve(other._data.capacity());
            memcpy(_data.data(), other._data.data(), other._data.size());

            this->stream = other.stream;

            return *this;
        }
        fmt_printf& operator=(fmt_printf && other) {
            this->_data = std::move(other._data);
            this->stream = other.stream;

            return *this;
        }

        inline size_t reserve(int const & count) {
            this->_data.reserve(count);
            return this->_data.capacity();
        }

        inline char* data(size_t const & pos = 0) {
            return this->_data.data() + pos;
        }

        // per thread flush.
        inline void flush() {
            if (this->_data.size() == 0) return;

#ifdef USE_OPENMP
#pragma omp critical
            {            
#endif
            fmt::print(stream, fmt::to_string(this->_data));
 #ifdef USE_OPENMP
            }
#endif
            this->_data.clear();        
        }

        inline size_t size() const { return _data.size(); }
        inline size_t capacity() const { return _data.capacity(); }

        template <typename ... Args>        
        inline void print(const char * format, Args... args) {
            format_to(this->_data, format, args...);
            this->flush();
        }
        template <typename ... Args>        
        inline void print(int const & ___bp__rank, int const & ___bp__tid, const char * format, Args... args) {
            format_to(this->_data, "[R{}T{}] ", ___bp__rank, ___bp__tid);
            format_to(this->_data, format, args...);
            this->flush();
        }

};
std::unordered_map<int, std::unordered_map<FILE *, fmt_printf>> fmt_printf::instances = std::unordered_map<int, std::unordered_map<FILE *, fmt_printf>>();


#define FMT_PRINT_ERR(format, ...)  do {\
    GET_RANK(); \
    GET_THREAD_ID(); \
    fmt_printf::get_instance(___bp__tid, stderr).print(___bp__rank, ___bp__tid, format, ##__VA_ARGS__); \
} while(false)

#define FMT_PRINT_RT(format, ...)  do {\
    GET_RANK(); \
    GET_THREAD_ID(); \
    fmt_printf::get_instance(___bp__tid, stdout).print(___bp__rank, ___bp__tid, format, ##__VA_ARGS__); \
} while(false)

#define FMT_ROOT_PRINT(format, ...) do {\
    GET_RANK(); \
    if (___bp__rank == 0) {\
        GET_THREAD_ID(); \
        fmt_printf::get_instance(___bp__tid, stdout).print(format, ##__VA_ARGS__); \
    } \
} while(false)

#define FMT_ROOT_PRINT_RT(format, ...) do {\
    GET_RANK(); \
    if (___bp__rank == 0) {\
        GET_THREAD_ID(); \
        fmt_printf::get_instance(___bp__tid, stdout).print(___bp__rank, ___bp__tid, format, ##__VA_ARGS__); \
    } \
} while(false)

#define FMT_MIN_MAX_DOUBLE_PRINT(prefix, val) do {\
    double v = val; \
    double mx = val; \
    double mn = val; \
    MPI_Reduce(&v, &mx, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD); \
    MPI_Reduce(&v, &mn, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD); \
    GET_RANK(); \
    if (___bp__rank == 0) {\
        GET_THREAD_ID(); \
        fmt_printf::get_instance(___bp__tid, stdout).print("{} [{} - {}]\n", prefix, mn, mx); \
    } \
} while(false)


#define FMT_PRINT(format, ...) do {\
    GET_THREAD_ID(); \
    fmt_printf::get_instance(___bp__tid, stdout).print(format, ##__VA_ARGS__); \
} while(false)


#define FMT_FLUSH() do { \
    GET_THREAD_ID(); \
    fmt_printf::get_instance(___bp__tid, stdout).flush(); \
    fmt_printf::get_instance(___bp__tid, stderr).flush(); \
} while(false)
