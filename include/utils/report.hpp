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

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef USE_OPENMP
#include <omp.h>
#endif

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
                fprintf(stdout, "thread = %d, curr %d, empty %lu\n", max, omp_get_thread_num(), instances.size());  fflush(stdout);
#else
                int max = 1;
#endif
                for (int p = 0; p < max; ++p) {
                    instances.emplace(p, std::unordered_map<FILE*, buffered_printf>());
                }
                fprintf(stdout, "thread = %d, result  %lu\n", max, instances.size());  fflush(stdout);
            }
            // if (instances.find(tid) == instances.end()) {
            //     instances.emplace(tid, std::unordered_map<FILE*, buffered_printf>());
            // }
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
            
            fprintf(stream, "%s", this->_data);
            memset(this->_data, 0, this->_size);
            this->_size = 0;
        }

        inline size_t size() { return _size; }
        inline size_t capacity() { return _capacity; }
};
// #ifdef USE_OPENMP
// // initialize with sufficient size so it's never resized, which could cause race condition.
// std::unordered_map<int, std::unordered_map<FILE *, buffered_printf>> buffered_printf::instances = std::unordered_map<int, std::unordered_map<FILE *, buffered_printf>>(omp_get_max_threads() * 2);
// #else
// std::unordered_map<int, std::unordered_map<FILE *, buffered_printf>> buffered_printf::instances = std::unordered_map<int, std::unordered_map<FILE *, buffered_printf>>(2);
// #endif
std::unordered_map<int, std::unordered_map<FILE *, buffered_printf>> buffered_printf::instances = std::unordered_map<int, std::unordered_map<FILE *, buffered_printf>>();

// goal: want to keep per thread /processs output as separated as possible in the stdout or stderr.
// also want to print the rank and thread for each
// challenge.  can't concat format strings  to do this 
//      build a new string -> not literal, so compiler warning
//      c++ doesn't seem to allow string literal to concate without operator anymore.


// has to use macro. format string will become variable instead of literal. 
#ifdef USE_MPI
#define GET_RANK()  int rank;  MPI_Comm_rank(MPI_COMM_WORLD, &rank)
#else
#define GET_RANK() int rank = 0
#endif

#ifdef USE_OPENMP
#define GET_THREAD_ID() int tid = omp_get_thread_num()
#else
#define GET_THREAD_ID() int tid = 0
#endif

#define BUFFERED_PRINT(tid, stream, ...) do {\
    ssize_t count = 0; \
    count = snprintf(NULL, 0, __VA_ARGS__);  \
    size_t pos = buffered_printf::get_instance(tid, stream).reserve(count); \
    count = sprintf(buffered_printf::get_instance(tid, stream).data(pos), __VA_ARGS__); \
} while (0) 


#define PRINT_ERR(...)  do {\
    GET_RANK(); \
    GET_THREAD_ID(); \
    BUFFERED_PRINT(tid, stderr, "[R%dT%d] ", rank, tid); \
    BUFFERED_PRINT(tid, stderr, __VA_ARGS__); \
    buffered_printf::get_instance(tid, stderr).flush(); \
} while(false)

#define PRINT_RT(...)  do {\
    GET_RANK(); \
    GET_THREAD_ID(); \
    BUFFERED_PRINT(tid, stdout, "[R%dT%d] ", rank, tid); \
    BUFFERED_PRINT(tid, stdout, __VA_ARGS__); \
    buffered_printf::get_instance(tid, stdout).flush(); \
} while(false)

#define ROOT_PRINT(...) do {\
    GET_RANK(); \
    if (rank == 0) {\
        GET_THREAD_ID(); \
        BUFFERED_PRINT(tid, stdout,  __VA_ARGS__); \
        buffered_printf::get_instance(tid, stdout).flush(); \
    } \
} while(false)

#define ROOT_PRINT_RT(...) do {\
    GET_RANK(); \
    if (rank == 0) {\
        GET_THREAD_ID(); \
        BUFFERED_PRINT(tid, stdout, "[R%dT%d] ", rank, tid); \
        BUFFERED_PRINT(tid, stdout, __VA_ARGS__); \
        buffered_printf::get_instance(tid, stdout).flush(); \
    } \
} while(false)

#define PRINT(...) fprintf(stdout, __VA_ARGS__)

#define FLUSH() do { fflush(stdout); fflush(stderr); } while(false)