#pragma once
#include "utils/report.hpp"

namespace splash { namespace ds {

// default constructor creates a null buffer.   allocate during run, by first calling resize.
template <typename T>
struct buffer {
        mutable T * data;
        mutable size_t count;

        buffer() : data(nullptr), count(0) {
// #ifdef USE_OPENMP
// 			FMT_PRINT_RT("construct BUFFER, thread {}\n", omp_get_thread_num());
// #endif  
		}
		~buffer() {
			if (data) {
				splash::utils::afree(data);
				data = nullptr;
                count = 0;
			}
		}
		buffer(buffer const & other) : count(other.count)  {
            data = reinterpret_cast<T*>(splash::utils::aalloc(count * sizeof(T)));
            if (other.data && other.count) {
                // memcpy(data, other.data, other.count * sizeof(T));
                std::copy(other.data, other.data + other.count, data);
            }
        }
		buffer& operator=(buffer const & other) {
            std::tie(data, count) = splash::utils::acresize(data, count, other.count);
            if (data && other.data)
                // memcpy(data, other.data, count * sizeof(T));
                std::copy(other.data, other.data + other.count, data);
        }
		buffer(buffer && other) : data(std::move(other.data)), count(other.count) {
            other.data = nullptr;
            other.count = 0;
        } 
		buffer& operator=(buffer && other) {
            if (data) splash::utils::afree(data);
            data = other.data; other.data = nullptr;
            count = other.count; other.count = 0;
        }

		void resize(size_t const & size) {
            std::tie(data, count) = splash::utils::acresize(data, count, size);
		}
};

}}
