/*
 * Copyright 2020 Georgia Tech Research Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 * Author(s): Tony C. Pan
 */

#pragma once
#include "splash/utils/report.hpp"

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
