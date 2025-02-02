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


#include "splash/ds/char_array.hpp" // char_array_template
#include <sys/mman.h>  // mmap
#include <fcntl.h>     // open
#include <sys/stat.h>  // stat
#include <iostream>    // ifstream
#include <fstream>     // ifstream
#include "splash/utils/memory.hpp"   // aalloc, afree
#include <cassert>

#include "splash/utils/benchmark.hpp"
#include "splash/utils/report.hpp"
#include "splash/utils/mpi_types.hpp"

#ifdef USE_MPI
#include <mpi.h>
#endif  // with mpi

namespace splash { namespace io { 


class FileReader2 {
	protected:

		splash::ds::char_array_template data;
		bool mapped;

		inline splash::ds::char_array_template load(const char * filename) {
			auto stime = getSysTime();

			splash::ds::char_array_template output = {nullptr, 0};
			
			std::ifstream file(filename, std::ios::binary | std::ios::in | std::ios::ate);
			output.size = file.tellg();
			file.seekg(0, std::ios::beg);

			output.ptr = reinterpret_cast<char *>(splash::utils::aalloc(output.size));
			file.read(output.ptr, output.size);

			auto etime = getSysTime();
			FMT_ROOT_PRINT("read file in {} sec\n", get_duration_s(stime, etime));
			return output;
		}


		inline splash::ds::char_array_template map(const char * filename) {
			auto stime = getSysTime();
			splash::ds::char_array_template output = {nullptr, 0};

			int fd = ::open(filename, O_RDONLY);
			struct stat st;
			int res = 0;
			if (fd != -1) {
				res = fstat(fd, &st);
			}

			if (res != -1) {
				output.ptr = reinterpret_cast<char*>(mmap(NULL, st.st_size, PROT_READ,
							MAP_PRIVATE, fd, 0));
				if (output.ptr != MAP_FAILED) {
					output.size = st.st_size;
				} else {
					output.ptr = nullptr;
				}
			}
			auto etime = getSysTime();
			FMT_ROOT_PRINT("map file in {} sec\n", get_duration_s(stime, etime));
			return output;
		}

		inline void unmap(splash::ds::char_array_template & buffer) {
#ifndef NDEBUG
			int rc = 
#endif
			munmap(reinterpret_cast<void *>(buffer.ptr), buffer.size);
			buffer.ptr = nullptr;
			buffer.size = 0;
			assert((rc == 0) && "failed to memunmap file");
		}

// NOT USED.
// #ifdef USE_MPI
// 		inline splash::ds::char_array_template open(const char * filename, MPI_Comm comm) {
// 			splash::ds::char_array_template output = {nullptr, 0};

// 			// MPI stuff.
// 			ssize_t filesize;
// 			int rank, procs;
// 			MPI_Comm_rank(comm, &rank);
// 			MPI_Comm_size(comm, &procs);
// 			int result;

// 			// -------- open file
// 			MPI_File fh;
// 			result = MPI_File_open(comm, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
// 			if(result != MPI_SUCCESS) 
// 				FMT_PRINT_RT("ERROR: MPI_File_open failed for {}\n", filename);
// 			else {
// 				// --------- get file size
// 				if (rank == 0) {
// 					MPI_Offset temp;
// 					result = MPI_File_get_size(fh, &temp);
// 					filesize = temp;
// 				}
// 				MPI_Bcast(&filesize, 1, MPI_LONG, 0, comm);

// 				// next partition
// 				output.size = filesize / procs;
// 				size_t bytes_rem = filesize % procs;
// 				MPI_Offset offset = output.size * rank;
// 				if (static_cast<size_t>(rank) < bytes_rem) {
// 					offset += rank;
// 					output.size += 1;
// 				} else {
// 					offset += bytes_rem;
// 				}
				
// 				// allocate buffer. 
// 				output.ptr = reinterpret_cast<char *>(splash::utils::aalloc((output.size + 2) * sizeof(char)));
// 				// read
// 				MPI_Status status;
// 				result = MPI_File_read_at_all(fh, offset, output.ptr, output.size, MPI_BYTE, &status);
// 				if(result != MPI_SUCCESS) 
// 					FMT_PRINT_RT("ERROR: MPI_File_read_at failed for rank {} at offset {} for length {}\n", rank, offset, output.size);
// 				int bytes_read;
// 				result = MPI_Get_elements(&status, MPI_BYTE, &bytes_read);
// 				if(result != MPI_SUCCESS)
// 					FMT_PRINT_RT("ERROR: MPI_Get_elements failed to get bytes_read\n");
// 				output.size = bytes_read;
// 				MPI_File_close(&fh);

// 				// move 1 byte to the left.  for data with 
// 				char send = output.ptr[0];
// 				if (rank == 0) {
// 					// add an '\n' in case the files has missing \n
// 					send = '\n';
// 				}
// 				int left = (rank + procs - 1) % procs;
// 				int right = (rank + 1) % procs;

// 				// move data
// 				MPI_Sendrecv(&send, 1, MPI_BYTE, left, 1,
// 							output.ptr + output.size, 1, MPI_BYTE, right, 1, comm, &status);
// 				// terminate with 0
// 				output.ptr[output.size + 1] = 0;
// 			}
// 			return output;
// 		}
// #endif

	public:

#ifdef USE_MPI
		// memmap the whole file on rank 1 only  NOT using MPI_File
		FileReader2(const char * filename, MPI_Comm comm = MPI_COMM_WORLD, bool _map = true) {
			int rank;
			MPI_Comm_rank(comm, &rank);

			if (rank == 0) {  // rank 0 read the file and broadcast.
				if (_map)
					data = this->map(filename);
				else
					data = this->load(filename);			
				mapped = _map;
				MPI_Bcast(&(data.size), 1, MPI_UNSIGNED_LONG, 0, comm);
			} else {
				MPI_Bcast(&(data.size), 1, MPI_UNSIGNED_LONG, 0, comm);
				data.ptr = reinterpret_cast<char *>(splash::utils::aalloc((data.size) * sizeof(char)));
				mapped = false;
			}

			// do in batches.
			char* ptr = data.ptr;
			size_t c = data.size;
			int block;
			for (c = 0; c < data.size; c += std::numeric_limits<int>::max(), ptr += std::numeric_limits<int>::max()) {
				block = std::min(static_cast<size_t>(std::numeric_limits<int>::max()), data.size - c);
				MPI_Bcast(ptr, block, MPI_BYTE, 0, comm);
			}

		}
#else
		// memmap the whole file
		FileReader2(const char * filename, bool _map = true) : mapped(_map) {
			if (_map)
				data = this->map(filename);
			else
				data = this->load(filename);			
		}
#endif
		virtual ~FileReader2() {
			if (mapped && (data.ptr != nullptr) && (data.size != 0)) {
				this->unmap(data);
			} else if (!mapped && (data.ptr != nullptr)) {
				splash::utils::afree(data.ptr);
                data.ptr = nullptr;
			}
		}

};



}}
