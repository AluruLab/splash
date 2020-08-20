/*
 * CSVMatrixReader.hpp
 *
 *  Created on: Mar 25, 2016
 *  Author: Liu, Yongchao
 *  Affiliation: School of Computational Science & Engineering
 *  						Georgia Institute of Technology, Atlanta, GA 30332
 *  URL: www.liuyc.org
 */

#pragma once

#include <string>
#include <algorithm>
#include <vector>
#include "ds/aligned_matrix.hpp"
#include <sys/mman.h>  // mmap
#include <fcntl.h>  // open
#include <sys/stat.h> //stat
#include <iostream>
#include <fstream>

#include "utils/benchmark.hpp"
#include "utils/report.hpp"

#ifdef USE_MPI
#include <mpi.h>
#endif  // with mpi

namespace splash { namespace io { 

struct char_array {
	char * start;
	size_t size;
};

class FileReader {
	protected:

		// strtok modifies the buffer in memory.  avoid.

		static const char EOL[];
		static constexpr char CR = '\r';
		static constexpr char COMMA = ',';
		static constexpr size_t EMPTY = 0;

		char_array data;
		bool mapped;

		inline char_array load(const char * filename) {
			auto stime = getSysTime();

			char_array output = {nullptr, 0};
			
			std::ifstream file(filename, std::ios::binary | std::ios::in | std::ios::ate);
			output.size = file.tellg();
			file.seekg(0, std::ios::beg);

			output.start = reinterpret_cast<char *>(splash::utils::aalloc(output.size));
			file.read(output.start, output.size);

			auto etime = getSysTime();
			ROOT_PRINT("read file in %f sec\n", get_duration_s(stime, etime));
			return output;
		}


		inline char_array map(const char * filename) {
			auto stime = getSysTime();
			char_array output = {nullptr, 0};

			int fd = ::open(filename, O_RDONLY);
			struct stat st;
			int res = 0;
			if (fd != -1) {
				res = fstat(fd, &st);
			}

			if (res != -1) {
				output.start = reinterpret_cast<char*>(mmap(NULL, st.st_size, PROT_READ,
							MAP_PRIVATE, fd, 0));
				if (output.start != MAP_FAILED) {
					output.size = st.st_size;
				} else {
					output.start = nullptr;
				}
			}
			auto etime = getSysTime();
			ROOT_PRINT("map file in %f sec\n", get_duration_s(stime, etime));
			return output;
		}

		inline void unmap(char_array & buffer) {
#ifndef NDEBUG
			int rc = 
#endif
			munmap(reinterpret_cast<void *>(buffer.start), buffer.size);
			buffer.start = nullptr;
			buffer.size = 0;
			assert((rc == 0) && "failed to memunmap file");
		}

#ifdef USE_MPI
		inline char_array open(const char * filename, MPI_Comm comm) {
			char_array output = {nullptr, 0};

			// MPI stuff.
			ssize_t filesize;
			int rank, procs;
			MPI_Comm_rank(comm, &rank);
			MPI_Comm_size(comm, &procs);
			int result;

			// -------- open file
			MPI_File fh;
			result = MPI_File_open(comm, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
			if(result != MPI_SUCCESS) 
				fprintf(stderr, "ERROR: MPI_File_open failed for %s\n", filename);
			else {
				// --------- get file size
				if (rank == 0) {
					MPI_Offset temp;
					result = MPI_File_get_size(fh, &temp);
					filesize = temp;
				}
				MPI_Bcast(&filesize, 1, MPI_LONG, 0, comm);

				// next partition
				output.size = filesize / procs;
				size_t bytes_rem = filesize % procs;
				MPI_Offset offset = output.size * rank;
				if (static_cast<size_t>(rank) < bytes_rem) {
					offset += rank;
					output.size += 1;
				} else {
					offset += bytes_rem;
				}
				
				// allocate buffer. 
				output.start = reinterpret_cast<char *>(splash::utils::aalloc((output.size + 2) * sizeof(char)));
				// read
				MPI_Status status;
				result = MPI_File_read_at_all(fh, offset, output.start, output.size, MPI_BYTE, &status);
				if(result != MPI_SUCCESS) 
					fprintf(stderr, "ERROR: MPI_File_read_at failed for rank %d at offset %lld for length %lu\n", rank, offset, output.size);
				int bytes_read;
				result = MPI_Get_elements(&status, MPI_BYTE, &bytes_read);
				if(result != MPI_SUCCESS)
					fprintf(stderr, "MPI_Get_elements failed to get bytes_read\n");
				output.size = bytes_read;
				MPI_File_close(&fh);

				// move 1 byte to the left.  for data with 
				char send = output.start[0];
				if (rank == 0) {
					// add an '\n' in case the files has missing \n
					send = '\n';
				}
				int left = (rank + procs - 1) % procs;
				int right = (rank + 1) % procs;

				// move data
				MPI_Sendrecv(&send, 1, MPI_BYTE, left, 1,
							output.start + output.size, 1, MPI_BYTE, right, 1, comm, &status);
				// terminate with 0
				output.start[output.size + 1] = 0;
			}
			return output;
		}
#endif

	public:

#ifdef USE_MPI
		// memmap the whole file on rank 1 only  NOT using MPI_File
		FileReader(const char * filename, MPI_Comm comm = MPI_COMM_WORLD, bool _map = true) {
			int rank;
			MPI_Comm_rank(comm, &rank);

			if (rank == 0) {  // rank 0 read the file and broadcast.
				if (_map)
					data = this->map(filename);
				else
					data = this->load(filename);			
				mapped = _map;
				MPI_Bcast(&(data.size), 1, MPI_UNSIGNED_LONG, 0, comm);
				MPI_Bcast(data.start, data.size, MPI_BYTE, 0, comm);
			} else {
				MPI_Bcast(&(data.size), 1, MPI_UNSIGNED_LONG, 0, comm);
				data.start = reinterpret_cast<char *>(splash::utils::aalloc((data.size) * sizeof(char)));
				mapped = false;
				MPI_Bcast(data.start, data.size, MPI_BYTE, 0, comm);
			}

		}
#else
		// memmap the whole file
		FileReader(const char * filename, bool _map = true) : mapped(_map) {
			if (_map)
				data = this->map(filename);
			else
				data = this->load(filename);			
		}
#endif
		virtual ~FileReader() {
			if (mapped && (data.start != nullptr) && (data.size != 0)) {
				this->unmap(data);
			} else if (!mapped && (data.start != nullptr)) {
				splash::utils::afree(data.start);
			}
		}



		// search buffer to get to the first non-delim character, and return size.
		template <size_t N>
		char_array trim_left(char_array & buffer, const char (&delim)[N]) {
			// auto stime = getSysTime();	

			char_array output = {buffer.start, EMPTY};
			if ((buffer.size == EMPTY) || (buffer.start == nullptr)) return output;  // buffer is empty, short circuit.
			auto delim_end = delim + N;

			// find first entry that is NOT in delim
			buffer.start = std::find_if(buffer.start, buffer.start + buffer.size,
				[&delim, &delim_end](char const & c) {
					// by failing to find a match
					for (size_t i = 0; i < N; ++i) {
						if (delim[i] == c) return false;
					}
					return true;
					// return (std::find(delim, delim_end, c) == delim_end);
				});
			
			output.size = std::distance(output.start, buffer.start);
			buffer.size -= output.size;

			// auto etime = getSysTime();
			// ROOT_PRINT("trim_left x in %f sec\n", get_duration_s(stime, etime));
			return output;
		}
		char_array trim_left(char_array & buffer, const char delim) {
			// auto stime = getSysTime();	
			char_array output = {buffer.start, EMPTY};
			if ((buffer.size == EMPTY) || (buffer.start == nullptr)) return output;  // buffer is empty, short circuit.

			// find first entry that is NOT in delim
			size_t i;
			for (i = 0; i < buffer.size; ++i) {
				if (buffer.start[i] != delim) break;
			}
			buffer.start += i;
			output.size = i;
			buffer.size -= i;

			// buffer.start = std::find_if(buffer.start, buffer.start + buffer.size,
			// 	[&delim](char const & c) {
			// 		// by failing to find a match
			// 		return delim != c;
			// 	});
			// output.size = std::distance(output.start, buffer.start);
			// buffer.size -= output.size;

			// auto etime = getSysTime();	
			// ROOT_PRINT("trim_left in %f sec\n", get_duration_s(stime, etime));
			return output;
		}

	protected:
		// return token (ptr and length). does not treat consecutive delimiters as one.
		// assumes buffer points to start of token.
		// if there is no more token, then return nullptr
		template <size_t N>
		char_array extract_token(char_array & buffer, const char (&delim)[N]) {
			// auto stime = getSysTime();	
			if ((buffer.size == EMPTY) || (buffer.start == nullptr)) return char_array{nullptr, EMPTY};  // buffer is empty, short circuit.
			char_array output = {buffer.start, EMPTY};
			
			auto delim_end = delim + N;

			// search for first match to delim
			buffer.start = std::find_if(buffer.start, buffer.start + buffer.size,
			[&delim, &delim_end](char const & c){
				// by finding a match
				for (size_t i = 0; i < N; ++i) {
					if (delim[i] == c) return true;
				}
				return false;
				// return (std::find(delim, delim_end, c) != delim_end);
			});

			// update the output and buffer sizes.
			output.size = std::distance(output.start, buffer.start);
			buffer.size -= output.size;
			// printf("extract_token: %lu\n", output.size);
			// auto etime = getSysTime();	
			// ROOT_PRINT("extract token x in %f sec\n", get_duration_s(stime, etime));
			return output;
		}
		char_array extract_token(char_array & buffer, const char delim) {
			// auto stime = getSysTime();	
			if ((buffer.size == EMPTY) || (buffer.start == nullptr)) return char_array{nullptr, EMPTY};  // buffer is empty, short circuit.
			char_array output = {buffer.start, EMPTY};
			
			// search for first match to delim
			size_t i;
			for (i = 0; i < buffer.size; ++i) {
				if (buffer.start[i] == delim) break;
			}
			buffer.start += i;
			output.size = i;
			buffer.size -= i;

			// buffer.start = std::find_if(buffer.start, buffer.start + buffer.size,
			// [&delim](char const & c){
			// 	return (delim == c);
			// 	// return (std::find(delim, delim_end, c) != delim_end);
			// });
			// // update the output and buffer sizes.
			// output.size = std::distance(output.start, buffer.start);
			// buffer.size -= output.size;
			// printf("extract_token: %lu\n", output.size);
			// auto etime = getSysTime();	
			// ROOT_PRINT("extract token in %f sec\n", get_duration_s(stime, etime));
			return output;
		}
		// remove 1 delim character from buffer (or \r\n)
		template <size_t N>
		char_array trim_left_1(char_array & buffer, const char (&delim)[N]) {
			// auto stime = getSysTime();	

			char_array output = {buffer.start, EMPTY};
			if ((buffer.size == EMPTY) || (buffer.start == nullptr)) return output;  // buffer is empty, short circuit.
			
			// check if CR and in delim.  if yes, skip.
			if ((buffer.start[0] == CR) &&
				(std::find(delim, delim + N, CR) != delim + N)) {
				--buffer.size;	
				++buffer.start;
			}

			if ((buffer.size > EMPTY) && 
				(std::find(delim, delim + N, buffer.start[0]) != delim + N)) {
				--buffer.size;	
				++buffer.start;
			}
			output.size = std::distance(output.start, buffer.start);
			// printf("trim_left_1: %lu\n", output.size);
			// auto etime = getSysTime();	
			// ROOT_PRINT("trim_left_1 x in %f sec\n", get_duration_s(stime, etime));
			return output;
		}
		char_array trim_left_1(char_array & buffer, const char delim) {
			// auto stime = getSysTime();	
			char_array output = {buffer.start, EMPTY};
			if ((buffer.size == EMPTY) || (buffer.start == nullptr)) return output;  // buffer is empty, short circuit.
			
			if ((buffer.size > EMPTY) && (buffer.start[0] == delim)) {
				--buffer.size;	
				++buffer.start;
			}
			output.size = std::distance(output.start, buffer.start);
			// printf("trim_left_1: %lu\n", output.size);
			// auto etime = getSysTime();	
			// ROOT_PRINT("trim_left_1 in %f sec\n", get_duration_s(stime, etime));
			return output;
		}

	public:
		// get a token. does not treat consecutive delimiters as one so output may be empty.
		template <size_t N>
		char_array get_token_or_empty(char_array & buffer, const char (&delim)[N]) {
			char_array output = extract_token(buffer, delim);
			trim_left_1(buffer, delim);
			return output;
		}
		char_array get_token_or_empty(char_array & buffer, const char delim) {
			char_array output = extract_token(buffer, delim);
			trim_left_1(buffer, delim);
			return output;
		}
		// return token (ptr and length). treats consecutive delimiters as one.
		template <size_t N>
		char_array get_token(char_array & buffer, const char (&delim)[N]) {
			char_array output = extract_token(buffer, delim);
			trim_left(buffer, delim);
			return output;
		}		// return token (ptr and length). treats consecutive delimiters as one.
		char_array get_token(char_array & buffer, const char delim) {
			char_array output = extract_token(buffer, delim);
			trim_left(buffer, delim);
			return output;
		}

		template <size_t N>
		size_t count_token_or_empty(char_array & buffer, const char (&delim)[N]) {
			// auto stime = getSysTime();
			if ((buffer.size == EMPTY) || (buffer.start == nullptr)) return 0;

			auto delim_end = delim + N;
			// now count delimiters
			size_t count = std::count_if(buffer.start, buffer.start + buffer.size,
				[&delim, &delim_end](const char & c){
					for (size_t i = 0; i < N; ++i) {
						if (delim[i] == c) return true;
					}
					return false;
					// return (std::find(delim, delim_end, c) != delim_end);
				}) + 1;

			// auto etime = getSysTime();
			// ROOT_PRINT("count tokens or empty in %f sec\n", get_duration_s(stime, etime));
			return count;
		}
		size_t count_token_or_empty(char_array & buffer, const char delim) {
			// auto stime = getSysTime();
			if ((buffer.size == EMPTY) || (buffer.start == nullptr)) return 0;

			// now count delimiters
			size_t count = 0;
			for (size_t i = 0; i < buffer.size; ++i) {
				if (buffer.start[i] == delim) ++count;
			}
			++count;
			// size_t count = std::count_if(buffer.start, buffer.start + buffer.size,
			// 	[&delim](const char & c){
			// 		return (delim == c);
			// 		// return (std::find(delim, delim_end, c) != delim_end);
			// 	}) + 1;

			// auto etime = getSysTime();
			// ROOT_PRINT("count tokens or empty in %f sec\n", get_duration_s(stime, etime));
			return count;
		}
		template <size_t N>
		size_t count_token(char_array & buffer, const char (&delim)[N]) {
			// auto stime = getSysTime();
			if ((buffer.size == EMPTY) || (buffer.start == nullptr)) return 0;

			size_t count = 0;

			// set up the adjacent find predicate
			auto delim_end = delim + N;
			auto adj_find_pred = [&delim, &delim_end](const char & x, const char & y){
				// bool x_no_delim = std::find(delim, delim_end, x) == delim_end;
				// bool y_delim = std::find(delim, delim_end, y) != delim_end;
				bool x_no_delim = true;
				bool y_delim = false;
				for (size_t i = 0; i < N; ++i) {
					x_no_delim &= (delim[i] != x);
					y_delim |= (delim[i] == y);
				}
				return x_no_delim && y_delim; 
			};

			// now count non-delimiter to delimiter transitions
			auto start = buffer.start;
			auto end = buffer.start + buffer.size;
			while ((start = std::adjacent_find(start, end, adj_find_pred)) != end) {
				// found a transition
				++start; // go to next char
				++count; // increment count
			}

			// check the last char to see if we we have a non-delim to null transition.
			bool z_no_delim = true;
			for (size_t i = 0; i < N; ++i) {
				z_no_delim &= (delim[i] != buffer.start[buffer.size - 1]);
			}
			count += z_no_delim;
			
			// if (std::find(delim, delim_end, buffer.start[buffer.size - 1]) == delim_end) {
			// 	++count;
			// }
			// auto etime = getSysTime();
			// ROOT_PRINT("count tokens in %f sec\n", get_duration_s(stime, etime));
			return count;
		}

		size_t count_token(char_array & buffer, const char delim) {
			// auto stime = getSysTime();
			if ((buffer.size == EMPTY) || (buffer.start == nullptr)) return 0;

			size_t count = 0;

			// set up the adjacent find predicate
			// auto adj_find_pred = [&delim](const char & x, const char & y){
			// 	return (x != delim) && (y == delim); 
			// };

			// now count non-delimiter to delimiter transitions
			// auto start = buffer.start;
			// auto end = buffer.start + buffer.size;
			// while ((start = std::adjacent_find(start, end, adj_find_pred)) != end) {
			// 	// found a transition
			// 	++start; // go to next char
			// 	++count; // increment count
			// }
			// check the last char to see if we we have a non-delim to null transition.
			// bool z_no_delim = true;
			// for (size_t i = 0; i < N; ++i) {
			// 	z_no_delim &= (delim[i] != buffer.start[buffer.size - 1]);
			// }
			// count += z_no_delim;


			for (size_t i = 1; i < buffer.size; ++i) {
				if ((buffer.start[i-1] != delim) && (buffer.start[i] == delim)) ++count;
			}
			count += (buffer.start[buffer.size - 1] != delim);
			
			// if (std::find(delim, delim_end, buffer.start[buffer.size - 1]) == delim_end) {
			// 	++count;
			// }
			// auto etime = getSysTime();
			// ROOT_PRINT("count tokens in %f sec\n", get_duration_s(stime, etime));
			return count;
		}


};
const char FileReader::EOL[] = "\r\n";

template<typename FloatType>
class CSVMatrixReader : public FileReader {


public:

	CSVMatrixReader(const char* filename) : FileReader(filename) {}
	virtual ~CSVMatrixReader() {}

#ifdef USE_MPI
	bool getMatrixSize(int& numVectors, int& vectorSize,
		MPI_Comm comm = MPI_COMM_WORLD) {
		return getMatrixSize_impl(numVectors, vectorSize, comm);
	}
	bool loadMatrixData(vector<string>& genes,
			vector<string>& samples, FloatType* vectors, const int numVectors, const int vectorSize,
			const int stride_bytes, MPI_Comm comm = MPI_COMM_WORLD) {
		return loadMatrixData_impl( genes, samples, vectors, numVectors, vectorSize, stride_bytes,
			comm);
	}
	bool loadMatrixData(vector<string>& genes,
			vector<string>& samples, splash::ds::aligned_matrix<FloatType> & output, 
			MPI_Comm comm = MPI_COMM_WORLD) {
		return loadMatrixData_impl( genes, samples, 
			output.data(), output.rows(), output.columns(), output.column_bytes(), 
			comm);
	}
#else
	/*get gene expression matrix size*/
	bool getMatrixSize(int& numVectors, int& vectorSize) {
		return getMatrixSize_impl(numVectors, vectorSize);
	}
	bool loadMatrixData(vector<string>& genes,
			vector<string>& samples, FloatType* vectors, const int numVectors, const int vectorSize,
			const int stride_bytes) {
		return loadMatrixData_impl( genes, samples, vectors, numVectors, vectorSize, stride_bytes);
	}
	bool loadMatrixData(vector<string>& genes,
			vector<string>& samples, splash::ds::aligned_matrix<FloatType> & output) {
		return loadMatrixData_impl( genes, samples, 
			output.data(), output.rows(), output.columns(), output.column_bytes());
	}

#endif



protected:
	/*get gene expression matrix size*/
	bool getMatrixSize_impl(int& numVectors, int& vectorSize);

	/*get the matrix data*/
	bool loadMatrixData_impl(vector<string>& genes,
			vector<string>& samples, FloatType* vectors, const int & numVectors, const int & vectorSize,
			const int & stride_bytes);

	/*get the matrix data*/
	// bool loadMatrixData_impl(vector<string>& genes,
	// 		vector<string>& samples, splash::ds::aligned_matrix<FloatType> & input);


#ifdef USE_MPI
	/*get gene expression matrix size*/
	bool getMatrixSize_impl(int& numVectors, int& vectorSize, 
		MPI_Comm comm) {
			return getMatrixSize_impl(numVectors, vectorSize);
		}

	bool loadMatrixData_impl(vector<string>& genes,
			vector<string>& samples, FloatType* vectors, const int & numVectors, const int & vectorSize,
			const int & stride_bytes, MPI_Comm comm) {
			return loadMatrixData_impl(genes, samples, vectors, numVectors, vectorSize, stride_bytes);
		}

	// bool loadMatrixData_impl(vector<string>& genes,
	// 		vector<string>& samples, splash::ds::aligned_matrix<FloatType> & input, MPI_Comm comm);
#endif
};

template<typename FloatType>
bool CSVMatrixReader<FloatType>::getMatrixSize_impl(
		int& numVectors, int& vectorSize) {
	auto stime = getSysTime();

	char_array buffer = this->data;
	if (buffer.start == nullptr) {
		fprintf(stderr, "File not read\n");
		return false;
	}

	numVectors = vectorSize = 0;
	
	// get rid of empty lines.
	this->trim_left(buffer, FileReader::EOL);

	/*read the header to get the number of samples*/
	char_array line = this->get_token(buffer, FileReader::EOL);
	if (line.size <= 0) {
		fprintf(stderr, "The file is incomplete\n");
		this->unmap(buffer);
		return false;
	}

	/*analyze the header on the first row*/
	vectorSize = this->count_token_or_empty(line, FileReader::COMMA) - 1;
	fprintf(stderr, "Number of samples: %d\n", vectorSize);

	/*get gene expression profiles.  skip empty lines*/ 
	numVectors = FileReader::count_token(buffer, FileReader::EOL);
	fprintf(stderr, "Number of gene expression profiles: %d\n", numVectors);

	auto etime = getSysTime();
	ROOT_PRINT("get matrix size in %f sec\n", get_duration_s(stime, etime));
	return true;
}

template<typename FloatType>
bool CSVMatrixReader<FloatType>::loadMatrixData_impl(
		vector<string>& genes, vector<string>& samples, FloatType* vectors,
		const int & numVectors, const int & vectorSize, 
		const int & stride_bytes) {
	auto stime = getSysTime();

	char_array buffer = this->data;
	char_array line, token;

	this->trim_left(buffer, FileReader::EOL);

	/*read the header to get the names of  samples*/
	line = this->get_token(buffer, FileReader::EOL);
	if (line.size <= 0) {
		fprintf(stderr, "The file is incomplete\n");
		return false;
	}
	auto etime = getSysTime();
	ROOT_PRINT("load 1st line in %f sec\n", get_duration_s(stime, etime));

	stime = getSysTime();
	/*analyze the header.  first entry is skipped.  save the sample names */
	int numSamples = 0;
	token = this->get_token_or_empty(line, FileReader::COMMA);  // skip first one.  this is column name
	token = this->get_token_or_empty(line, FileReader::COMMA);  
	for (; (token.start != nullptr) && (numSamples < vectorSize); 
		token = this->get_token_or_empty(line, FileReader::COMMA), ++numSamples) {
		samples.emplace_back(std::string(token.start, token.size));
	}
	/*check consistency*/
	if (numSamples < vectorSize) {
		fprintf(stderr,
				"ERROR The number of samples (%d) read is less than vectorSize (%d)\n",
				numSamples, vectorSize);
		return false;
	}
	etime = getSysTime();
	ROOT_PRINT("parse column headers %d in %f sec\n", numSamples, get_duration_s(stime, etime));

	stime = getSysTime();

	/*get gene expression profiles*/
	/*extract gene expression values*/  // WAS READING TRANSPOSED.  NO LONGER.
	/* input is column major (row is 1 gene).  memory is row major (row is 1 sample) */
	FloatType * vec;
	int numGenes = 0;
	// get just the non-empty lines
	line = this->get_token(buffer, FileReader::EOL);
	for (; (line.start != nullptr)  && (numGenes < numVectors);
		line = this->get_token(buffer, FileReader::EOL),
		++numGenes) {

		// parse the row name
		token = this->get_token_or_empty(line, FileReader::COMMA);
		genes.emplace_back(std::string(token.start, token.size));

		// parse the rest of data.
		vec = reinterpret_cast<FloatType*>(reinterpret_cast<unsigned char *>(vectors) + numGenes * stride_bytes);
		numSamples = 0;
		token = this->get_token_or_empty(line, FileReader::COMMA);		  
		for (; (token.start != nullptr) && (numSamples < vectorSize); 
			token = this->get_token_or_empty(line, FileReader::COMMA), ++numSamples, ++vec) {
			// std::string s(token.start, token.size);
			// *(vec) = atof(s.c_str()); // will read until a non-numeric char is encountered.
			if (token.size > 0)
				*(vec) = atof(token.start); // will read until a non-numeric char is encountered.
		}
		// NOTE: missing entries are treated as 0.
	}
	/*consistency check*/
	if (numGenes < numVectors) {
		fprintf(stderr,
				"ERROR The number of genes (%d) read is less than numVectors (%d)\n",
				numGenes, numVectors);
		return false;
	}
	etime = getSysTime();
	ROOT_PRINT("load values in %f sec\n", get_duration_s(stime, etime));

	return true;
}



// #ifdef USE_MPI

// template <typename FloatType>
// bool CSVMatrixReader<FloatType>::getMatrixSize_impl(int& numVectors, int& vectorSize, 
// 		MPI_Comm comm) {

// 	char* buffer = NULL, *tok;
// 	// size_t bufferSize = 0;
// 	int numChars; //, index;
// 	// bool firstEntry;
// 	const char delim[] = ",";
	
// 	// MPI stuff.
// 	ssize_t filesize;
// 	int rank, procs;
// 	MPI_Comm_rank(comm, &rank);
// 	MPI_Comm_size(comm, &procs);
// 	int result;

// 	// -------- open file
// 	MPI_File fh;
// 	result = MPI_File_open(comm, fileName.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
//   	if(result != MPI_SUCCESS) 
//     	fprintf(stderr, "ERROR: MPI_File_open failed for %s\n", fileName.c_str());

// 	// --------- get file size
// 	if (rank == 0) {
// 		MPI_Offset temp;
// 		result = MPI_File_get_size(fh, &temp);
// 		filesize = temp;
// 	}
// 	MPI_Bcast(&filesize, 1, MPI_LONG, 0, comm);

// 	// next partition
// 	size_t bytes_per_proc = filesize / procs;
// 	size_t bytes_rem = filesize % procs;
// 	MPI_Offset offset = bytes_per_proc * rank;
// 	if (static_cast<size_t>(rank) < bytes_rem) {
// 		offset += rank;
// 		bytes_per_proc += 1;
// 	} else {
// 		offset += bytes_rem;
// 	}
	
// 	// allocate buffer. 
// 	char * read_buffer = reinterpret_cast<char *>(splash::utils::aalloc((bytes_per_proc + 2) * sizeof(char)));
// 	// read
// 	MPI_Status status;
// 	result = MPI_File_read_at_all(fh, offset, read_buffer, bytes_per_proc, MPI_BYTE, &status);
// 	if(result != MPI_SUCCESS) 
//     	fprintf(stderr, "ERROR: MPI_File_read_at failed for rank %d at offset %lld for length %lu\n", rank, offset, bytes_per_proc);
// 	int bytes_read;
// 	result = MPI_Get_elements(&status, MPI_BYTE, &bytes_read);
//   	if(result != MPI_SUCCESS)
//     	fprintf(stderr, "MPI_Get_elements failed to get bytes_read\n");
// 	MPI_File_close(&fh);


// 	// move 1 byte to the left.  this allows all true empty lines to be ignored.  otherwise can't tell newline at start of read_buffer from tree empty lines.
// 	char send = read_buffer[0];
// 	if (rank == 0) {
// 		// add an '\n' in case the files has missing \n
// 		send = '\n';
// 	}
// 	int left = (rank + procs - 1) % procs;
// 	int right = (rank + 1) % procs;

// 	// move data
// 	MPI_Sendrecv(&send, 1, MPI_BYTE, left, 1,
// 				read_buffer + bytes_read, 1, MPI_BYTE, right, 1, comm, &status);
// 	// terminate with 0
// 	read_buffer[bytes_read + 1] = 0;
	
// 	// ------ count number of lines. (MPI_Allreduce)
// 	// how to count empty lines?

// 	// ------ and number of columns (rank 0, MPI_Bcast)

// 	char * fullbuf = read_buffer;
// 	ssize_t max = bytes_read + 1;


// 	/*read the header to get the number of samples*/
// 	numVectors = vectorSize = 0;
// 	bool err = false;
// 	if (rank == 0) {
// 		// set to first non-eol character.
// 		numChars = getline_start(fullbuf, max);
// 		if (numChars < 0) {
// 			fprintf(stderr, "Incomplete file at line %d, count %d\n", __LINE__, numChars);
// 			fflush(stderr);
// 			err = true;
// 		}

// 		numChars = getline(fullbuf, max, buffer);
// 		if (numChars < 0) {
// 			fprintf(stderr, "The processor has incomplete data, count %d\n", numChars);
// 			fflush(stderr);
// 			err = true;
// 		}
// 		buffer[numChars] = 0;  // mark end of line for strtok.

// 		/*analyze the header.  first entry are gene */
// 		/*save sample names*/
// 		for (tok = strtok(buffer, delim); tok != NULL; tok = strtok(NULL, delim)) {
// 			vectorSize++;
// 		}
// 		vectorSize -= 1;

// 	}
// 	MPI_Bcast(&vectorSize, 1, MPI_INT, 0, comm);
// 	if (rank == 0)	fprintf(stderr, "Number of samples: %d\n", vectorSize);

// 	if (err) return false;

// 	/*get gene expression profiles*/
// 	// char * last;
// 	while ((numChars = getline(fullbuf, max, buffer)) != -1) {

// 		if ((numChars == 0) || ((numChars == 1) && (buffer[0] == '\r'))) {
// 			continue;   // empty line
// 			// note if EOL is at beginning of buffer, and rank > 0, then the EOL would have been sent to rank-1.
// 			//   the EOL would be counted in the previous page.
// 		} 
// 		if (buffer[numChars] == '\n') {
// 			// buffer[20] = 0;
// 			// fprintf(stderr, "[%d] first line %s\n", rank, buffer);
// 			// if ((rank == 3) && (numVectors == 0)) {
// 			// 	buffer[numChars] = 0;
// 			// 	fprintf(stderr, "[%d] first line %s\n", rank, buffer);
// 			// }
// 			// if ((rank == 2) && (numVectors == 32)) {
// 			// 	buffer[numChars] = 0;
// 			// 	fprintf(stderr, "[%d] first line %s\n", rank, buffer);
// 			// }

// 			++numVectors;   // line with EOL.
// 			// last = buffer;
// 		}

// 	}
// 	// last[10] = 0;
// 	// fprintf(stderr, "[%d] last line %s.\n", rank, last);

// 	fprintf(stderr, "rank %d Number of gene expression profiles: %d\n", rank, numVectors);
// 	fflush(stderr);
// 	// allreduce
// 	MPI_Allreduce(MPI_IN_PLACE, &numVectors, 1, MPI_INT, MPI_SUM, comm);
// 	if (rank == 0)	fprintf(stderr, "Number of gene expression profiles: %d\n", numVectors);

// 	splash::utils::afree(read_buffer);

// 	return true;

// }


// template<typename FloatType>
// bool CSVMatrixReader<FloatType>::loadMatrixData_impl(
// 		vector<string>& genes, vector<string>& samples, FloatType* vectors,
// 		const int numVectors, const int vectorSize, 
// 		const int stride_bytes, MPI_Comm comm) {	
// 	char* buffer = NULL, *tok;
// 	// size_t bufferSize = 0;
// 	int numChars, index;
// 	// bool firstEntry;
// 	const char delim[] = ",";
	
// 	// MPI stuff.
// 	ssize_t filesize;
// 	int rank, procs;
// 	MPI_Comm_rank(comm, &rank);
// 	MPI_Comm_size(comm, &procs);
// 	int result;

// 	// -------- open file
// 	MPI_File fh;
// 	result = MPI_File_open(comm, fileName.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
//   	if(result != MPI_SUCCESS) 
//     	fprintf(stderr, "ERROR: MPI_File_open failed for %s\n", fileName.c_str());

// 	// --------- get file size
// 	if (rank == 0) {
// 		MPI_Offset temp;
// 		result = MPI_File_get_size(fh, &temp);
// 		filesize = temp;
// 	}
// 	MPI_Bcast(&filesize, 1, MPI_LONG, 0, comm);

// 	// next partition
// 	size_t bytes_per_proc = filesize / procs;
// 	size_t bytes_rem = filesize % procs;
// 	MPI_Offset offset = bytes_per_proc * rank;
// 	if (static_cast<size_t>(rank) < bytes_rem) {
// 		offset += rank;
// 		bytes_per_proc += 1;
// 	} else {
// 		offset += bytes_rem;
// 	}
	
// 	// allocate buffer.  over provision by 12.5%.
// 	char * read_buffer = reinterpret_cast<char *>(splash::utils::aalloc((filesize + 2) * sizeof(char)));
// 	// read
// 	MPI_Status status;
// 	result = MPI_File_read_at_all(fh, offset, read_buffer + offset, bytes_per_proc, MPI_BYTE, &status);
// 	if(result != MPI_SUCCESS) 
//     	fprintf(stderr, "ERROR: MPI_File_read_at failed for rank %d at offset %lld for length %lu\n", rank, offset, bytes_per_proc);
// 	int bytes_read;
// 	result = MPI_Get_elements(&status, MPI_BYTE, &bytes_read);
//   	if(result != MPI_SUCCESS)
//     	fprintf(stderr, "MPI_Get_elements failed to get bytes_read\n");


// 	// ======= allgatherv the data, then parse.  instead of parse, then gather, because we have column major data.
// 	int * recvcounts = reinterpret_cast<int *>(splash::utils::aalloc(procs * sizeof(int)));
// 	int * displs = reinterpret_cast<int *>(splash::utils::aalloc(procs * sizeof(int)));
// 	recvcounts[rank] = bytes_read;
// 	MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, recvcounts, 1, MPI_INT, comm);
// 	displs[0] = 0;
// 	for (int i = 1; i < procs; ++i) {
// 		displs[i] = displs[i-1] + recvcounts[i-1];
// 		// if (rank == 0) fprintf(stderr, "%d: count %d displ %d\n", i-1, recvcounts[i-1], displs[i-1]);
// 	}
// 	// if (rank == 0) fprintf(stderr, "%d: count %d displ %d\n", procs-1, recvcounts[procs-1], displs[procs-1]);
	
// 	MPI_Allgatherv(MPI_IN_PLACE, bytes_read, MPI_BYTE, read_buffer, recvcounts, displs, MPI_BYTE, comm);
// 	bytes_read = displs[procs - 1] + recvcounts[procs - 1];
// 	read_buffer[bytes_read] = '\n';
// 	read_buffer[bytes_read + 1] = 0;  // zero terminated.
// 	splash::utils::afree(recvcounts);
// 	splash::utils::afree(displs);

// 	// // scan for first endline.  assumption is that send_count << bytes_read.
// 	// int send_count = 0;
// 	// if (rank > 0) {
// 	// 	// find start of end of line.
// 	// 	send_count = get_next_eol(read_buffer, 0, bytes_read);
// 	// 	// next normal character.
// 	// 	send_count += get_next_noneol(read_buffer, send_count, bytes_read);
// 	// }

// 	// // move data as needed.
// 	// // get send/recv counts.
// 	// int recv_count = 0;
// 	// int left = (rank + procs - 1) % procs;
// 	// int right = (rank + 1) % procs;
// 	// MPI_Sendrecv(send_count, 1, MPI_INT, left, 0,
// 	// 			recv_count, 1, MPI_INT, right, 0, comm, &status);

// 	// // move data
// 	// MPI_Sendrecv(read_buffer, send_count, MPI_BYTE, left, 1,
// 	// 			read_buffer + bytes_read, recv_count, MPI_BYTE, right, 1, comm, &status);
// 	// // then move within memory.
// 	// bytes_read = bytes_read - send_count + recv_count;

// 	MPI_File_close(&fh);



// 	// ======= DATA NOW IN MEMORY.  PARSE.
// 	// now parse.

// 	int numGenes = 0;
// 	int numSamples = 0;

// 	char * fullbuf = read_buffer;
// 	ssize_t max = bytes_read + 1;

// 	// set to first non-eol character.
// 	numChars = getline_start(fullbuf, max);
// 	if (numChars < 0) {
// 		fprintf(stderr, "Incomplete file at line %d\n", __LINE__);
// 		return false;
// 	}


// 	/*read the header to get the number of samples*/
// 	numChars = getline(fullbuf, max, buffer);
// 	if (numChars <= 0) {
// 		fprintf(stderr, "The processor has incomplete data\n");
// 		return false;
// 	}
// 	buffer[numChars] = 0;  // mark end of line for strtok.

// 	/*analyze the header.  first entry is gene */
// 	tok = strtok(buffer, delim);
// 	if(tok == NULL){
// 		fprintf(stderr, "Incomplete header at line %d\n", __LINE__);
// 		return false;
// 	}
// 	/*save sample names*/
// 	for (tok = strtok(NULL, delim); tok != NULL; tok = strtok(NULL, delim)) {
// 		samples.push_back(string(tok));
// 		numSamples++;
// 	}
// 	/*check consistency*/
// 	if ((numSamples != vectorSize) || (static_cast<size_t>(numSamples) != samples.size())) {
// 		fprintf(stderr,
// 				"The number of samples (%d) not equal to number of vectors (%d) sampels size %lu\n",
// 				numSamples, vectorSize, samples.size());
// 		return false;
// 	}


// 	/*get gene expression profiles*/
// 	numGenes = 0;
// 	FloatType* vec;
// 	while ((numChars = getline(fullbuf, max, buffer)) != -1) {

// 		if ((numChars == 0) || ((numChars == 1) && (buffer[0] == '\r'))) {
// 			continue;   // empty line
// 		} else if (buffer[numChars] == 0) {
// 			// incomplete line.  should not be here since we have appended \n at the end,
// 			continue;
// 		}
// 		// full line.  process.
// 		buffer[numChars] = 0;

// 		/*consistency check*/
// 		if (numGenes >= numVectors) {
// 			fprintf(stderr,
// 					"Error: rank %d number of genes (%d) is about to exceed (%d)\n", rank, numGenes, numVectors);
// 			return false;
// 		}
	
// 		/*skip the first column*/
// 		tok = strtok(buffer, delim);
// 		if(tok == NULL){
// 			fprintf(stderr, "incomplete file at line %d\n", __LINE__);
// 			return false;
// 		}
// 		/*save the locus id*/
// 		genes.push_back(string(tok));

// 		/*extract gene expression values*/
// 		/* input is row major (row is 1 gene).  memory is row major (row is 1 gene) */
// 		vec = reinterpret_cast<FloatType*>(reinterpret_cast<unsigned char*>(vectors) + numGenes * stride_bytes);
// 		index = 0;
// 		for (tok = strtok(NULL, delim); tok != NULL;
// 				tok = strtok(NULL, delim)) {
	
// 			if (index >= vectorSize) {
// 				break;
// 			}

// 			/*save the value*/
// 			*(vec) = atof(tok);
// 			++vec;

// 			/*increase the index*/
// 			++index;
// 		}

// 		// if (rank == 0) {
// 		// 	buffer[20] = 0;
// 		// 	fprintf(stderr, "[%d] row %d first:  %s\n", rank, numGenes, buffer);
// 		// }


// 		/*increase the gene index*/
// 		++numGenes;

// 	}

// 	if ((numGenes != numVectors) || (static_cast<size_t>(numGenes) != genes.size())) {
// 		fprintf(stderr,
// 				"Error: number of genes (%d) is inconsistent with numVectors (%d) and  gene size %lu\n", numGenes, numVectors, genes.size());
// 		return false;
// 	}
// 	splash::utils::afree(read_buffer);

// 	return true;
// }

// #endif // with_mpi.


}}