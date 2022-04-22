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
 * Author(s): Yongchao Liu, Tony C. Pan
 */

#pragma once

#include "splash/io/CustomFileReader.hpp"
#include <string>
#include <vector>
#include "splash/ds/aligned_matrix.hpp"
#include "splash/utils/report.hpp"

using namespace std;

#ifndef EXP_SKIP_TWO_ROWS
#define EXP_SKIP_TWO_ROWS false
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif  // with mpi

namespace splash { namespace io { 

/*use the same format with ARACNE*/
template<typename FloatType>
class EXPMatrixReader {
protected:

	// get line, and return line ptr and size, plus update buffer and remaining buffer size.
	// input must be zero terminated.
	// excluded the \r and \n.  returned size is for real characters only.  can return 0 for empty
	static inline ssize_t getline(char * & buffer, ssize_t & buffer_size,  char * & line) {
		line = buffer;

		if (buffer_size <= 0) {
			return -1;  // empty.
		}

		ssize_t len = 0;
		// get eol
		for (; (len < buffer_size) && (buffer[len] != '\n') ; ++len);
		if (len < buffer_size) {
			// found an eol.
			buffer += len + 1;
			buffer_size -= (len + 1);  // could become zero.
			// buffer[len] == \n
		} else {
			// reached end of buffer, not a full line.
			buffer += len;
			buffer_size -= len;  //becomes 0
			// buffer[len] = 0;
		}

		return len;   // include the eol characters
	}
	static inline ssize_t getline_start(char * & buffer, ssize_t & buffer_size) {
		if (buffer_size <= 0) return -1;

		ssize_t len = 0;
		// get eol
		for (; (len < buffer_size) && ((buffer[len] == '\r') || (buffer[len] == '\n')); ++len);
		if (len == 0) {
			// current character is not EOL
			return len;
		} else if (len < buffer_size) {
			// found EOL.  skip it.
			++len;			
		} 

		buffer += len;
		buffer_size -= len;

		return len;   // include the eol characters
	}


public:
	/*get gene expression matrix size*/
#ifdef USE_MPI
	static bool getMatrixSize(string const & fileName, ssize_t& numVectors, ssize_t& vectorSize, const bool skip = EXP_SKIP_TWO_ROWS,
		MPI_Comm comm = MPI_COMM_WORLD) {
		return getMatrixSize_impl(fileName, numVectors, vectorSize, comm, skip);
#else
	static bool getMatrixSize(string const & fileName, ssize_t& numVectors, ssize_t& vectorSize, const bool skip = EXP_SKIP_TWO_ROWS) {
		return getMatrixSize_impl(fileName, numVectors, vectorSize, skip);
#endif
	}

	/*get the matrix data*/
#ifdef USE_MPI
	static bool loadMatrixData(string const & fileName, vector<string>& genes,
			vector<string>& samples, FloatType* vectors, const ssize_t & numVectors, const ssize_t & vectorSize,
			const ssize_t & stride_bytes, const bool skip = EXP_SKIP_TWO_ROWS, MPI_Comm comm = MPI_COMM_WORLD) {
		return loadMatrixData_impl(fileName, genes, samples, vectors, numVectors, vectorSize, stride_bytes,
			comm, skip);
#else
	static bool loadMatrixData(string const & fileName, vector<string>& genes,
			vector<string>& samples, FloatType* vectors, const ssize_t & numVectors, const ssize_t & vectorSize,
			const ssize_t & stride_bytes, const bool skip = EXP_SKIP_TWO_ROWS) {
		return loadMatrixData_impl(fileName, genes, samples, vectors, numVectors, vectorSize, stride_bytes, skip);
#endif
	}

	/*get the matrix data*/
#ifdef USE_MPI
	static bool loadMatrixData(string const & fileName, vector<string>& genes,
			vector<string>& samples, splash::ds::aligned_matrix<FloatType> & output,
			const bool skip = EXP_SKIP_TWO_ROWS, MPI_Comm comm = MPI_COMM_WORLD) {
		return loadMatrixData_impl(fileName, genes, samples, 
			output.data(), output.rows(), output.columns(), output.column_bytes(), 
			comm, skip);
#else
	static bool loadMatrixData(string const & fileName, vector<string>& genes,
			vector<string>& samples, splash::ds::aligned_matrix<FloatType> & output,
			const bool skip = EXP_SKIP_TWO_ROWS) {
		return loadMatrixData_impl(fileName, genes, samples, 
			output.data(), output.rows(), output.columns(), output.column_bytes(), skip);
#endif
	}


protected:
	/*get gene expression matrix size*/
	static bool getMatrixSize_impl(string const & fileName, ssize_t& numVectors, ssize_t& vectorSize, const bool skip = EXP_SKIP_TWO_ROWS);

	/*get the matrix data*/
	static bool loadMatrixData_impl(string const & fileName, vector<string>& genes,
			vector<string>& samples, FloatType* vectors, const ssize_t & numVectors, const ssize_t & vectorSize,
			const ssize_t & stride_bytes, const bool skip = EXP_SKIP_TWO_ROWS);

	/*get the matrix data*/
	// static bool loadMatrixData_impl(string const & fileName, vector<string>& genes,
	// 		vector<string>& samples, splash::ds::aligned_matrix<FloatType> & input, const bool skip = EXP_SKIP_TWO_ROWS);


#ifdef USE_MPI
	/*get gene expression matrix size*/
	static bool getMatrixSize_impl(string const & fileName, ssize_t& numVectors, ssize_t& vectorSize, 
		MPI_Comm comm,
		const bool skip = EXP_SKIP_TWO_ROWS);

	static bool loadMatrixData_impl(string const & fileName, vector<string>& genes,
			vector<string>& samples, FloatType* vectors, const ssize_t & numVectors, const ssize_t & vectorSize,
			const ssize_t & stride_bytes, MPI_Comm comm, const bool skip = EXP_SKIP_TWO_ROWS);

	// static bool loadMatrixData_impl(string const & fileName, vector<string>& genes,
	// 		vector<string>& samples, splash::ds::aligned_matrix<FloatType> & input, MPI_Comm comm, const bool skip = EXP_SKIP_TWO_ROWS);
#endif
};

template<typename FloatType>
bool EXPMatrixReader<FloatType>::getMatrixSize_impl(string const & fileName, 
		ssize_t& numVectors, ssize_t& vectorSize, const bool skip) {
	char* buffer = NULL, *tok;
	size_t bufferSize = 0;
	ssize_t numChars;
	const char delim[] = "\t";
	splash::io::CustomFileReader fileReader;

	/*open the file*/
	if (!fileReader.open(fileName.c_str(), "rb")) {
		FMT_PRINT_RT("ERROR: Failed to open file {}\n", fileName.c_str());
		return false;
	}

	numVectors = vectorSize = 0;
	/*read the header to get the number of samples*/
	numChars = fileReader.getline(&buffer, &bufferSize);
	if (numChars <= 0) {
		FMT_PRINT_RT("ERROR: The file is incomplete\n");
		fileReader.close();
		return false;
	}

	/*analyze the header on the first row*/
	for (tok = strtok(buffer, delim); tok != NULL; tok = strtok(NULL, delim)) {
		vectorSize++;
	}
	vectorSize -= 2; /*exclude the first columns of the header: prob id and locus id*/
	FMT_ROOT_PRINT("Number of samples: {}\n", vectorSize);

	if(skip){
		/*skip the second and the third rows*/
		if(fileReader.getline(&buffer, &bufferSize) <= 0){
			FMT_PRINT_RT("ERROR: EXP file is incomplete at the second row\n");
			fileReader.close();
			return false;
		}
		if(fileReader.getline(&buffer, &bufferSize) <= 0){
			FMT_PRINT_RT("ERROR: EXP file is incomplete at the third row\n");
			fileReader.close();
			return false;
		}
	}

	/*get gene expression profiles*/
	while ((numChars = fileReader.getline(&buffer, &bufferSize)) != -1) {
		/*empty line*/
		if (numChars == 0) {
			continue;
		}
		++numVectors;
	}
	FMT_ROOT_PRINT("Number of gene expression profiles: {}\n", numVectors);

	/*close the file*/
	fileReader.close();

	if (buffer != NULL)
		free(buffer);
	return true;
}

template<typename FloatType>
bool EXPMatrixReader<FloatType>::loadMatrixData_impl(string const & fileName,
		vector<string>& genes, vector<string>& samples, FloatType* vectors,
		const ssize_t & numVectors, const ssize_t & vectorSize, 
		const ssize_t & stride_bytes, const bool skip) {
	char* buffer = NULL, *tok;
	size_t bufferSize = 0;
	ssize_t numChars, index;
	// bool firstEntry;
	const char delim[] = "\t";
	splash::io::CustomFileReader fileReader;

	/*open the file*/
	if (!fileReader.open(fileName.c_str(), "rb")) {
		FMT_PRINT_RT("ERROR: Failed to open file {}\n", fileName.c_str());
		return false;
	}

	ssize_t numGenes = 0;
	ssize_t numSamples = 0;
	/*read the header to get the number of samples*/
	numChars = fileReader.getline(&buffer, &bufferSize);
	if (numChars <= 0) {
		FMT_PRINT_RT("ERROR: The file is incomplete\n");
		fileReader.close();
		return false;
	}

	/*analyze the header.  first 2 entries are gene and id */
	tok = strtok(buffer, delim);
	if(tok == NULL){
		FMT_PRINT_RT("ERROR: Incomplete header at line {}\n", __LINE__);
		fileReader.close();
		return false;
	}
	tok = strtok(NULL, delim);
	if(tok == NULL){
		FMT_PRINT_RT("ERROR: Incomplete header at line {}\n", __LINE__);
		fileReader.close();
		return false;
	}
	/*save sample names*/
	for (tok = strtok(NULL, delim); tok != NULL; tok = strtok(NULL, delim)) {
		samples.push_back(string(tok));
		numSamples++;
	}

	/*check consistency*/
	if (numSamples != vectorSize) {
		FMT_PRINT_RT("The number of samples ({}) not equal to number of vectors ({})\n",
				numSamples, vectorSize);
		fileReader.close();
		return false;
	}

	if(skip){
		/*skip the second and third rows*/
		if(fileReader.getline(&buffer, &bufferSize) <= 0){
			FMT_PRINT_RT("ERROR: Incomplete file at line {}\n", __LINE__);
			fileReader.close();
			return false;
		}
		if(fileReader.getline(&buffer, &bufferSize) <= 0){
			FMT_PRINT_RT("ERROR: Incomplete file at line {}\n", __LINE__);
			fileReader.close();
			return false;
		}
	}

	/*get gene expression profiles*/
	numGenes = 0;
	FloatType* vec;
	while ((numChars = fileReader.getline(&buffer, &bufferSize)) != -1) {
		/*empty line*/
		if (numChars == 0) {
			continue;
		}
		/*consistency check*/
		if (numGenes >= numVectors) {
			FMT_PRINT_RT("Error: number of genes ({}) is not equal to ({})\n", numGenes, numVectors);
			fileReader.close();
			return false;
		}

		/*skip the first two columns*/
		tok = strtok(buffer, delim);
		if(tok == NULL){
			FMT_PRINT_RT("ERROR: incomplete file at line {}\n", __LINE__);
			fileReader.close();
			return false;
		}
		/*save the locus id*/
		genes.push_back(string(tok));


		tok = strtok(NULL, delim);
		if(tok == NULL){
			FMT_PRINT_RT("ERROR: incomplete file at line {}\n", __LINE__);
			fileReader.close();
			return false;
		}

		/*extract gene expression values*/  // WAS READING TRANSPOSED.  NO LONGER.
		/* input is column major (row is 1 gene).  memory is row major (row is 1 sample) */
		vec = reinterpret_cast<FloatType*>(reinterpret_cast<unsigned char *>(vectors) + numGenes * stride_bytes);
		index = 0;
		for (tok = strtok(NULL, delim); tok != NULL;
				tok = strtok(NULL, delim)) {
	
			if (index >= vectorSize) {
				break;
			}

			/*save the value*/
			*(vec) = atof(tok);
			++vec;

			/*increase the index*/
			++index;
		}

		/*increase the gene index*/
		++numGenes;
	}
	if (numGenes != numVectors) {
		FMT_PRINT_RT("Error: number of genes ({}) is inconsistent with numVectors ({})\n", numGenes, numVectors);
		fileReader.close();
		return false;
	}

	/*close the file*/
	fileReader.close();

	return true;
}

// template<typename FloatType>
// bool EXPMatrixReader<FloatType>::loadMatrixData_impl(string const & fileName,
// 		vector<string>& genes, vector<string>& samples, 
// 		splash::ds::aligned_matrix<FloatType> & output, const bool skip) {
// 	char* buffer = NULL, *tok;
// 	size_t bufferSize = 0;
// 	int numChars, index;
// 	// bool firstEntry;
// 	const char delim[] = "\t";
// 	splash::io::CustomFileReader fileReader;

// 	/*open the file*/
// 	if (!fileReader.open(fileName.c_str(), "rb")) {
// 		FMT_PRINT_RT("ERROR: Failed to open file {}\n", fileName.c_str());
// 		return false;
// 	}

// 	int numGenes = 0;
// 	int numSamples = 0;
// 	/*read the header to get the number of samples*/
// 	numChars = fileReader.getline(&buffer, &bufferSize);
// 	if (numChars <= 0) {
// 		FMT_PRINT_RT("ERROR: The file is incomplete\n");
// 		fileReader.close();
// 		return false;
// 	}

// 	/*analyze the header.  first 2 entries are gene and id */
// 	tok = strtok(buffer, delim);
// 	if(tok == NULL){
// 		FMT_PRINT_RT("ERROR: Incomplete header at line {}\n", __LINE__);
// 		fileReader.close();
// 		return false;
// 	}
// 	tok = strtok(NULL, delim);
// 	if(tok == NULL){
// 		FMT_PRINT_RT("ERROR: Incomplete header at line {}\n", __LINE__);
// 		fileReader.close();
// 		return false;
// 	}
// 	/*save sample names*/
// 	for (tok = strtok(NULL, delim); tok != NULL; tok = strtok(NULL, delim)) {
// 		samples.push_back(string(tok));
// 		numSamples++;
// 	}

// 	/*check consistency*/
// 	if (numSamples != output.columns()) {
// 		FMT_PRINT_RT("The number of samples ({}) not equal to number of vectors ({})\n",
// 				numSamples, output.columns());
// 		fileReader.close();
// 		return false;
// 	}

// 	if(skip){
// 		/*skip the second and third rows*/
// 		if(fileReader.getline(&buffer, &bufferSize) <= 0){
// 			FMT_PRINT_RT("ERROR: Incomplete file at line {}\n", __LINE__);
// 			fileReader.close();
// 			return false;
// 		}
// 		if(fileReader.getline(&buffer, &bufferSize) <= 0){
// 			FMT_PRINT_RT("ERROR: Incomplete file at line {}\n", __LINE__);
// 			fileReader.close();
// 			return false;
// 		}
// 	}

// 	/*get gene expression profiles*/
// 	numGenes = 0;
// 	FloatType* vec;
// 	while ((numChars = fileReader.getline(&buffer, &bufferSize)) != -1) {
// 		/*empty line*/
// 		if (numChars == 0) {
// 			continue;
// 		}
// 		/*consistency check*/
// 		if (numGenes >= output.rows()) {
// 			FMT_PRINT_RT("Error: number of genes ({}) is not equal to ({})\n", numGenes, output.rows());
// 			fileReader.close();
// 			return false;
// 		}

// 		/*skip the first two columns*/
// 		tok = strtok(buffer, delim);
// 		if(tok == NULL){
// 			FMT_PRINT_RT("ERROR: incomplete file at line {}\n", __LINE__);
// 			fileReader.close();
// 			return false;
// 		}
// 		/*save the locus id*/
// 		genes.push_back(string(tok));


// 		tok = strtok(NULL, delim);
// 		if(tok == NULL){
// 			FMT_PRINT_RT("ERROR: incomplete file at line {}\n", __LINE__);
// 			fileReader.close();
// 			return false;
// 		}

// 		/*extract gene expression values*/  // WAS READING TRANSPOSED.  NO LONGER.
// 		/* input is column major (row is 1 gene).  memory is row major (row is 1 sample) */
// 		vec = output.data(numGenes);
// 		index = 0;
// 		for (tok = strtok(NULL, delim); tok != NULL;
// 				tok = strtok(NULL, delim)) {
	
// 			if (index >= output.columns()) {
// 				break;
// 			}

// 			/*save the value*/
// 			*(vec) = atof(tok);
// 			++vec;

// 			/*increase the index*/
// 			++index;
// 		}

// 		/*increase the gene index*/
// 		++numGenes;
// 	}
// 	if (numGenes != output.rows()) {
// 		FMT_PRINT_RT("Error: number of genes ({}) is inconsistent with output.rows() ({})\n", numGenes, output.rows());
// 		fileReader.close();
// 		return false;
// 	}

// 	/*close the file*/
// 	fileReader.close();

// 	return true;
// }



#ifdef USE_MPI

template <typename FloatType>
bool EXPMatrixReader<FloatType>::getMatrixSize_impl(string const & fileName, ssize_t& numVectors, ssize_t& vectorSize, 
		MPI_Comm comm,
		const bool skip) {

	char* buffer = NULL, *tok;
	// size_t bufferSize = 0;
	ssize_t numChars; //, index;
	// bool firstEntry;
	const char delim[] = "\t";
	
	// MPI stuff.
	ssize_t filesize;
	int rank, procs;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &procs);
	int result;

	// -------- open file
	MPI_File fh;
	result = MPI_File_open(comm, fileName.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
  	if(result != MPI_SUCCESS) 
    	FMT_PRINT_RT("ERROR: MPI_File_open failed for {}\n", fileName.c_str());

	// --------- get file size
	if (rank == 0) {
		MPI_Offset temp;
		result = MPI_File_get_size(fh, &temp);
		filesize = temp;
	}
	MPI_Bcast(&filesize, 1, MPI_LONG, 0, comm);

	// next partition
	size_t bytes_per_proc = filesize / procs;
	size_t bytes_rem = filesize % procs;
	MPI_Offset offset = bytes_per_proc * rank;
	if (static_cast<size_t>(rank) < bytes_rem) {
		offset += rank;
		bytes_per_proc += 1;
	} else {
		offset += bytes_rem;
	}
	
	// allocate buffer. 
	char * read_buffer = reinterpret_cast<char *>(splash::utils::aalloc((bytes_per_proc + 2) * sizeof(char)));
	// read
	MPI_Status status;
	result = MPI_File_read_at_all(fh, offset, read_buffer, bytes_per_proc, MPI_BYTE, &status);
	if(result != MPI_SUCCESS) 
    	FMT_PRINT_RT("ERROR: MPI_File_read_at failed for rank {} at offset {} for length {}\n", rank, offset, bytes_per_proc);
	int bytes_read;
	result = MPI_Get_elements(&status, MPI_BYTE, &bytes_read);
  	if(result != MPI_SUCCESS)
    	FMT_PRINT_RT("ERROR: MPI_Get_elements failed to get bytes_read\n");
	MPI_File_close(&fh);


	// move 1 byte to the left.  this allows all true empty lines to be ignored.  otherwise can't tell newline at start of read_buffer from tree empty lines.
	char send = read_buffer[0];
	if (rank == 0) {
		// add an '\n' in case the files has missing \n
		send = '\n';
	}
	int left = (rank + procs - 1) % procs;
	int right = (rank + 1) % procs;

	// move data
	MPI_Sendrecv(&send, 1, MPI_BYTE, left, 1,
				read_buffer + bytes_read, 1, MPI_BYTE, right, 1, comm, &status);
	// terminate with 0
	read_buffer[bytes_read + 1] = 0;
	
	// ------ count number of lines. (MPI_Allreduce)
	// how to count empty lines?

	// ------ and number of columns (rank 0, MPI_Bcast)

	char * fullbuf = read_buffer;
	ssize_t max = bytes_read + 1;


	/*read the header to get the number of samples*/
	numVectors = vectorSize = 0;
	bool err = false;
	if (rank == 0) {
		// set to first non-eol character.
		numChars = getline_start(fullbuf, max);
		if (numChars < 0) {
			FMT_PRINT_RT("ERROR: Incomplete file at line {}, count {}\n", __LINE__, numChars);
			fflush(stderr);
			err = true;
		}

		numChars = getline(fullbuf, max, buffer);
		if (numChars < 0) {
			FMT_PRINT_RT("ERROR: The processor has incomplete data, count {}\n", numChars);
			fflush(stderr);
			err = true;
		}
		buffer[numChars] = 0;  // mark end of line for strtok.

		/*analyze the header.  first 2 entries are gene and id */
		/*save sample names*/
		for (tok = strtok(buffer, delim); tok != NULL; tok = strtok(NULL, delim)) {
			vectorSize++;
		}
		vectorSize -= 2;

		// skip the lines is required.
		if(skip){
			/*skip the second and third rows*/
			/*skip the second and the third rows*/
			if(getline(fullbuf, max, buffer) < 0){
				FMT_PRINT_RT("ERROR: EXP file is incomplete at the second row, count {}\n", numChars);
				fflush(stderr);
				err = true;
			}
			if(getline(fullbuf, max, buffer) < 0){
				FMT_PRINT_RT("ERROR: EXP file is incomplete at the third row, count {}\n", numChars);
				fflush(stderr);
				err = true;
			}
		}
	}
	MPI_Bcast(&vectorSize, 1, MPI_INT, 0, comm);
	FMT_ROOT_PRINT("Number of samples: {}\n", vectorSize);

	if (err) return false;

	/*get gene expression profiles*/
	// char * last;
	while ((numChars = getline(fullbuf, max, buffer)) != -1) {

		if ((numChars == 0) || ((numChars == 1) && (buffer[0] == '\r'))) {
			continue;   // empty line
			// note if EOL is at beginning of buffer, and rank > 0, then the EOL would have been sent to rank-1.
			//   the EOL would be counted in the previous page.
		} 
		if (buffer[numChars] == '\n') {
			// buffer[20] = 0;
			// FMT_ROOT_PRINT("[{}] first line {}\n", rank, buffer);
			// if ((rank == 3) && (numVectors == 0)) {
			// 	buffer[numChars] = 0;
			// 	FMT_ROOT_PRINT("[{}] first line {}\n", rank, buffer);
			// }
			// if ((rank == 2) && (numVectors == 32)) {
			// 	buffer[numChars] = 0;
			// 	FMT_ROOT_PRINT("[{}] first line {}\n", rank, buffer);
			// }

			++numVectors;   // line with EOL.
			// last = buffer;
		}

	}
	// last[10] = 0;
	// FMT_ROOT_PRINT("[{}] last line {}.\n", rank, last);

	// FMT_ROOT_PRINT("rank {} Number of gene expression profiles: {}\n", rank, numVectors);
	fflush(stderr);
	// allreduce
	MPI_Allreduce(MPI_IN_PLACE, &numVectors, 1, MPI_INT, MPI_SUM, comm);
	FMT_ROOT_PRINT("Number of gene expression profiles: {}\n", numVectors);

	splash::utils::afree(read_buffer);

	return true;

}


template<typename FloatType>
bool EXPMatrixReader<FloatType>::loadMatrixData_impl(string const & fileName,
		vector<string>& genes, vector<string>& samples, FloatType* vectors,
		const ssize_t & numVectors, const ssize_t & vectorSize, 
		const ssize_t & stride_bytes, MPI_Comm comm, const bool skip) {
	char* buffer = NULL, *tok;
	// size_t bufferSize = 0;
	ssize_t numChars, index;
	// bool firstEntry;
	const char delim[] = "\t";
	
	// MPI stuff.
	ssize_t filesize;
	int rank, procs;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &procs);
	int result;

	// -------- open file
	MPI_File fh;
	result = MPI_File_open(comm, fileName.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
  	if(result != MPI_SUCCESS) 
    	FMT_PRINT_RT("ERROR: MPI_File_open failed for {}\n", fileName.c_str());

	// --------- get file size
	if (rank == 0) {
		MPI_Offset temp;
		result = MPI_File_get_size(fh, &temp);
		filesize = temp;
	}
	MPI_Bcast(&filesize, 1, MPI_LONG, 0, comm);

	// next partition
	size_t bytes_per_proc = filesize / procs;
	size_t bytes_rem = filesize % procs;
	MPI_Offset offset = bytes_per_proc * rank;
	if (static_cast<size_t>(rank) < bytes_rem) {
		offset += rank;
		bytes_per_proc += 1;
	} else {
		offset += bytes_rem;
	}
	
	// allocate buffer.  over provision by 12.5%.
	char * read_buffer = reinterpret_cast<char *>(splash::utils::aalloc((filesize + 2) * sizeof(char)));
	// read
	MPI_Status status;
	result = MPI_File_read_at_all(fh, offset, read_buffer + offset, bytes_per_proc, MPI_BYTE, &status);
	if(result != MPI_SUCCESS) 
    	FMT_PRINT_RT("ERROR: MPI_File_read_at failed for rank {} at offset {} for length {}\n", rank, offset, bytes_per_proc);
	int bytes_read;
	result = MPI_Get_elements(&status, MPI_BYTE, &bytes_read);
  	if(result != MPI_SUCCESS)
    	FMT_PRINT_RT("ERROR: MPI_Get_elements failed to get bytes_read\n");


	// ======= allgatherv the data, then parse.  instead of parse, then gather, because we have column major data.
	int * recvcounts = reinterpret_cast<int *>(splash::utils::aalloc(procs * sizeof(int)));
	int * displs = reinterpret_cast<int *>(splash::utils::aalloc(procs * sizeof(int)));
	recvcounts[rank] = bytes_read;
	MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, recvcounts, 1, MPI_INT, comm);
	displs[0] = 0;
	for (int i = 1; i < procs; ++i) {
		displs[i] = displs[i-1] + recvcounts[i-1];
		// FMT_ROOT_PRINT("{}: count {} displ {}\n", i-1, recvcounts[i-1], displs[i-1]);
	}
	// FMT_ROOT_PRINT("{}: count {} displ {}\n", procs-1, recvcounts[procs-1], displs[procs-1]);
	
	MPI_Allgatherv(MPI_IN_PLACE, bytes_read, MPI_BYTE, read_buffer, recvcounts, displs, MPI_BYTE, comm);
	bytes_read = displs[procs - 1] + recvcounts[procs - 1];
	read_buffer[bytes_read] = '\n';
	read_buffer[bytes_read + 1] = 0;  // zero terminated.
	splash::utils::afree(recvcounts);
	splash::utils::afree(displs);

	// // scan for first endline.  assumption is that send_count << bytes_read.
	// int send_count = 0;
	// if (rank > 0) {
	// 	// find start of end of line.
	// 	send_count = get_next_eol(read_buffer, 0, bytes_read);
	// 	// next normal character.
	// 	send_count += get_next_noneol(read_buffer, send_count, bytes_read);
	// }

	// // move data as needed.
	// // get send/recv counts.
	// int recv_count = 0;
	// int left = (rank + procs - 1) % procs;
	// int right = (rank + 1) % procs;
	// MPI_Sendrecv(send_count, 1, MPI_INT, left, 0,
	// 			recv_count, 1, MPI_INT, right, 0, comm, &status);

	// // move data
	// MPI_Sendrecv(read_buffer, send_count, MPI_BYTE, left, 1,
	// 			read_buffer + bytes_read, recv_count, MPI_BYTE, right, 1, comm, &status);
	// // then move within memory.
	// bytes_read = bytes_read - send_count + recv_count;

	MPI_File_close(&fh);



	// ======= DATA NOW IN MEMORY.  PARSE.
	// now parse.

	ssize_t numGenes = 0;
	ssize_t numSamples = 0;

	char * fullbuf = read_buffer;
	ssize_t max = bytes_read + 1;

	// set to first non-eol character.
	numChars = getline_start(fullbuf, max);
	if (numChars < 0) {
		FMT_PRINT_RT("ERROR: Incomplete file at line {}\n", __LINE__);
		return false;
	}


	/*read the header to get the number of samples*/
	numChars = getline(fullbuf, max, buffer);
	if (numChars <= 0) {
		FMT_PRINT_RT("ERROR: The processor has incomplete data\n");
		return false;
	}
	buffer[numChars] = 0;  // mark end of line for strtok.

	/*analyze the header.  first 2 entries are gene and id */
	tok = strtok(buffer, delim);
	if(tok == NULL){
		FMT_PRINT_RT("ERROR: Incomplete header at line {}\n", __LINE__);
		return false;
	}
	tok = strtok(NULL, delim);
	if(tok == NULL){
		FMT_PRINT_RT("ERROR: Incomplete header at line {}\n", __LINE__);
		return false;
	}
	/*save sample names*/
	for (tok = strtok(NULL, delim); tok != NULL; tok = strtok(NULL, delim)) {
		samples.push_back(string(tok));
		numSamples++;
	}
	/*check consistency*/
	if ((numSamples != vectorSize) || (static_cast<size_t>(numSamples) != samples.size())) {
		FMT_PRINT_RT("The number of samples ({}) not equal to number of vectors ({}) sampels size {}\n",
				numSamples, vectorSize, samples.size());
		return false;
	}


	if(skip){
		/*skip the second and third rows*/
		for ( int i = 0; i < 2; ++i) {
			numChars = getline(fullbuf, max, buffer);

			if(numChars < 0){
				FMT_PRINT_RT("ERROR: Incomplete file at line {}\n", __LINE__);
				return false;
			}
		}		
	}


	/*get gene expression profiles*/
	numGenes = 0;
	FloatType* vec;
	while ((numChars = getline(fullbuf, max, buffer)) != -1) {

		if ((numChars == 0) || ((numChars == 1) && (buffer[0] == '\r'))) {
			continue;   // empty line
		} else if (buffer[numChars] == 0) {
			// incomplete line.  should not be here since we have appended \n at the end,
			continue;
		}
		// full line.  process.
		buffer[numChars] = 0;

		/*consistency check*/
		if (numGenes >= numVectors) {
			FMT_PRINT_RT("Error: rank {} number of genes ({}) is about to exceed ({})\n", rank, numGenes, numVectors);
			return false;
		}
	
		/*skip the first two columns*/
		tok = strtok(buffer, delim);
		if(tok == NULL){
			FMT_PRINT_RT("ERROR: incomplete file at line {}\n", __LINE__);
			return false;
		}
		/*save the locus id*/
		genes.push_back(string(tok));


		tok = strtok(NULL, delim);
		if(tok == NULL){
			FMT_PRINT_RT("ERROR: incomplete file at line {}\n", __LINE__);
			return false;
		}

		/*extract gene expression values*/
		/* input is row major (row is 1 gene).  memory is row major (row is 1 gene) */
		vec = reinterpret_cast<FloatType*>(reinterpret_cast<unsigned char*>(vectors) + numGenes * stride_bytes);
		index = 0;
		for (tok = strtok(NULL, delim); tok != NULL;
				tok = strtok(NULL, delim)) {
	
			if (index >= vectorSize) {
				break;
			}

			/*save the value*/
			*(vec) = atof(tok);
			++vec;

			/*increase the index*/
			++index;
		}

		// 	buffer[20] = 0;
		// 	FMT_ROOT_PRINT("[{}] row {} first:  {}\n", rank, numGenes, buffer);


		/*increase the gene index*/
		++numGenes;

	}

	if ((numGenes != numVectors) || (static_cast<size_t>(numGenes) != genes.size())) {
		FMT_PRINT_RT("Error: number of genes ({}) is inconsistent with numVectors ({}) and  gene size {}\n", numGenes, numVectors, genes.size());
		return false;
	}
	splash::utils::afree(read_buffer);

	return true;
}

// template<typename FloatType>
// bool EXPMatrixReader<FloatType>::loadMatrixData_impl(string const & fileName,
// 		vector<string>& genes, vector<string>& samples, 
// 		splash::ds::aligned_matrix<FloatType> & output, MPI_Comm comm, const bool skip) {
// 	char* buffer = NULL, *tok;
// 	// size_t bufferSize = 0;
// 	int numChars, index;
// 	// bool firstEntry;
// 	const char delim[] = "\t";
	
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
//     	FMT_PRINT_RT("ERROR: MPI_File_open failed for {}\n", fileName.c_str());

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
// 	if (rank < bytes_rem) {
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
//     	FMT_PRINT_RT("ERROR: MPI_File_read_at failed for rank {} at offset {} for length {}\n", rank, offset, bytes_per_proc);
// 	int bytes_read;
// 	result = MPI_Get_elements(&status, MPI_BYTE, &bytes_read);
//   	if(result != MPI_SUCCESS)
//     	FMT_PRINT_RT("ERROR: MPI_Get_elements failed to get bytes_read\n");


// 	// ======= allgatherv the data, then parse.  instead of parse, then gather, because we have column major data.
// 	int * recvcounts = reinterpret_cast<int *>(splash::utils::aalloc(procs * sizeof(int)));
// 	int * displs = reinterpret_cast<int *>(splash::utils::aalloc(procs * sizeof(int)));
// 	recvcounts[rank] = bytes_read;
// 	MPI_Allgather(MPI_IN_PLACE, 1, MPI_INT, recvcounts, 1, MPI_INT, comm);
// 	displs[0] = 0;
// 	for (int i = 1; i < procs; ++i) {
// 		displs[i] = displs[i-1] + recvcounts[i-1];
// 		// FMT_ROOT_PRINT("{}: count {} displ {}\n", i-1, recvcounts[i-1], displs[i-1]);
// 	}
// 	// FMT_ROOT_PRINT("{}: count {} displ {}\n", procs-1, recvcounts[procs-1], displs[procs-1]);
	
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
// 		FMT_PRINT_RT("ERROR: Incomplete file at line {}\n", __LINE__);
// 		return false;
// 	}


// 	/*read the header to get the number of samples*/
// 	numChars = getline(fullbuf, max, buffer);
// 	if (numChars <= 0) {
// 		FMT_PRINT_RT("ERROR: The processor has incomplete data\n");
// 		return false;
// 	}
// 	buffer[numChars] = 0;  // mark end of line for strtok.

// 	/*analyze the header.  first 2 entries are gene and id */
// 	tok = strtok(buffer, delim);
// 	if(tok == NULL){
// 		FMT_PRINT_RT("ERROR: Incomplete header at line {}\n", __LINE__);
// 		return false;
// 	}
// 	tok = strtok(NULL, delim);
// 	if(tok == NULL){
// 		FMT_PRINT_RT("ERROR: Incomplete header at line {}\n", __LINE__);
// 		return false;
// 	}
// 	/*save sample names*/
// 	for (tok = strtok(NULL, delim); tok != NULL; tok = strtok(NULL, delim)) {
// 		samples.push_back(string(tok));
// 		numSamples++;
// 	}
// 	/*check consistency*/
// 	if ((numSamples != output.columns()) || (numSamples != samples.size())) {
// 		FMT_PRINT_RT("The number of samples ({}) not equal to number of vectors ({}) sampels size {}\n",
// 				numSamples, output.columns(), samples.size());
// 		return false;
// 	}


// 	if(skip){
// 		/*skip the second and third rows*/
// 		for ( int i = 0; i < 2; ++i) {
// 			numChars = getline(fullbuf, max, buffer);

// 			if(numChars < 0){
// 				FMT_PRINT_RT("ERROR: Incomplete file at line {}\n", __LINE__);
// 				return false;
// 			}
// 		}		
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
// 		if (numGenes >= output.rows()) {
// 			FMT_PRINT_RT("Error: rank {} number of genes ({}) is about to exceed ({})\n", rank, numGenes, output.rows());
// 			return false;
// 		}
	
// 		/*skip the first two columns*/
// 		tok = strtok(buffer, delim);
// 		if(tok == NULL){
// 			FMT_PRINT_RT("ERROR: incomplete file at line {}\n", __LINE__);
// 			return false;
// 		}
// 		/*save the locus id*/
// 		genes.push_back(string(tok));


// 		tok = strtok(NULL, delim);
// 		if(tok == NULL){
// 			FMT_PRINT_RT("ERROR: incomplete file at line {}\n", __LINE__);
// 			return false;
// 		}

// 		/*extract gene expression values*/
// 		/* input is row major (row is 1 gene).  memory is row major (row is 1 gene) */
// 		vec = output.data(numGenes);
// 		index = 0;
// 		for (tok = strtok(NULL, delim); tok != NULL;
// 				tok = strtok(NULL, delim)) {
	
// 			if (index >= output.columns()) {
// 				break;
// 			}

// 			/*save the value*/
// 			*(vec) = atof(tok);
// 			++vec;

// 			/*increase the index*/
// 			++index;
// 		}

// 		// 	buffer[20] = 0;
// 		// 	FMT_ROOT_PRINT("[{}] row {} first:  {}\n", rank, numGenes, buffer);


// 		/*increase the gene index*/
// 		++numGenes;

// 	}

// 	if ((numGenes != output.rows()) || (numGenes != genes.size())) {
// 		FMT_PRINT_RT("Error: number of genes ({}) is inconsistent with numVectors ({}) and  gene size {}\n", numGenes, output.rows(), genes.size());
// 		return false;
// 	}
// 	splash::utils::afree(read_buffer);

// 	return true;
// }
#endif // with_mpi.


}}