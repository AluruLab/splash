/*
 * EXPMatrixReader2.hpp
 *
 *  Created on: Aug 21, 2020
 *  Author: Tony Pan
 *  Affiliation: Institute for Data Engineering and Science
 *  						Georgia Institute of Technology, Atlanta, GA 30332
 */

// using the new file reader and atof.  data loading performance:
// clang:  1000x1000 exp array, i5-8350, 0.3219s down to 0.08444s
// g++: 0.08357s now.

#pragma once

#include <string>  // string
#include <algorithm> 
#include <vector>
#include "ds/aligned_matrix.hpp"  // matrix
#include "ds/char_array.hpp"  // char_array_template
#include "io/FileReader2.hpp"  // file reading.

#include "utils/benchmark.hpp"
#include "utils/report.hpp"
#include "utils/string_utils.hpp"  // atof - significant speedup

#ifndef EXP_SKIP_TWO_ROWS
#define EXP_SKIP_TWO_ROWS false
#endif

#ifdef USE_MPI
#include <mpi.h>
#endif  // with mpi

namespace splash { namespace io { 

/*use the same format with ARACNE.  tab delimited*/
template<typename FloatType>
class EXPMatrixReader2 : public FileReader2 {
	
		protected:
		int atof_type;

	public:

		EXPMatrixReader2(const char* filename, int const & _atof_type = 1) : FileReader2(filename), atof_type(_atof_type)  {}
		virtual ~EXPMatrixReader2() {}


#ifdef USE_MPI
		bool getMatrixSize(int& numVectors, int& vectorSize, const bool skip = EXP_SKIP_TWO_ROWS,
			MPI_Comm comm = MPI_COMM_WORLD) {
			return getMatrixSize_impl(numVectors, vectorSize, comm, skip);
		}
		bool loadMatrixData(std::vector<std::string>& genes,
				std::vector<std::string>& samples, FloatType* vectors, const int numVectors, const int vectorSize,
				const int stride_bytes, const bool skip = EXP_SKIP_TWO_ROWS, MPI_Comm comm = MPI_COMM_WORLD) {
			return loadMatrixData_impl( genes, samples, vectors, numVectors, vectorSize, stride_bytes,
				comm, skip);
		}
		bool loadMatrixData(std::vector<std::string>& genes,
				std::vector<std::string>& samples, splash::ds::aligned_matrix<FloatType> & output, const bool skip = EXP_SKIP_TWO_ROWS, 
				MPI_Comm comm = MPI_COMM_WORLD) {
			return loadMatrixData_impl( genes, samples, 
				output.data(), output.rows(), output.columns(), output.column_bytes(), 
				comm, skip);
		}
#else
		/*get gene expression matrix size*/
		bool getMatrixSize(int& numVectors, int& vectorSize, const bool skip = EXP_SKIP_TWO_ROWS) {
			return getMatrixSize_impl(numVectors, vectorSize, skip);
		}
		bool loadMatrixData(std::vector<std::string>& genes,
				std::vector<std::string>& samples, FloatType* vectors, const int numVectors, const int vectorSize,
				const int stride_bytes, const bool skip = EXP_SKIP_TWO_ROWS) {
			if (atof_type == 0)  // default
				return loadMatrixData_impl( genes, samples, vectors, numVectors, vectorSize, stride_bytes, skip);
			else // fast and precise atof
				return loadMatrixData_impl_fast( genes, samples, vectors, numVectors, vectorSize, stride_bytes, skip);
		}
		bool loadMatrixData(std::vector<std::string>& genes,
				std::vector<std::string>& samples, splash::ds::aligned_matrix<FloatType> & output, const bool skip = EXP_SKIP_TWO_ROWS) {
			return loadMatrixData( genes, samples, 
				output.data(), output.rows(), output.columns(), output.column_bytes(), skip);
		}

#endif



	protected:
		/*get gene expression matrix size*/
		bool getMatrixSize_impl(int& numVectors, int& vectorSize, const bool skip = EXP_SKIP_TWO_ROWS) {
				auto stime = getSysTime();

			splash::ds::char_array_template buffer = this->data;
			if (buffer.ptr == nullptr) {
				fprintf(stderr, "File not read\n");
				return false;
			}

			numVectors = vectorSize = 0;
			
			// get rid of empty lines.
			buffer.trim_left<LF>();

			/*read the header to get the number of samples*/
			splash::ds::char_array_template line = buffer.get_token<LF>();
			if (line.size <= 0) {
				fprintf(stderr, "The file is incomplete\n");
				return false;
			}

			/*analyze the header on the first row*/
			fprintf(stderr, "line size = %lu, ptr = %p\n", line.size, line.ptr);
			vectorSize = line.count_token_or_empty<TAB>() - 2;
			fprintf(stderr, "Number of samples: %d\n", vectorSize);

			if (skip) {
				buffer.get_token_or_empty<LF>();
				buffer.get_token_or_empty<LF>();
			}

			/*get gene expression profiles.  skip empty lines*/ 
			numVectors = buffer.count_token<LF>();
			fprintf(stderr, "Number of gene expression profiles: %d\n", numVectors);

			auto etime = getSysTime();
			ROOT_PRINT("get matrix size in %f sec\n", get_duration_s(stime, etime));
			return true;

		}

		/*get the matrix data*/
		bool loadMatrixData_impl(std::vector<std::string>& genes,
				std::vector<std::string>& samples, FloatType* vectors, const int & numVectors, const int & vectorSize,
				const int & stride_bytes, const bool skip = EXP_SKIP_TWO_ROWS) {
			auto stime = getSysTime();

			splash::ds::char_array_template buffer = this->data;
			splash::ds::char_array_template line, token;

			buffer.trim_left<LF>();

			/*read the header to get the names of  samples*/
			line = buffer.get_token<LF>();
			if (line.size <= 0) {
				fprintf(stderr, "The file is incomplete\n");
				return false;
			}
			auto etime = getSysTime();
			ROOT_PRINT("load 1st line in %f sec\n", get_duration_s(stime, etime));

			stime = getSysTime();
			/*analyze the header.  first entry is skipped.  save the sample names */
			int numSamples = 0;
			token = line.get_token_or_empty<TAB>();  // skip first 2.  this is column name and id
			token = line.get_token_or_empty<TAB>();  
			token = line.get_token_or_empty<TAB>();  
			for (; (token.ptr != nullptr) && (numSamples < vectorSize); 
				token = line.get_token_or_empty<TAB>(), ++numSamples) {
				samples.emplace_back(std::string(token.ptr, token.size));
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
			if (skip) {
				buffer.get_token_or_empty<LF>();
				buffer.get_token_or_empty<LF>();
			}

			/*get gene expression profiles*/
			/*extract gene expression values*/  // WAS READING TRANSPOSED.  NO LONGER.
			/* input is column major (row is 1 gene).  memory is row major (row is 1 sample) */
			FloatType * vec;
			int numGenes = 0;
			// get just the non-empty lines
			line = buffer.get_token<LF>();
			for (; (line.ptr != nullptr)  && (numGenes < numVectors);
				line = buffer.get_token<LF>(),
				++numGenes) {

				// parse the row name
				token = line.get_token_or_empty<TAB>();
				genes.emplace_back(std::string(token.ptr, token.size));
				// skip alias.
				token = line.get_token_or_empty<TAB>();		  

				// parse the rest of data.
				vec = reinterpret_cast<FloatType*>(reinterpret_cast<unsigned char *>(vectors) + numGenes * stride_bytes);
				numSamples = 0;
				token = line.get_token_or_empty<TAB>();		  
				for (; (token.ptr != nullptr) && (numSamples < vectorSize); 
					token = line.get_token_or_empty<TAB>(), ++numSamples, ++vec) {
					if (token.size > 0)
						*(vec) = atof(token.ptr); // will read until a non-numeric char is encountered.
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


		/*get the matrix data*/
		bool loadMatrixData_impl_fast(std::vector<std::string>& genes,
				std::vector<std::string>& samples, FloatType* vectors, const int & numVectors, const int & vectorSize,
				const int & stride_bytes, const bool skip = EXP_SKIP_TWO_ROWS) {
			auto stime = getSysTime();

			splash::ds::char_array_template buffer = this->data;
			splash::ds::char_array_template line, token;

			buffer.trim_left<LF>();

			/*read the header to get the names of  samples*/
			line = buffer.get_token<LF>();
			if (line.size <= 0) {
				fprintf(stderr, "The file is incomplete\n");
				return false;
			}
			auto etime = getSysTime();
			ROOT_PRINT("load 1st line in %f sec\n", get_duration_s(stime, etime));

			stime = getSysTime();
			/*analyze the header.  first entry is skipped.  save the sample names */
			int numSamples = 0;
			token = line.get_token_or_empty<TAB>();  // skip first 2.  this is column name and id
			token = line.get_token_or_empty<TAB>();  
			token = line.get_token_or_empty<TAB>();  
			for (; (token.ptr != nullptr) && (numSamples < vectorSize); 
				token = line.get_token_or_empty<TAB>(), ++numSamples) {
				samples.emplace_back(std::string(token.ptr, token.size));
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
			if (skip) {
				buffer.get_token_or_empty<LF>();
				buffer.get_token_or_empty<LF>();
			}

			/*get gene expression profiles*/
			/*extract gene expression values*/  // WAS READING TRANSPOSED.  NO LONGER.
			/* input is column major (row is 1 gene).  memory is row major (row is 1 sample) */
			FloatType * vec;
			int numGenes = 0;
			// get just the non-empty lines
			line = buffer.get_token<LF>();
			for (; (line.ptr != nullptr)  && (numGenes < numVectors);
				line = buffer.get_token<LF>(),
				++numGenes) {

				// parse the row name
				token = line.get_token_or_empty<TAB>();
				genes.emplace_back(std::string(token.ptr, token.size));
				// skip alias.
				token = line.get_token_or_empty<TAB>();		  

				// parse the rest of data.
				vec = reinterpret_cast<FloatType*>(reinterpret_cast<unsigned char *>(vectors) + numGenes * stride_bytes);
				numSamples = 0;
				token = line.get_token_or_empty<TAB>();		  
				for (; (token.ptr != nullptr) && (numSamples < vectorSize); 
					token = line.get_token_or_empty<TAB>(), ++numSamples, ++vec) {

					if (token.size > 0)
						*(vec) = splash::utils::atof(token.ptr); // will read until a non-numeric char is encountered.
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

		/*get the matrix data*/
		// bool loadMatrixData_impl(std::vector<std::string>& genes,
		// 		std::vector<std::string>& samples, splash::ds::aligned_matrix<FloatType> & input);



#ifdef USE_MPI
		/*get gene expression matrix size*/
		bool getMatrixSize_impl(int& numVectors, int& vectorSize, MPI_Comm comm,
			const bool skip = EXP_SKIP_TWO_ROWS) {
				return getMatrixSize_impl(numVectors, vectorSize, skip);
			}

		bool loadMatrixData_impl(std::vector<std::string>& genes,
				std::vector<std::string>& samples, FloatType* vectors, const int & numVectors, const int & vectorSize,
				const int & stride_bytes, MPI_Comm comm,
				const bool skip = EXP_SKIP_TWO_ROWS) {
			if (atof_type == 0)  // default
				return loadMatrixData_impl( genes, samples, vectors, numVectors, vectorSize, stride_bytes, skip);
			else // fast and precise atof
				return loadMatrixData_impl_fast( genes, samples, vectors, numVectors, vectorSize, stride_bytes, skip);
			}

		// bool loadMatrixData_impl(std::vector<std::string>& genes,
		// 		std::vector<std::string>& samples, splash::ds::aligned_matrix<FloatType> & input, MPI_Comm comm);
#endif


};


}}