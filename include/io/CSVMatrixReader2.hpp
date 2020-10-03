/*
 *  CSVMatrixReader2.hpp
 *
 *  Created on: Aug 21, 2020
 *  Author: Tony Pan
 *  Affiliation: Institute for Data Engineering and Science
 *  						Georgia Institute of Technology, Atlanta, GA 30332
 *  
 */

#pragma once

#include <string>  // string
#include <algorithm> 
#include <vector>
#include "ds/aligned_matrix.hpp"  // matrix
#include "ds/char_array.hpp"  // char_array_template
#include "io/FileReader2.hpp"  // file reading.

#include "utils/benchmark.hpp"
#include "utils/report.hpp"
#include "utils/string_utils.hpp"  // atof - significant speedup.

#ifdef USE_MPI
#include <mpi.h>
#endif  // with mpi

namespace splash { namespace io { 


template<typename FloatType>
class CSVMatrixReader2 : public FileReader2 {

	protected:
		int atof_type;

	public:

		CSVMatrixReader2(const char* filename, int const & _atof_type = 1) : FileReader2(filename), atof_type(_atof_type) {}
		virtual ~CSVMatrixReader2() {}

#ifdef USE_MPI
		bool getMatrixSize(ssize_t& numVectors, ssize_t& vectorSize,
			MPI_Comm comm = MPI_COMM_WORLD) {
			return getMatrixSize_impl(numVectors, vectorSize, comm);
		}
		bool loadMatrixData(std::vector<std::string>& genes,
				std::vector<std::string>& samples, FloatType* vectors, const ssize_t numVectors, const ssize_t vectorSize,
				const ssize_t stride_bytes, MPI_Comm comm = MPI_COMM_WORLD) {
			return loadMatrixData_impl( genes, samples, vectors, numVectors, vectorSize, stride_bytes,
				comm);
		}
		bool loadMatrixData(std::vector<std::string>& genes,
				std::vector<std::string>& samples, splash::ds::aligned_matrix<FloatType> & output, 
				MPI_Comm comm = MPI_COMM_WORLD) {
			return loadMatrixData_impl( genes, samples, 
				output.data(), output.rows(), output.columns(), output.column_bytes(), 
				comm);
		}
#else
		/*get gene expression matrix size*/
		bool getMatrixSize(ssize_t& numVectors, ssize_t& vectorSize) {
			return getMatrixSize_impl(numVectors, vectorSize);
		}
		bool loadMatrixData(std::vector<std::string>& genes,
				std::vector<std::string>& samples, FloatType* vectors, const ssize_t numVectors, const ssize_t vectorSize,
				const ssize_t stride_bytes) {
			if (atof_type == 0)  // default
				return loadMatrixData_impl( genes, samples, vectors, numVectors, vectorSize, stride_bytes);
			else // fast and precise atof
				return loadMatrixData_impl_fast( genes, samples, vectors, numVectors, vectorSize, stride_bytes);
		}
		bool loadMatrixData(std::vector<std::string>& genes,
				std::vector<std::string>& samples, splash::ds::aligned_matrix<FloatType> & output) {
			return loadMatrixData( genes, samples, 
				output.data(), output.rows(), output.columns(), output.column_bytes());
		}

#endif



	protected:
		/*get gene expression matrix size*/
		bool getMatrixSize_impl(ssize_t& numVectors, ssize_t& vectorSize) {
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
			vectorSize = line.count_token_or_empty<COMMA>() - 1;
			fprintf(stderr, "Number of samples: %ld\n", vectorSize);

			/*get gene expression profiles.  skip empty lines*/ 
			numVectors = buffer.count_token<LF>();
			fprintf(stderr, "Number of gene expression profiles: %ld\n", numVectors);

			auto etime = getSysTime();
			ROOT_PRINT("get matrix size in %f sec\n", get_duration_s(stime, etime));
			return true;

		}

		/*get the matrix data*/
		bool loadMatrixData_impl(std::vector<std::string>& genes,
				std::vector<std::string>& samples, FloatType* vectors, const ssize_t & numVectors, const ssize_t & vectorSize,
				const ssize_t & stride_bytes) {
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
			ssize_t numSamples = 0;
			token = line.get_token_or_empty<COMMA>();  // skip first one.  this is column name
			token = line.get_token_or_empty<COMMA>();  
			for (; (token.ptr != nullptr) && (numSamples < vectorSize); 
				token = line.get_token_or_empty<COMMA>(), ++numSamples) {
				samples.emplace_back(std::string(token.ptr, token.size));
			}
			/*check consistency*/
			if (numSamples < vectorSize) {
				fprintf(stderr,
						"ERROR The number of samples (%ld) read is less than vectorSize (%ld)\n",
						numSamples, vectorSize);
				return false;
			}
			etime = getSysTime();
			ROOT_PRINT("parse column headers %ld in %f sec\n", numSamples, get_duration_s(stime, etime));

			stime = getSysTime();

			/*get gene expression profiles*/
			/*extract gene expression values*/  // WAS READING TRANSPOSED.  NO LONGER.
			/* input is column major (row is 1 gene).  memory is row major (row is 1 sample) */
			FloatType * vec;
			ssize_t numGenes = 0;
			// get just the non-empty lines
			line = buffer.get_token<LF>();
			FloatType val;
			for (; (line.ptr != nullptr)  && (numGenes < numVectors);
				line = buffer.get_token<LF>(),
				++numGenes) {

				// parse the row name
				token = line.get_token_or_empty<COMMA>();
				genes.emplace_back(std::string(token.ptr, token.size));

				// parse the rest of data.
				vec = reinterpret_cast<FloatType*>(reinterpret_cast<unsigned char *>(vectors) + numGenes * stride_bytes);
				numSamples = 0;
				token = line.get_token_or_empty<COMMA>();		  
				for (; (token.ptr != nullptr) && (numSamples < vectorSize); 
					token = line.get_token_or_empty<COMMA>(), ++numSamples, ++vec) {

					if (token.size > 0) {
					 	val = atof(token.ptr); // will read until a non-numeric char is encountered.
						*vec = val;
					}
				}
				// NOTE: missing entries are treated as 0.
			}
			/*consistency check*/
			if (numGenes < numVectors) {
				fprintf(stderr,
						"ERROR The number of genes (%ld) read is less than numVectors (%ld)\n",
						numGenes, numVectors);
				return false;
			}
			etime = getSysTime();
			ROOT_PRINT("load values in %f sec\n", get_duration_s(stime, etime));

			return true;

		}


		/*get the matrix data*/
		bool loadMatrixData_impl_fast(std::vector<std::string>& genes,
				std::vector<std::string>& samples, FloatType* vectors, const ssize_t & numVectors, const ssize_t & vectorSize,
				const ssize_t & stride_bytes) {
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
			ssize_t numSamples = 0;
			token = line.get_token_or_empty<COMMA>();  // skip first one.  this is column name
			token = line.get_token_or_empty<COMMA>();  
			for (; (token.ptr != nullptr) && (numSamples < vectorSize); 
				token = line.get_token_or_empty<COMMA>(), ++numSamples) {
				samples.emplace_back(std::string(token.ptr, token.size));
			}
			/*check consistency*/
			if (numSamples < vectorSize) {
				fprintf(stderr,
						"ERROR The number of samples (%ld) read is less than vectorSize (%ld)\n",
						numSamples, vectorSize);
				return false;
			}
			etime = getSysTime();
			ROOT_PRINT("parse column headers %ld in %f sec\n", numSamples, get_duration_s(stime, etime));

			stime = getSysTime();

			/*get gene expression profiles*/
			/*extract gene expression values*/  // WAS READING TRANSPOSED.  NO LONGER.
			/* input is column major (row is 1 gene).  memory is row major (row is 1 sample) */
			FloatType * vec;
			ssize_t numGenes = 0;
			// get just the non-empty lines
			line = buffer.get_token<LF>();
			FloatType val;
			for (; (line.ptr != nullptr)  && (numGenes < numVectors);
				line = buffer.get_token<LF>(),
				++numGenes) {

				// parse the row name
				token = line.get_token_or_empty<COMMA>();
				genes.emplace_back(std::string(token.ptr, token.size));

				// parse the rest of data.
				vec = reinterpret_cast<FloatType*>(reinterpret_cast<unsigned char *>(vectors) + numGenes * stride_bytes);
				numSamples = 0;
				token = line.get_token_or_empty<COMMA>();		  
				for (; (token.ptr != nullptr) && (numSamples < vectorSize); 
					token = line.get_token_or_empty<COMMA>(), ++numSamples, ++vec) {

					if (token.size > 0) {
						val = splash::utils::atof(token.ptr); // will read until a non-numeric char is encountered.
						*vec = val;
					}
				}
				// NOTE: missing entries are treated as 0.
			}
			/*consistency check*/
			if (numGenes < numVectors) {
				fprintf(stderr,
						"ERROR The number of genes (%ld) read is less than numVectors (%ld)\n",
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
		bool getMatrixSize_impl(ssize_t& numVectors, ssize_t& vectorSize, MPI_Comm comm) {
				return getMatrixSize_impl(numVectors, vectorSize);
			}

		bool loadMatrixData_impl(std::vector<std::string>& genes,
				std::vector<std::string>& samples, FloatType* vectors, const ssize_t & numVectors, const ssize_t & vectorSize,
				const ssize_t & stride_bytes, MPI_Comm comm) {
			if (atof_type == 0)  // default
				return loadMatrixData_impl( genes, samples, vectors, numVectors, vectorSize, stride_bytes);
			else // fast and precise atof
				return loadMatrixData_impl_fast( genes, samples, vectors, numVectors, vectorSize, stride_bytes);
			}

		// bool loadMatrixData_impl(std::vector<std::string>& genes,
		// 		std::vector<std::string>& samples, splash::ds::aligned_matrix<FloatType> & input, MPI_Comm comm);
#endif
};




}}