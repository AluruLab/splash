/*
 *  MatrixWriter.hpp
 *
 *  Created on: June 12, 2020
 *  Author: Tony C. Pan
 *  Affiliation: School of Computational Science & Engineering
 *  						Georgia Institute of Technology, Atlanta, GA 30332
 */

#pragma once


#include <iostream>
#include <fstream>

#include <vector>
#include <string>
#include <limits>

#include "utils/benchmark.hpp"
#include "ds/aligned_matrix.hpp"

namespace splash { namespace io { 

template<typename FloatType>
class MatrixWriter {

public:

    static bool storeMatrixData(const char * fileName, size_t const & rows, size_t const & cols, 
        FloatType* vectors, size_t const & stride_bytes) {
        return MatrixWriter::storeMatrixData(std::string(fileName), rows, cols, vectors, stride_bytes);
    }


    static bool storeMatrixData(std::string const & fileName, size_t const & rows, size_t const & cols, 
        FloatType* vectors, size_t const & stride_bytes) {
        
		std::vector<std::string> row_names(rows);
		for (size_t i = 0; i < rows; ++i) {
			row_names[i] = std::to_string(i);
		}
		std::vector<std::string> col_names(cols);
		for (size_t i = 0; i < cols; ++i) {
			col_names[i] = std::to_string(i);
		}
		return MatrixWriter::storeMatrixData(fileName, 
			row_names,
			col_names,
			vectors, stride_bytes);
    }

    static bool storeMatrixData(const char * fileName, splash::ds::aligned_matrix<FloatType> const & input) {
        return MatrixWriter::storeMatrixData(std::string(fileName), input);
    }
    static bool storeMatrixData(std::string const & fileName, splash::ds::aligned_matrix<FloatType> const & input) {
        
		std::vector<std::string> row_names(input.rows());
		for (size_t i = 0; i < input.rows(); ++i) {
			row_names[i] = std::to_string(i);
		}
		std::vector<std::string> col_names(input.columns());
		for (size_t i = 0; i < input.columns(); ++i) {
			col_names[i] = std::to_string(i);
		}
		return MatrixWriter::storeMatrixData(fileName, 
			row_names,
			col_names,
			input);
    }

    static bool storeMatrixData(const char * fileName, std::vector<std::string> const & row_names,
			std::vector<std::string> const & col_names, FloatType* vectors, 
            size_t const & stride_bytes) {
            return MatrixWriter::storeMatrixData(std::string(fileName), row_names, col_names, vectors, stride_bytes);
        }

	/*dump the matrix data.  matrix in row-major.  output is 1 line per row.*/
	static bool storeMatrixData(std::string const & fileName, std::vector<std::string> const & row_names,
			std::vector<std::string> const & col_names, FloatType* vectors, 
            size_t const & stride_bytes) {

        const char delim[] = ",";

        auto stime = getSysTime();

        // open file
        std::ofstream ofs;
        ofs.open(fileName);
        if (! ofs.is_open()) {
            fprintf(stderr, "Failed to open file %s\n", fileName.c_str());
            return false;
        }

        // write header (samples)
        // empty first entry.
        for (size_t i = 0; i < col_names.size(); ++i) {
            ofs << delim << col_names[i];
        }
        ofs << std::endl;

        ofs.precision(std::numeric_limits<FloatType>::max_digits10);

        // now write all data.
        FloatType* row;
        for (size_t i = 0; i < row_names.size(); ++i) {
            ofs << row_names[i];

            row = reinterpret_cast<FloatType*>(reinterpret_cast<unsigned char *>(vectors) + i * stride_bytes);
            for (size_t j = 0; j < col_names.size(); ++j) {
                ofs << delim << row[j];
            }
            ofs << std::endl;
        }


        ofs.close();
        auto etime = getSysTime();
        fprintf(stderr, "Wrote file %s in %f sec\n", fileName.c_str(), get_duration_s(stime, etime));

        return true;

    }

    static bool storeMatrixData(const char * fileName, std::vector<std::string> const & row_names,
			std::vector<std::string> const & col_names, splash::ds::aligned_matrix<FloatType> const & input) {
            return MatrixWriter::storeMatrixData(std::string(fileName), row_names, col_names, input);
        }

	/*dump the matrix data.  matrix in row-major.  output is 1 line per row.*/
	static bool storeMatrixData(std::string const & fileName, std::vector<std::string> const & row_names,
			std::vector<std::string> const & col_names, splash::ds::aligned_matrix<FloatType> const & input) {
        
        const char delim[] = ",";

        auto stime = getSysTime();

        // open file
        std::ofstream ofs;
        ofs.open(fileName);
        if (! ofs.is_open()) {
            fprintf(stderr, "Failed to open file %s\n", fileName.c_str());
            return false;
        }

        // write header (samples)
        // empty first entry.
        for (size_t i = 0; i < col_names.size(); ++i) {
            ofs << delim << col_names[i];
        }
        ofs << std::endl;

        ofs.precision(std::numeric_limits<FloatType>::max_digits10);

        // now write all data.
        auto row = input.data();
        for (size_t i = 0; i < row_names.size(); ++i) {
            ofs << row_names[i];

            row = input.data(i);
            for (size_t j = 0; j < col_names.size(); ++j) {
                ofs << delim << row[j];
            }
            ofs << std::endl;
        }


        ofs.close();
        auto etime = getSysTime();
        fprintf(stderr, "Wrote file %s in %f sec\n", fileName.c_str(), get_duration_s(stime, etime));

        return true;

    }

};



}}