/*
 *  EXPMatrixWriter.hpp
 *
 *  Created on: June 12, 2020
 *  Author: Tony C. Pan
 *  Affiliation: School of Computational Science & Engineering
 *  						Georgia Institute of Technology, Atlanta, GA 30332
 */

#pragma once

#include "utils/benchmark.hpp"
#include <iostream>
#include <fstream>

#include <vector>
#include <string>
#include <limits>

#include "utils/benchmark.hpp"
#include "ds/aligned_matrix.hpp"
#include "utils/report.hpp"

#ifndef EXP_SKIP_TWO_ROWS
#define EXP_SKIP_TWO_ROWS false
#endif

namespace splash { namespace io { 

template<typename FloatType>
class EXPMatrixWriter {

public:

    static bool storeMatrixData(const char * fileName, size_t const & rows, size_t const & cols, 
        FloatType* vectors, size_t const & stride_bytes, bool const & skip = EXP_SKIP_TWO_ROWS) {
            return EXPMatrixWriter::storeMatrixData(std::string(fileName), rows, cols, vectors, stride_bytes, skip);
        }


    static bool storeMatrixData(std::string const & fileName, size_t const & rows, size_t const & cols,
        FloatType* vectors, size_t const & stride_bytes, bool const & skip = EXP_SKIP_TWO_ROWS) {
        
		std::vector<std::string> row_names(rows);
		for (size_t i = 0; i < rows; ++i) {
			row_names[i] = std::to_string(i);
		}
		std::vector<std::string> col_names(cols);
		for (size_t i = 0; i < cols; ++i) {
			col_names[i] = std::to_string(i);
		}
		return EXPMatrixWriter::storeMatrixData(fileName, 
			row_names,
			col_names,
			vectors, stride_bytes, skip);
    }

    static bool storeMatrixData(const char * fileName, 
        splash::ds::aligned_matrix<FloatType> const & input, bool const & skip = EXP_SKIP_TWO_ROWS) {
        return EXPMatrixWriter::storeMatrixData(std::string(fileName), input.rows(), input.columns(), input.data(), input.column_bytes(), skip);
    }
    static bool storeMatrixData(std::string const & fileName, 
        splash::ds::aligned_matrix<FloatType> const & input, bool const & skip = EXP_SKIP_TWO_ROWS) {
        return EXPMatrixWriter::storeMatrixData(fileName, input.rows(), input.columns(), input.data(), input.column_bytes(), skip);
    }
    
    static bool storeMatrixData(const char * fileName, std::vector<std::string> const & genes,
			std::vector<std::string> const & samples, FloatType* vectors, 
            size_t const & stride_bytes, bool const & skip = EXP_SKIP_TWO_ROWS) {
            return EXPMatrixWriter::storeMatrixData(std::string(fileName), genes, samples, vectors, stride_bytes, skip);
        }

	/*dump the matrix data.  matrix has 1 sample per row.  output has to be 1 gene per row (transposed).*/
	static bool storeMatrixData(std::string const & fileName, std::vector<std::string> const & genes,
            std::vector<std::string> const & samples, FloatType* vectors, 
            size_t const & stride_bytes, bool const & skip = EXP_SKIP_TWO_ROWS) {

        const char delim[] = "\t";

        auto stime = getSysTime();

        FMT_ROOT_PRINT("EXPMatrixWriter writing r {} x c {}\n", genes.size(), samples.size() );

        fmt::memory_buffer data;
        {
            format_to(data, "Id{}Alias", delim);
            for (size_t i = 0; i < samples.size(); ++i) {
                format_to(data, "{}{}", delim, samples[i]);
            }
            format_to(data, "\n");


            // if skip, insert 2 empty lines
            if (skip) {
                format_to(data, "empty\nempty\n");
            }

            // now convert all data.  again, data is distributed so iterate over local rows.
            FloatType const * row;
            for (size_t i = 0; i < genes.size(); ++i) {
                format_to(data, "{}{}---", genes[i], delim);

                row = reinterpret_cast<FloatType *>(reinterpret_cast<unsigned char *>(vectors) + i * stride_bytes);
                for (size_t j = 0; j < samples.size(); ++j) {
                    format_to(data, "{}{}", delim, row[j]);
                }
                format_to(data, "\n");
            }
        }

        // open file
        std::ofstream ofs;
        ofs.open(fileName);
        if (! ofs.is_open()) {
            FMT_PRINT_RT("ERROR: Failed to open file {}\n", fileName.c_str());
            return false;
        }
        
        ofs.write(data.data(), data.size());

        ofs.close();

        auto etime = getSysTime();
        FMT_ROOT_PRINT("Wrote file {} in {} sec\n", fileName.c_str(), get_duration_s(stime, etime));

        return true;

    }

    static bool storeMatrixData(const char * fileName, std::vector<std::string> const & genes,
			std::vector<std::string> const & samples, splash::ds::aligned_matrix<FloatType> const & input,
            bool const & skip = EXP_SKIP_TWO_ROWS) {
            return EXPMatrixWriter::storeMatrixData(std::string(fileName), genes, samples, input.data(), input.column_bytes(), skip);
        }

	/*dump the matrix data.  matrix has 1 sample per row.  output has to be 1 gene per row (transposed).*/
	static bool storeMatrixData(std::string const & fileName, std::vector<std::string> const & genes,
            std::vector<std::string> const & samples, splash::ds::aligned_matrix<FloatType> const & input,
            bool const & skip = EXP_SKIP_TWO_ROWS) {
        return EXPMatrixWriter::storeMatrixData(fileName, genes, samples, input.data(), input.column_bytes(), skip);
    }

};



}}