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

#include <iostream>
#include <fstream>

#include <vector>
#include <string>
#include <limits>

#include "splash/utils/benchmark.hpp"
#include "splash/ds/aligned_matrix.hpp"
#include "splash/utils/report.hpp"

#ifndef EXP_SKIP_TWO_ROWS
#define EXP_SKIP_TWO_ROWS false
#endif

namespace splash { namespace io { 

template<typename FloatType>
class EXPMatrixWriter {

public:

    static bool storeMatrixData(const char * fileName, size_t const & rows, size_t const & cols, 
        FloatType const * vectors, size_t const & stride_bytes, bool const & skip = EXP_SKIP_TWO_ROWS) {
            return EXPMatrixWriter::storeMatrixData(std::string(fileName), rows, cols, vectors, stride_bytes, skip);
        }


    static bool storeMatrixData(std::string const & fileName, size_t const & rows, size_t const & cols,
        FloatType const * vectors, size_t const & stride_bytes, bool const & skip = EXP_SKIP_TWO_ROWS) {
        
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
			std::vector<std::string> const & samples, FloatType const * vectors, 
            size_t const & stride_bytes, bool const & skip = EXP_SKIP_TWO_ROWS) {
            return EXPMatrixWriter::storeMatrixData(std::string(fileName), genes, samples, vectors, stride_bytes, skip);
        }

	/*dump the matrix data.  matrix has 1 sample per row.  output has to be 1 gene per row (transposed).*/
	static bool storeMatrixData(std::string const & fileName, std::vector<std::string> const & genes,
            std::vector<std::string> const & samples, FloatType const * vectors, 
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

                row = reinterpret_cast<FloatType const *>(reinterpret_cast<unsigned char const *>(vectors) + i * stride_bytes);
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