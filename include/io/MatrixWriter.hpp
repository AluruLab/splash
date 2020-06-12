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

#include "benchmark.hpp"


template<typename FloatType>
class MatrixWriter {

public:

    static bool storeMatrixData(const char * fileName, size_t const & rows, size_t const & cols, 
        FloatType* vectors, int const & vectorSizeAligned) {
            return MatrixWriter::storeMatrixData(std::string(fileName), rows, cols, vectors, vectorSizeAligned);
        }


    static bool storeMatrixData(std::string const & fileName, size_t const & rows, size_t const & cols, 
        FloatType* vectors, int const & vectorSizeAligned) {
        
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
			vectors, vectorSizeAligned);
    }
    
    static bool storeMatrixData(const char * fileName, std::vector<std::string> const & row_names,
			std::vector<std::string> const & col_names, FloatType* vectors, 
            int const & vectorSizeAligned) {
            return MatrixWriter::storeMatrixData(std::string(fileName), row_names, col_names, vectors, vectorSizeAligned);
        }

	/*dump the matrix data.  matrix in row-major.  output is 1 line per row.*/
	static bool storeMatrixData(std::string const & fileName, std::vector<std::string> const & row_names,
			std::vector<std::string> const & col_names, FloatType* vectors, 
            int const & rowStride) {

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
        for (int i = 0; i < col_names.size(); ++i) {
            ofs << delim << col_names[i];
        }
        ofs << std::endl;

        ofs.precision(std::numeric_limits<FloatType>::max_digits10);

        // now write all data.
        for (int i = 0; i < row_names.size(); ++i) {
            ofs << row_names[i];

            for (int j = 0, jj = i * rowStride; 
                j < col_names.size(); ++j, ++jj) {
                ofs << delim << vectors[jj];
            }
            ofs << std::endl;
        }


        ofs.close();
        auto etime = getSysTime();
        fprintf(stderr, "Wrote file %s in %f sec\n", fileName.c_str(), get_duration_s(stime, etime));

        return true;

    }
};


#endif

