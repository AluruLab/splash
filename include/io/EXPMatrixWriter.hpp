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

#include "benchmark.hpp"

#ifndef EXP_SKIP_TWO_ROWS
#define EXP_SKIP_TWO_ROWS false
#endif

namespace splash { namespace io { 

template<typename FloatType>
class EXPMatrixWriter {

public:

    static bool storeMatrixData(const char * fileName, size_t const & rows, size_t const & cols, 
        FloatType* vectors, int const & vectorSizeAligned, bool const & skip = EXP_SKIP_TWO_ROWS) {
            return EXPMatrixWriter::storeMatrixData(std::string(fileName), rows, cols, vectors, vectorSizeAligned, skip);
        }


    static bool storeMatrixData(std::string const & fileName, size_t const & rows, size_t const & cols,
        FloatType* vectors, int const & vectorSizeAligned, bool const & skip = EXP_SKIP_TWO_ROWS) {
        
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
			vectors, vectorSizeAligned, skip);
    }
    
    static bool storeMatrixData(const char * fileName, std::vector<std::string> const & genes,
			std::vector<std::string> const & samples, FloatType* vectors, 
            int const & vectorSizeAligned, bool const & skip = EXP_SKIP_TWO_ROWS) {
            return EXPMatrixWriter::storeMatrixData(std::string(fileName), genes, samples, vectors, vectorSizeAligned, skip);
        }

	/*dump the matrix data.  matrix has 1 sample per row.  output has to be 1 gene per row (transposed).*/
	static bool storeMatrixData(std::string const & fileName, std::vector<std::string> const & genes,
            std::vector<std::string> const & samples, FloatType* vectors, 
            int const & vectorSizeAligned, bool const & skip = EXP_SKIP_TWO_ROWS) {

        const char delim[] = "\t";

        auto stime = getSysTime();

        fprintf(stderr, "EXPMatrixWriter writing r %lu x c %lu\n", genes.size(), samples.size() );
        // open file
        std::ofstream ofs;
        ofs.open(fileName);
        if (! ofs.is_open()) {
            fprintf(stderr, "Failed to open file %s\n", fileName.c_str());
            return false;
        }

        // write header (samples)
        // header has ID and ALIAS
        ofs << "Id" << delim << "Alias";
        for (int i = 0; i < samples.size(); ++i) {
            ofs << delim << samples[i];
        }
        ofs << std::endl;

        // if skip, insert 2 empty lines
        if (skip) {
            ofs << "empty" << std::endl << "empty" << std::endl;
        }
        
        ofs.precision(std::numeric_limits<FloatType>::max_digits10);

        // now write all data. 
        for (int i = 0; i < genes.size(); ++i) {

            ofs << genes[i] << delim << "---";

            for (int j = 0, jj = i * vectorSizeAligned; 
                j < samples.size(); ++j, ++jj) {
                ofs << delim << vectors[jj];
            }
            // this writes in transposed way.  not using.
            // for (int j = 0, jj = i; j < samples.size(); ++j, jj += vectorSizeAligned) {
            //     ofs << delim << vectors[jj];
            // }
            ofs << std::endl;
        }


        ofs.close();

        auto etime = getSysTime();
        fprintf(stderr, "Wrote file %s in %f sec\n", fileName.c_str(), get_duration_s(stime, etime));

        return true;

    }
};



}}