/*
 *  VectorWriter.hpp
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



template<typename FloatType>
class VectorWriter {


public:

    static bool storeMatrixData(const char * fileName, size_t const & cols, 
        FloatType* vectors) {
            return VectorWriter::storeMatrixData(std::string(fileName), cols, vectors);
        }


    static bool storeMatrixData(std::string const & fileName, size_t const & cols, 
        FloatType* vectors) {
        
		std::vector<std::string> col_names(cols);
		for (size_t i = 0; i < cols; ++i) {
			col_names[i] = std::to_string(i);
		}
		return VectorWriter::storeMatrixData(fileName, 
			col_names,
			vectors);
    }
    
    static bool storeMatrixData(const char * fileName, 
			std::vector<std::string> const & col_names, FloatType* vectors) {
            return VectorWriter::storeMatrixData(std::string(fileName), col_names, vectors);
        }

	/*dump the matrix data.  matrix in row-major.  output is 1 line per row.*/
	static bool storeMatrixData(std::string const & fileName, 
			std::vector<std::string> const & col_names, FloatType* vectors) {

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
        for (int j = 0, jj = i * rowStride; 
            j < col_names.size(); ++j, ++jj) {
            ofs << delim << vectors[jj];
        }
        ofs << std::endl;


        ofs.close();
        auto etime = getSysTime();
        fprintf(stderr, "Wrote file %s in %f sec\n", fileName.c_str(), get_duration_s(stime, etime));

        return true;

    }
};


#endif

