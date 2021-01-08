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
#include <sstream> // stringstream

#include <vector>
#include <string>
#include <limits>

#include "utils/benchmark.hpp"
#include "utils/report.hpp"
#include "ds/aligned_matrix.hpp"
#include "utils/partition.hpp"

#define FMT_HEADER_ONLY
#include "fmt/format.h"

namespace splash { namespace io { 

template<typename FloatType>
class MatrixWriter {

public:

    static bool storeMatrixData(const char * fileName, size_t const & rows, size_t const & cols, 
        FloatType const * vectors, size_t const & stride_bytes) {
        return MatrixWriter::storeMatrixData(std::string(fileName), rows, cols, vectors, stride_bytes);
    }


    static bool storeMatrixData(std::string const & fileName, size_t const & rows, size_t const & cols, 
        FloatType const * vectors, size_t const & stride_bytes) {
        
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
        return MatrixWriter::storeMatrixData(std::string(fileName), input.rows(), input.columns(), input.data(), input.column_bytes());
    }
    static bool storeMatrixData(std::string const & fileName, splash::ds::aligned_matrix<FloatType> const & input) {        
        return MatrixWriter::storeMatrixData(fileName, input.rows(), input.columns(), input.data(), input.column_bytes());
    }

    static bool storeMatrixData(const char * fileName, std::vector<std::string> const & row_names,
			std::vector<std::string> const & col_names, FloatType const * vectors, 
            size_t const & stride_bytes) {
            return MatrixWriter::storeMatrixData(std::string(fileName), row_names, col_names, vectors, stride_bytes);
        }

	/*dump the matrix data.  matrix in row-major.  output is 1 line per row.*/
	static bool storeMatrixData(std::string const & fileName, std::vector<std::string> const & row_names,
			std::vector<std::string> const & col_names, FloatType const * vectors, 
            size_t const & stride_bytes) {

        const char delim[] = ",";

        auto stime = getSysTime();

        // open file

        fmt::memory_buffer data;
        {
            for (size_t i = 0; i < col_names.size(); ++i) {
                format_to(data, "{}{}", delim, col_names[i]);
            }
            format_to(data, "\n");

            // now convert all data.  again, data is distributed so iterate over local rows.
            FloatType const * row;
            for (size_t i = 0; i < row_names.size(); ++i) {
                format_to(data, "{}", row_names[i]);

                row = reinterpret_cast<FloatType const *>(reinterpret_cast<unsigned char const *>(vectors) + i * stride_bytes);
                for (size_t j = 0; j < col_names.size(); ++j) {
                    format_to(data, "{}{}", delim, row[j]);
                }
                format_to(data, "\n");
            }
        }

        std::ofstream ofs;
        ofs.open(fileName);
        if (! ofs.is_open()) {
            FMT_PRINT_RT("Failed to open file {}\n", fileName.c_str());
            return false;
        }
        ofs.write(data.data(), data.size());
        ofs.close();
        auto etime = getSysTime();
        FMT_PRINT_RT("Wrote file {} in {} sec\n", fileName.c_str(), get_duration_s(stime, etime));

        return true;

    }
	// static bool storeMatrixData(std::string const & fileName, std::vector<std::string> const & row_names,
	// 		std::vector<std::string> const & col_names, FloatType const * vectors, 
    //         size_t const & stride_bytes) {

    //     const char delim[] = ",";

    //     auto stime = getSysTime();

    //     // open file
    //     std::ofstream ofs;
    //     ofs.open(fileName);
    //     if (! ofs.is_open()) {
    //         FMT_PRINT_RT("Failed to open file {}\n", fileName.c_str());
    //         return false;
    //     }

    //     // write header (samples)
    //     // empty first entry.
    //     for (size_t i = 0; i < col_names.size(); ++i) {
    //         ofs << delim << col_names[i];
    //     }
    //     ofs << std::endl;

    //     ofs.precision(std::numeric_limits<FloatType>::max_digits10);

    //     // now write all data.
    //     FloatType const * row;
    //     for (size_t i = 0; i < row_names.size(); ++i) {
    //         ofs << row_names[i];

    //         row = reinterpret_cast<FloatType const *>(reinterpret_cast<unsigned char const *>(vectors) + i * stride_bytes);
    //         for (size_t j = 0; j < col_names.size(); ++j) {
    //             ofs << delim << row[j];
    //         }
    //         ofs << std::endl;
    //     }


    //     ofs.close();
    //     auto etime = getSysTime();
    //     FMT_PRINT_RT("Wrote file {} in {} sec\n", fileName.c_str(), get_duration_s(stime, etime));

    //     return true;

    // }

    static bool storeMatrixData(const char * fileName, std::vector<std::string> const & row_names,
			std::vector<std::string> const & col_names, splash::ds::aligned_matrix<FloatType> const & input) {
        return MatrixWriter::storeMatrixData(std::string(fileName), row_names, col_names, input.data(), input.column_bytes());
    }

	/*dump the matrix data.  matrix in row-major.  output is 1 line per row.*/
	static bool storeMatrixData(std::string const & fileName, std::vector<std::string> const & row_names,
			std::vector<std::string> const & col_names, splash::ds::aligned_matrix<FloatType> const & input) {
        
        return MatrixWriter::storeMatrixData(fileName, row_names, col_names, input.data(), input.column_bytes());
    }

#ifdef USE_MPI
    // assume input is distributed by row. 
	static bool storeMatrixDistributed(std::string const & fileName, std::vector<std::string> const & row_names,
			std::vector<std::string> const & col_names, splash::ds::aligned_matrix<FloatType> const & input, MPI_Comm const & comm) {
        int procs, rank;
        MPI_Comm_size(comm, &procs);
        MPI_Comm_rank(comm, &rank);

        // create a stringstream.
        const char delim[] = ",";

        // do some tests of sizes.
        // NOTE: does not support full size input but each process writing only a part of the input.
        if (row_names.size() == input.rows()) {
            // whole data present.
            if (procs == 1) {
                if (rank == 0)  // rank 0 only.
                    return MatrixWriter::storeMatrixData(fileName, row_names, col_names, input);
            } else {
                FMT_ROOT_PRINT("ERROR: unpartitioned data {} but multiple MPI procs {}.\n", input.rows(), procs);
                return false;
            }
        }
        

        // distributed. get get offset and count for the rows.
        size_t total = input.rows();
        MPI_Allreduce(MPI_IN_PLACE, &total, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);

        if (total != row_names.size()) {
            FMT_ROOT_PRINT("ERROR: input partitioning incorrect.  total {} != num of rows {}\n", total, row_names.size());
            return false;
        }

        auto stime = getSysTime();

        size_t offset = input.rows();
        MPI_Exscan(MPI_IN_PLACE, &offset, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);
        if (rank == 0) offset = 0;

        fmt::memory_buffer data;
        {
            if (rank == 0) {
                for (size_t i = 0; i < col_names.size(); ++i) {
                    format_to(data, "{}{}", delim, col_names[i]);
                }
                format_to(data, "\n");
            }
            
            // now convert all data.  again, data is distributed so iterate over local rows.
            FloatType const * row;
            for (size_t r = 0, n = offset; r < input.rows(); ++r, ++n) {
                format_to(data, "{}", row_names[n]);

                row = input.data(r);   // recall data is partitioned by row to all procs.
                for (size_t c = 0; c < input.columns(); ++c) {
                    format_to(data, "{}{}", delim, row[c]);
                }
                format_to(data, "\n");
            }
        }

        auto etime = getSysTime();
        FMT_ROOT_PRINT("converted data to string, {} chars in {} sec\n", data.size(), get_duration_s(stime, etime));

        // now get ready to write out.

        stime = getSysTime();

        size_t lsize = data.size();
        offset = 0;
        // do some counts.
        MPI_Exscan(&lsize, &offset, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);
        if (rank == 0) offset = 0;

        // FMT_PRINT_RT("MPI_IO writing {} bytes, starting at {}\n", lsize, offset);

        size_t mx = 0;
        MPI_Allreduce(&lsize, &mx, 1, MPI_UNSIGNED_LONG, MPI_MAX, comm);

        // now use MPI-IO to write out.
        MPI_File fh;
        MPI_File_open(comm, fileName.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
        // truncate the file
        MPI_File_set_size (fh, 0);
        MPI_Offset base;
        MPI_File_get_position(fh, &base);
        MPI_Status status;

        // write out, 1 billion at a time because of MPI int limit.
        const size_t step = 1 << 30;
        size_t loffset = 0;
        size_t cstep;
        for (size_t s = 0; s < mx; s += step) {  // s may be replaceable with loffset, but it probably would mean overlapping regions. safer to keep them separate..
            cstep = std::min(step, lsize);
            lsize -= cstep;
            MPI_File_write_at_all(fh, base + offset, data.data() + loffset, static_cast<int>(cstep), MPI_CHAR, &status);
            offset += cstep;
            loffset += cstep;
        }

        MPI_File_close(&fh);
        // FMT_PRINT_RT("MPI_IO remaining {} bytes, finishing at {}\n", lsize, offset);
        MPI_Bcast(&offset, 1, MPI_UNSIGNED_LONG, procs - 1, comm);

        etime = getSysTime();
        FMT_ROOT_PRINT("MPI-IO wrote {} bytes in {} sec\n", offset, get_duration_s(stime, etime));

        return true;



    }

    // // assume input is distributed by row. 
	// static bool storeMatrixDistributed(std::string const & fileName, std::vector<std::string> const & row_names,
	// 		std::vector<std::string> const & col_names, splash::ds::aligned_matrix<FloatType> const & input, MPI_Comm const & comm) {
    //     int procs, rank;
    //     MPI_Comm_size(comm, &procs);
    //     MPI_Comm_rank(comm, &rank);

    //     // create a stringstream.
    //     const char delim[] = ",";

    //     // do some tests of sizes.
    //     // NOTE: does not support full size input but each process writing only a part of the input.
    //     if (row_names.size() == input.rows()) {
    //         // whole data present.
    //         if (procs == 1) {
    //             if (rank == 0)  // rank 0 only.
    //                 return MatrixWriter::storeMatrixData(fileName, row_names, col_names, input);
    //         } else {
    //             FMT_ROOT_PRINT("ERROR: unpartitioned data {} but multiple MPI procs {}.\n", input.rows(), procs);
    //             return false;
    //         }
    //     }
        

    //     // distributed. get get offset and count for the rows.
    //     size_t total = input.rows();
    //     MPI_Allreduce(MPI_IN_PLACE, &total, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);

    //     if (total != row_names.size()) {
    //         FMT_ROOT_PRINT("ERROR: input partitioning incorrect.  total {} != num of rows {}\n", total, row_names.size());
    //         return false;
    //     }

    //     auto stime = getSysTime();

    //     size_t offset = input.rows();
    //     MPI_Exscan(MPI_IN_PLACE, &offset, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);
    //     if (rank == 0) offset = 0;

    //     std::string data;
    //     {
    //         std::stringstream ss;
    //         // only rank 0 writes column names.
    //         if (rank == 0) {
    //             for (size_t i = 0; i < col_names.size(); ++i) {
    //                 ss << delim << col_names[i];
    //             }
    //             ss << std::endl;
    //         }

    //         ss.precision(std::numeric_limits<FloatType>::max_digits10);

    //         // now convert all data.  again, data is distributed so iterate over local rows.
    //         FloatType const * row;
    //         for (size_t r = 0, n = offset; r < input.rows(); ++r, ++n) {
    //             ss << row_names[n];

    //             row = input.data(r);   // recall data is partitioned by row to all procs.
    //             for (size_t c = 0; c < input.columns(); ++c) {
    //                 ss << delim << row[c];
    //             }
    //             ss << std::endl;
    //         }
    //         data = ss.str();
    //     }

    //     auto etime = getSysTime();
    //     FMT_ROOT_PRINT("converted data to string, {} chars in {} sec\n", data.length(), get_duration_s(stime, etime));

    //     // now get ready to write out.

    //     stime = getSysTime();

    //     size_t lsize = data.length();
    //     offset = 0;
    //     // do some counts.
    //     MPI_Exscan(&lsize, &offset, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);
    //     if (rank == 0) offset = 0;

    //     // FMT_PRINT_RT("MPI_IO writing {} bytes, starting at {}\n", lsize, offset);

    //     size_t mx = 0;
    //     MPI_Allreduce(&lsize, &mx, 1, MPI_UNSIGNED_LONG, MPI_MAX, comm);

    //     // now use MPI-IO to write out.
    //     MPI_File fh;
    //     MPI_File_open(comm, fileName.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
    //     // truncate the file
    //     MPI_File_set_size (fh, 0);
    //     MPI_Offset base;
    //     MPI_File_get_position(fh, &base);
    //     MPI_Status status;

    //     // write out, 1 billion at a time because of MPI int limit.
    //     const size_t step = 1 << 30;
    //     size_t loffset = 0;
    //     size_t cstep;
    //     for (size_t s = 0; s < mx; s += step) {  // s may be replaceable with loffset, but it probably would mean overlapping regions. safer to keep them separate..
    //         cstep = std::min(step, lsize);
    //         lsize -= cstep;
    //         MPI_File_write_at_all(fh, base + offset, data.data() + loffset, static_cast<int>(cstep), MPI_CHAR, &status);
    //         offset += cstep;
    //         loffset += cstep;
    //     }

    //     MPI_File_close(&fh);
    //     // FMT_PRINT_RT("MPI_IO remaining {} bytes, finishing at {}\n", lsize, offset);
    //     MPI_Bcast(&offset, 1, MPI_UNSIGNED_LONG, procs - 1, comm);

    //     etime = getSysTime();
    //     FMT_ROOT_PRINT("MPI-IO wrote {} bytes in {} sec\n", offset, get_duration_s(stime, etime));

    //     return true;



    // }    
#endif

};



}}