/*
 *  HDF5MatrixWriter.hpp
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
#include <bitset>  // for python pandas "transposed" attribute.
#include <algorithm>  // for ::std::fill.

#include "utils/benchmark.hpp"
#include "utils/report.hpp"
#include "ds/aligned_matrix.hpp"
#include "utils/partition.hpp"

#ifdef USE_MPI
#include <mpi.h>
#endif

#include <hdf5.h>
#include "utils/hdf5_types.hpp"

namespace splash { namespace io { 

template<typename FloatType>
class HDF5MatrixWriter {
protected:
    std::string filename;

    //***** attributes:
        /* for / group:  
         * CLASS: UTF8 String GROUP
         * PYTABLES_FORMAT_VERSION: 2.1
         * TITLE: 
         * VERSION: 1.0
         * 
         */
        /* for array group:  
         * CLASS: UTF8 String GROUP
         * TITLE: 
         * VERSION: 1.0
         * axis0_variety: regular
         * axis1_variety: regular
         * block0_items_variety: regular
         * encoding: UTF-8
         * errors:strict
         * nblocks: 1
         * ndim: 2
         * pandas_type: frame
         * pandas_version: 0.15.2
         */
        /* for dataset:  
         * CLASS: ARRAY
         * FLAVOR: numpy
         * TITLE: 
         * VERSION: 2.4
         * kind: string
         * name: N. (axis0, block0_items)  genes (axis1)
         * transposed: 0x01
         */
    // single threaded.
    template <typename T>
    void writeAttribute(hid_t obj_id, std::string const & name, T const & value) {
        splash::utils::hdf5::datatype<T> type;
        hid_t type_id = type.value;

        hid_t space_id = H5Screate(H5S_SCALAR);

        hid_t attr_id;
        auto exists = H5Aexists(obj_id, name.c_str());
        if (exists) {
            // open
            attr_id = H5Aopen(obj_id, name.c_str(), H5P_DEFAULT);
        } else {
            // create
            attr_id = H5Acreate(obj_id, name.c_str(), type_id, space_id, H5P_DEFAULT, H5P_DEFAULT);
        }
        H5Awrite(attr_id, type_id, &value);

        H5Sclose(space_id);
        H5Aclose(attr_id);
    }

    void writeAttribute(hid_t obj_id, std::string const & name, const char* value, bool const & ascii = false) {
        writeAttribute(obj_id, name, std::string(value), ascii);
    }

    void writeAttribute(hid_t obj_id, std::string const & name, std::string const & value, bool const & ascii = false) {
        bool empty = value.length() == 0;
        // type
        hid_t type_id = H5Tcopy (H5T_C_S1);
        if (!ascii) H5Tset_cset(type_id, H5T_CSET_UTF8);  // for compatibility with Python Pandas.
        size_t len = empty ? 1 : value.length();
        H5Tset_size(type_id, len);

        hid_t space_id = empty ? H5Screate(H5S_NULL) : H5Screate(H5S_SCALAR);

        hid_t attr_id;
        auto exists = H5Aexists(obj_id, name.c_str());
        if (exists) {
            // open
            attr_id = H5Aopen(obj_id, name.c_str(), H5P_DEFAULT);
        } else {
            // create
            attr_id = H5Acreate(obj_id, name.c_str(), type_id, space_id, H5P_DEFAULT, H5P_DEFAULT);
        }
        if (!empty)
            H5Awrite(attr_id, type_id, value.c_str());

        H5Sclose(space_id);
        H5Aclose(attr_id);
        H5Tclose(type_id);
    }

    void _writeGroupAttributes(hid_t group_id) {
        writeAttribute(group_id, "CLASS", "GROUP");
        writeAttribute(group_id, "TITLE", "");
        writeAttribute(group_id, "VERSION", "1.0");
    }
    void writeRootGroupAttributes(hid_t root_id) {
        _writeGroupAttributes(root_id);
        writeAttribute(root_id, "PYTABLES_FORMAT_VERSION", "2.1");
    }
    void writeDataGroupAttributes(hid_t group_id, int64_t const & nblocks, int64_t const & ndim) {
        _writeGroupAttributes(group_id);
        writeAttribute(group_id, "axis0_variety", "regular");
        writeAttribute(group_id, "axis1_variety", "regular");
        writeAttribute(group_id, "block0_items_variety", "regular");
        writeAttribute(group_id, "encoding", "UTF-8"); 
        writeAttribute(group_id, "errors", "strict");
        writeAttribute(group_id, "nblocks", nblocks);
        writeAttribute(group_id, "ndim", ndim);
        writeAttribute(group_id, "pandas_type", "frame");
        writeAttribute(group_id, "pandas_version", "0.15.2");
    }
    void writeDatasetAttributes(hid_t dataset_id) {
        writeAttribute(dataset_id, "CLASS", "ARRAY");
        writeAttribute(dataset_id, "FLAVOR", "numpy");
        writeAttribute(dataset_id, "TITLE", "");
        writeAttribute(dataset_id, "VERSION", "2.4");
        ::std::bitset<8> transposed(1);
        writeAttribute(dataset_id, "transposed", transposed);
    }
    void writeAxisAttributes(hid_t dataset_id, std::string const & kind, std::string const & name, bool const & ascii_name = true) {
        writeDatasetAttributes(dataset_id);
        writeAttribute(dataset_id, "kind", kind);
        writeAttribute(dataset_id, "name", name, ascii_name); 
    }

    void writeStrings(hid_t file_id, std::string const & path, std::string const & meaning, std::vector<std::string> const & names,
        bool const & ascii_meaning = false) {
        // from https://stackoverflow.com/questions/581209/how-to-best-write-out-a-stdvector-stdstring-container-to-a-hdf5-dataset
        // modified to use C api.

        // HDF5 only understands vector of char* :-(
        size_t max_len = 0;
        for (size_t ii = 0; ii < names.size(); ++ii)
        {
            max_len = std::max(max_len, names[ii].length());
        }
        ++max_len;
        // copy data into continuous block.
        char * data = reinterpret_cast<char *>(calloc(names.size(), max_len * sizeof(char)));
        char * ptr = data;
        for (size_t ii = 0; ii < names.size(); ++ii, ptr += max_len)
        {
            memcpy(ptr, names[ii].c_str(), names[ii].length());  // length is <= maxlen
        }

        //
        //  one dimension
        // 
        hsize_t str_dimsf[1] = { names.size() };
        hid_t dataspace_id = H5Screate_simple(1, str_dimsf, NULL);

        // Variable length string
        hid_t type_id = H5Tcopy (H5T_C_S1);
        // H5Tset_cset(type_id, H5T_CSET_UTF8);  // Axis names are in ASCII in pandas H5.
        H5Tset_size(type_id, max_len);
        hid_t dataset_id = H5Dcreate(file_id, path.c_str(), type_id, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (dataset_id < 0) {
            // failed.  print error and return
            FMT_PRINT_RT("ERROR: unable to create dataset {}.\n", path);
            return;
        } else {
            FMT_PRINT_RT("Created dataset {}.\n", path);
        }

        // out data needs to be a continuous block!!!!
        H5Dwrite(dataset_id, type_id, dataspace_id, H5S_ALL, H5P_DEFAULT, data);
        writeAxisAttributes(dataset_id, "string", meaning, ascii_meaning);

        // H5Dflush(dataset_id);  // here does not cause "HDF5: infinite loop closing library" with MPI-IO
        H5Dclose(dataset_id);
        H5Sclose(dataspace_id);
        H5Tclose(type_id);
        free(data);

    }

    // may specify cols to be fewer than stride_bytes can support....
    void writeValues(hid_t file_id, std::string const & path, size_t const & rows, size_t const & cols, 
        FloatType const * vectors, size_t const & stride_bytes) {

        if ((stride_bytes % sizeof(FloatType)) > 0) {
            // unsupported.  This means having to write row by row, and some procs may have to write 0 bytes - complicating PHDF5 write.
            FMT_PRINT_ERR("ERROR: column stride not a multiple of element data type.  This is not support and will be deprecated.\n");
            return;
        }

        splash::utils::hdf5::datatype<FloatType> h5type;
        hid_t type_id = h5type.value;

        hsize_t filespace_dim[2] = { rows, cols };  // what the file will contain.
        hid_t filespace_id = H5Screate_simple(2, filespace_dim, NULL);

        // try opening the group first.  
        // NOTE: cannot delete dataset or replace dataset with smaller one.
        // check if opened.  if not, create it.
        hid_t dataset_id = H5Dcreate(file_id, path.c_str(), type_id, filespace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (dataset_id < 0) {
            // failed.  print error and return
            FMT_PRINT_RT("ERROR: unable to create dataset.\n");
            return;
        }

        // what the memory contains.  assume continuous.
        hsize_t memspace_dim[2] = { rows, stride_bytes / sizeof(FloatType) };
        hid_t memspace_id = H5Screate_simple(2, memspace_dim, NULL);
        // select hyperslab of memory, for row by row traversal
        hsize_t mstart[2] = {0, 0};  // element offset for first block
        hsize_t mcount[2] = {rows, 1}; // # of blocks
        hsize_t mstride[2] = {1, stride_bytes / sizeof(FloatType)};  // element stride to get to next block
        hsize_t mblock[2] = {1, cols};  // block size  1xcols
        H5Sselect_hyperslab(memspace_id, H5S_SELECT_SET, mstart, mstride, mcount, mblock);
        
        // may need to use hyperslab....
        H5Dwrite(dataset_id, type_id, memspace_id, filespace_id, H5P_DEFAULT, vectors);
        writeDatasetAttributes(dataset_id);

        // H5Dflush(dataset_id);  // here does not cause "HDF5: infinite loop closing library" with MPI-IO
        H5Sclose(memspace_id);
        H5Sclose(filespace_id);
        H5Dclose(dataset_id);

    }

#ifdef USE_MPI
    void writeValues(hid_t file_id, std::string const & path, size_t const & rows, size_t const & cols, 
        FloatType const * vectors, size_t const & stride_bytes, MPI_Comm const & comm) {

        if ((stride_bytes % sizeof(FloatType)) > 0) {
            // unsupported.  This means having to write row by row, and some procs may have to write 0 bytes - complicating PHDF5 write.
            FMT_PRINT_ERR("ERROR: column stride not a multiple of element data type.  This is not support and will be deprecated.\n");
            return;
        }

        int procs, rank;
        MPI_Comm_size(comm, &procs);
        MPI_Comm_rank(comm, &rank);


        hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

        // get total and offsets.
        // get offsets.
        size_t row_offset = rows;
        MPI_Exscan(MPI_IN_PLACE, &row_offset, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);
        if (rank == 0) row_offset = 0;
        size_t row_total = rows;
        MPI_Allreduce(MPI_IN_PLACE, &row_total, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);
        size_t cols_max = cols;
        MPI_Allreduce(MPI_IN_PLACE, &cols_max, 1, MPI_UNSIGNED_LONG, MPI_MAX, comm);

        // get data type
        splash::utils::hdf5::datatype<FloatType> h5type;
        hid_t type_id = h5type.value;

        // create file space
        hsize_t filespace_dim[2] = { row_total, cols_max };
        hid_t filespace_id = H5Screate_simple(2, filespace_dim, NULL);

        // try opening the group first.  
        // NOTE: cannot delete dataset or replace dataset with smaller one.
        // check if opened.  if not, create it.
        hid_t dataset_id = H5Dcreate(file_id, path.c_str(), type_id, filespace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (dataset_id < 0) {
            // failed.  print error and return
            FMT_PRINT_RT("ERROR: unable to create dataset.\n");
            return;
        }

        // each process defines dataset in memory and hyperslab in file.
        hsize_t start[2] = {row_offset, 0};  // starting offset, row, then col.
        hsize_t count[2] = {rows, cols};   // number of row and col blocks.

        H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET, start, NULL, count, NULL);

        hsize_t memspace_dim[2] = {rows, stride_bytes / sizeof(FloatType)};
        hid_t memspace_id = H5Screate_simple(2, memspace_dim, NULL);
        // select hyperslab of memory, for row by row traversal
        hsize_t mstart[2] = {0, 0};  // element offset for first block
        hsize_t mcount[2] = {rows, 1}; // # of blocks
        hsize_t mstride[2] = {1, stride_bytes / sizeof(FloatType)};  // element stride to get to next block
        hsize_t mblock[2] = {1, cols};  // block size  1xcols
        H5Sselect_hyperslab(memspace_id, H5S_SELECT_SET, mstart, mstride, mcount, mblock);
        

        H5Dwrite(dataset_id, type_id, memspace_id, filespace_id, plist_id, vectors);
        writeDatasetAttributes(dataset_id);  // ALL PROCS WRITE?

        // H5Dflush(dataset_id);  // KNOWN to cause "HDF5: infinite loop closing library" with MPI-IO
        H5Sclose(memspace_id);
        H5Sclose(filespace_id);
        H5Dclose(dataset_id);
        H5Pclose(plist_id);

    }
#endif

public:

    HDF5MatrixWriter(std::string const & _filename) : filename(_filename) {}

    virtual ~HDF5MatrixWriter() {}


    bool storeMatrixData(std::string const & path, size_t const & rows, size_t const & cols, 
        FloatType const * vectors, size_t const & stride_bytes) {
        
		std::vector<std::string> row_names(rows);
		for (size_t i = 0; i < rows; ++i) {
			row_names[i] = std::to_string(i);
		}
		std::vector<std::string> col_names(cols);
		for (size_t i = 0; i < cols; ++i) {
			col_names[i] = std::to_string(i);
		}
		return HDF5MatrixWriter::storeMatrixData(path, 
			row_names,
			col_names,
			vectors, stride_bytes);
    }


    bool storeMatrixData(const char * path, size_t const & rows, size_t const & cols, 
        FloatType const * vectors, size_t const & stride_bytes) {
        return HDF5MatrixWriter::storeMatrixData(std::string(path), rows, cols, vectors, stride_bytes);
    }

    bool storeMatrixData(const char * path, splash::ds::aligned_matrix<FloatType> const & input) {
        return HDF5MatrixWriter::storeMatrixData(std::string(path), input.rows(), input.columns(), input.data(), input.column_bytes());
    }
    bool storeMatrixData(std::string const & path, splash::ds::aligned_matrix<FloatType> const & input) {        
        return HDF5MatrixWriter::storeMatrixData(path, input.rows(), input.columns(), input.data(), input.column_bytes());
    }

    bool storeMatrixData(std::string const & path, std::vector<std::string> const & row_names,
			std::vector<std::string> const & col_names, FloatType const * vectors, 
            size_t const & stride_bytes) {
        auto stime = getSysTime();
        hid_t file_id;

        // create, truncate if exists.
        file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (file_id < 0) {
            FMT_PRINT_ERR("ERROR: failed to open PHDF5 file {}\n", filename);
            return false;
        }
        writeRootGroupAttributes(file_id);

        // ------------- create the group
        hid_t group_id;
        auto exists = H5Lexists(file_id, path.c_str(), H5P_DEFAULT);
        if (exists > 0) {
            group_id = H5Gopen(file_id, path.c_str(), H5P_DEFAULT);
        } else if (exists == 0) {
            group_id = H5Gcreate(file_id, path.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        } else {
            FMT_PRINT_ERR("WARN: unable to get group {} in file {}\n", path, filename);
            return false;
        }
        writeDataGroupAttributes(group_id, 1, 2);

        // ------------- write the names.
        writeStrings(group_id, "axis1", "genes", row_names);  // row names are in axis 1?
        writeStrings(group_id, "axis0", "N.", col_names, true);  // col names are in axis 0?
        writeStrings(group_id, "block0_items", "N.", col_names, true);  // col names are in axis 0?

        auto etime = getSysTime();
        FMT_PRINT_RT("Wrote names in file {}{} in {} sec\n", filename, path, get_duration_s(stime, etime));

        stime = getSysTime();

        // ---------- write data.
        writeValues(group_id, "block0_values", row_names.size(), col_names.size(), vectors, stride_bytes);

        // H5Gflush(group_id);  // here does not cause "HDF5: infinite loop closing library" with MPI-IO
        H5Gclose(group_id);  // close the group immediately
        H5Fflush(file_id, H5F_SCOPE_GLOBAL);  //must flush else could get problem opening next.
        H5Fclose(file_id);

        etime = getSysTime();
        FMT_PRINT_RT("Wrote values in file {}{} in {} sec\n", filename, path, get_duration_s(stime, etime));

        return true;
    }
    

	/*dump the matrix data.  matrix in row-major.  output is 1 line per row.*/
	bool storeMatrixData(const char * path, std::vector<std::string> const & row_names,
			std::vector<std::string> const & col_names, FloatType const * vectors, 
            size_t const & stride_bytes) {
        return HDF5MatrixWriter::storeMatrixData(std::string(path), row_names, col_names, vectors, stride_bytes);

    }

    bool storeMatrixData(const char * path, std::vector<std::string> const & row_names,
			std::vector<std::string> const & col_names, splash::ds::aligned_matrix<FloatType> const & input) {
        return HDF5MatrixWriter::storeMatrixData(std::string(path), row_names, col_names, input.data(), input.column_bytes());
    }

	/*dump the matrix data.  matrix in row-major.  output is 1 line per row.*/
	bool storeMatrixData(std::string const & path, std::vector<std::string> const & row_names,
			std::vector<std::string> const & col_names, splash::ds::aligned_matrix<FloatType> const & input) {
        
        return HDF5MatrixWriter::storeMatrixData(path, row_names, col_names, input.data(), input.column_bytes());
    }

#ifdef USE_MPI
	bool storeMatrixDistributed(std::string const & path, std::vector<std::string> const & row_names,
			std::vector<std::string> const & col_names, splash::ds::aligned_matrix<FloatType> const & input, MPI_Comm const & comm) {
        int procs, rank;
        MPI_Comm_size(comm, &procs);
        MPI_Comm_rank(comm, &rank);

        if (procs == 1) { // rank 0 only.
            return HDF5MatrixWriter::storeMatrixData(path, row_names, col_names, input);
        }
        FMT_PRINT_ERR("NOTICE: parallel write using HDF5 OpenMPI versions in Ubuntu 18.04.5 LTS does not work.\n");

        // ------- do some tests of sizes.
        // NOTE: does not support full size input but each process writing only a part of the input.
        hid_t file_id;
        herr_t status;

        // file and group creation.
        auto stime = getSysTime();
        hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_fapl_mpio(plist_id, comm, MPI_INFO_NULL);

        // clears file.
        file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
        if (file_id < 0) {
            FMT_PRINT_ERR("ERROR: failed to open PHDF5 file {}\n", filename);
            H5Pclose(plist_id);
            return false;
        }
        writeRootGroupAttributes(file_id);

        // create group
        hid_t group_id = H5Gcreate(file_id, path.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (group_id < 0) {
            FMT_PRINT_ERR("ERROR: failed to open PHDF5 group {}/{}\n", filename, path);
            H5Pclose(plist_id);
            status = H5Fclose(file_id);
            if (status < 0) FMT_PRINT_ERR("ERROR: file is not closed\n");
            return false;
        } else {
            writeDataGroupAttributes(group_id, 1, 2);
            // H5Gflush(group_id); // KNOWN to cause "HDF5: infinite loop closing library" with MPI-IO
            H5Gclose(group_id);
            H5Fflush(file_id, H5F_SCOPE_GLOBAL);
            status = H5Fclose(file_id);
            if (status < 0) FMT_PRINT_ERR("ERROR: file is not closed\n");
        }

        auto etime = getSysTime();
        FMT_ROOT_PRINT("create group in {} sec\n", get_duration_s(stime, etime));

        stime = getSysTime();

        // do some checks.
        // distributed. get get offset and count for the rows.
        size_t total = input.rows();
        MPI_Allreduce(MPI_IN_PLACE, &total, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);
        if (total != row_names.size()) {
            FMT_ROOT_PRINT("ERROR: input partitioning incorrect.  total {} != num of rows {}\n", total, row_names.size());
            H5Pclose(plist_id);
            return false;
        }
        etime = getSysTime();
        FMT_ROOT_PRINT("Total rows {} in {} sec\n", total, get_duration_s(stime, etime));

        // first write the column and row names,
        stime = getSysTime();
        if (rank == 0) {
        
            // check if file exists.  if yes, open,  if no, create
            file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
            if (file_id < 0) {
                FMT_PRINT_ERR("ERROR: failed to open PHDF5 file {}\n", filename);
            }

            hid_t group_id;
            auto exists = H5Lexists(file_id, path.c_str(), H5P_DEFAULT);
            if (exists > 0) {
                group_id = H5Gopen(file_id, path.c_str(), H5P_DEFAULT);
                // ------------- write the names.
                writeStrings(group_id, "axis1", "genes", row_names);  // row names are in axis 1?
                writeStrings(group_id, "axis0", "N.", col_names);  // col names are in axis 0?
                writeStrings(group_id, "block0_items", "N.", col_names);  // col names are in axis 0?

                // H5Gflush(group_id);  // here does not cause "HDF5: infinite loop closing library" with MPI-IO
                H5Gclose(group_id);  // close the group immediately
                H5Fflush(file_id, H5F_SCOPE_GLOBAL);  //must flush else could get problem opening next.
            } else {
                FMT_PRINT_ERR("WARN: unable to get group {} in file {}\n", path, filename);
                H5Pclose(plist_id);

                return false;
            }

            status = H5Fclose(file_id);
            if (status < 0) FMT_PRINT_ERR("ERROR: file is not closed\n");
        }
        etime = getSysTime();
        FMT_ROOT_PRINT("Wrote headers for {}/{} in {} sec\n", filename, path, get_duration_s(stime, etime));

        MPI_Barrier(comm);
        //============ now write the data in parallel.

        stime = getSysTime();

        if ((file_id = H5Fopen(filename.c_str(), H5F_ACC_RDWR, plist_id)) < 0) {
            FMT_PRINT_ERR("ERROR: failed to open PHDF5 file to write values {}\n", filename);
            H5Pclose(plist_id);
            return false;
        }

        auto exists = H5Lexists(file_id, path.c_str(), H5P_DEFAULT);
        if (exists <= 0) {
            FMT_PRINT_ERR("ERROR: failed to get PHDF5 file {} group {} to write values\n", filename, path);
            H5Pclose(plist_id);
            status = H5Fclose(file_id);
            if (status < 0) FMT_PRINT_ERR("ERROR: file is not closed\n");
            return false;
        }
        group_id = H5Gopen(file_id, path.c_str(), H5P_DEFAULT);
        if (group_id < 0) {
            FMT_PRINT_ERR("ERROR: failed to open PHDF5 group {}/{}\n", filename, path);
            H5Pclose(plist_id);
            status = H5Fclose(file_id);
            if (status < 0) FMT_PRINT_ERR("ERROR: file is not closed 3\n");

            return false;
        }

        writeValues(group_id, "block0_values", 
            input.rows(), input.columns(), input.data(), input.column_bytes(), comm);

        // H5Gflush(group_id);  // KNOWN to cause "HDF5: infinite loop closing library" with MPI-IO
        H5Fflush(file_id, H5F_SCOPE_GLOBAL);  //must flush else could get problem opening next.

        H5Gclose(group_id);
        H5Pclose(plist_id);
        status = H5Fclose(file_id);
        if (status < 0) FMT_PRINT_ERR("ERROR: file is not closed 3\n");

        etime = getSysTime();
        FMT_ROOT_PRINT("PHDF5 MPI-IO wrote {}/{} in {} sec\n", filename, path, get_duration_s(stime, etime));

        return true;
        
    }

    // assume input is distributed by row. 
	bool storeMatrixDistributed(char const *path, std::vector<std::string> const & row_names,
			std::vector<std::string> const & col_names, splash::ds::aligned_matrix<FloatType> const & input, MPI_Comm const & comm) {
        return storeMatrixDistributed(std::string(path), row_names, col_names, input, comm);

    }
#endif

};



}}