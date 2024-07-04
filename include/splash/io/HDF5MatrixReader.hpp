/*
 * Copyright 2021 Georgia Tech Research Corporation
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

/*  
 * serial and parallel hdf5 output.
 * following group structure is used:
 * 	root: no datasets
 *  top level children: contains source expression data as "dataset", and any other data (e.g. gene names, sample names, regulatory factors, etc) 
 * 		labeled with dataset names . e.g. /flower, /leaf, etc.
 *  derived data appears as children immediately below its source data.
 * 		e.g. /flower/mi, /leaf/pearson, /flower/mi/dpi
 *  aggregated results appear under deepest level of its aggregate sources.  naming should be enforced programatically.
 * 		e.g. /flower/mi/dpi/dpi+mi.  
 * 
 * Need tool to ingest/extract datasets
 */

#pragma once

#include <ostream>
#include <string>  // string
#include <algorithm> 
#include <vector>
#include <string>
#include <iostream>
#include "splash/ds/aligned_matrix.hpp"  // matrix
#include "splash/ds/char_array.hpp"  // char_array_template

#include "splash/utils/benchmark.hpp"
#include "splash/utils/report.hpp"
#include "splash/utils/mpi_types.hpp"



#ifdef USE_MPI
#include <mpi.h>
#endif  // with mpi

#include <hdf5.h>
#include "splash/utils/hdf5_types.hpp"


namespace splash { namespace io { 


template<typename FloatType>
class HDF5MatrixReader {

	protected:
		std::string filename;
		std::string gene_data_path;
		std::string samples_data_path;
		std::string matrix_data_path;

		bool getSize(hid_t file_id, std::string const & path, ssize_t & rows, ssize_t & cols) {
			// https://stackoverflow.com/questions/15786626/get-the-dimensions-of-a-hdf5-dataset#:~:text=First%20you%20need%20to%20get,int%20ndims%20%3D%20H5Sget_simple_extent_ndims(dspace)%3B
			auto exists = H5Lexists(file_id, path.c_str(), H5P_DEFAULT);
			if (exists <= 0) return false;

			hid_t dataset_id = H5Dopen(file_id, path.c_str(), H5P_DEFAULT);

			hid_t dataspace_id = H5Dget_space(dataset_id);
			const int ndims = H5Sget_simple_extent_ndims(dataspace_id);
			hsize_t dims[ndims];
			H5Sget_simple_extent_dims(dataspace_id, dims, NULL);

			rows = dims[0]; // rows
			cols = dims[1]; // cols.

			H5Sclose(dataspace_id);
			H5Dclose(dataset_id);

			return true;
		}

		// max length is set.
		inline size_t strnlen(char * str, size_t n) {
			size_t i = 0;
			while ((i < n) && (str[i] != 0)) ++i;
			return i;
		}

		bool readStrings(hid_t file_id, std::string const & path, std::vector<std::string> & out ) {
			// from https://stackoverflow.com/questions/581209/how-to-best-write-out-a-stdvector-stdstring-container-to-a-hdf5-dataset
			// MODIFIED to use C api.

			// open data set
			auto exists = H5Lexists(file_id, path.c_str(), H5P_DEFAULT);
			if (exists <= 0) return false;

			hid_t dataset_id = H5Dopen(file_id, path.c_str(), H5P_DEFAULT);

			hid_t filetype_id = H5Dget_type(dataset_id);
            if(H5Tdetect_class(filetype_id, H5T_STRING) <= 0){
                FMT_ROOT_PRINT("ERROR: NOT a string type dataset {}.\n", path);
                H5Dclose(dataset_id);
                H5Tclose(filetype_id);
                return false;
            }


			size_t max_len = H5Tget_size(filetype_id);
            // auto cid =  H5Tget_class (filetype_id);
			// FMT_ROOT_PRINT("In read dataset, cid [{}];  ptr size {}\n",
            //               cid == H5T_STRING ? "string" : "non-string",  max_len );
			// open data space and get dimensions
			hid_t dataspace_id = H5Dget_space(dataset_id);
			const int ndims = H5Sget_simple_extent_ndims(dataspace_id);
			hsize_t dims[ndims];
			H5Sget_simple_extent_dims(dataspace_id, dims, NULL);
			
			// FMT_ROOT_PRINT("In read STRING dataset, got number of strings: [{}] \n", dims[0]);
            if(H5Tis_variable_str(filetype_id)) {
			    // Variable length string type -- max_len is the ptr size 
                // data is array of pointers : char **
                // https://docs.hdfgroup.org/hdf5/v1_14/group___h5_t.html#title25
			    char **data = reinterpret_cast<char **>(calloc(dims[0] ,
                                                        max_len * sizeof(char)));

			    // prepare output
			    //auto status = 
			    H5Dread(dataset_id, filetype_id, H5S_ALL, dataspace_id, H5P_DEFAULT, data);

			    // convert to string objects
			    out.clear();
			    out.reserve(dims[0]);
			    char * ptr = *data;
                for(size_t x=0; x < dims[0]; ++x, ptr = *(data + x)) {
			    	auto l = strlen(ptr);
			    	// FMT_ROOT_PRINT("GOT STRING {} {:p} {} \"{}\"\n", x, ptr, l, std::string(ptr, l) );
			    	out.emplace_back(ptr, l);
			    }
                // TODO(x): vlen reclaim is supposed to be deprecated
                //   libraries in current server doesn't reflect this
                // https://docs.hdfgroup.org/hdf5/v1_12/group___h5_d.html#title31
			    H5Dvlen_reclaim (filetype_id, dataspace_id, H5P_DEFAULT, data);
			    free(data);
            } else {
      			// Fixed length strings :  max_len is the size of the string;
                // data should be a continuous memory block.  
                char* data = reinterpret_cast<char*>(
                    calloc((dims[0] + 1), max_len * sizeof(char)));
                data[dims[0]*max_len] = 0;
                out.clear();
                out.reserve(dims[0]);

                // read the block of data
                H5Dread(dataset_id, filetype_id, H5S_ALL, dataspace_id,
                        H5P_DEFAULT, data);
                //
                char * ptr = data;
                for(size_t x=0; x < dims[0]; ++x, ptr += max_len) {
                    // auto l = strlen(ptr);
                    // FMT_ROOT_PRINT("GOT STRING {} {:p} {} \"{}\"\n", x, ptr, l, std::string(ptr, l) );
                    out.emplace_back(ptr, strnlen(ptr, max_len));
                }
                free(data);
            }
            // std::cout << " OUT " << out.size() << " " << dims[0] << std::endl;

			H5Dclose(dataset_id);
			H5Sclose(dataspace_id);
			H5Tclose(filetype_id);
			
			return true;
		}

		// specify number of rows and cols to read.
		bool readValues(hid_t file_id, std::string const & path,
			size_t const & rows, size_t const & cols, 
			FloatType * vectors, size_t const & stride_bytes) {

			if ((stride_bytes % sizeof(FloatType)) > 0) {
				// unsupported.  This means having to write row by row, and some procs may have to write 0 bytes - complicating PHDF5 write.
            	FMT_PRINT_ERR("ERROR: column stride not a multiple of element data type.  This is not support and will be deprecated.\n");
            	return false;
			}
			
			// open data set
			auto exists = H5Lexists(file_id, path.c_str(), H5P_DEFAULT);
			if (exists <= 0) return false;

			hid_t dataset_id = H5Dopen(file_id, path.c_str(), H5P_DEFAULT);

			// open data space and get dimensions
			hid_t filespace_id = H5Dget_space(dataset_id);
			const int ndims = H5Sget_simple_extent_ndims(filespace_id);
			hsize_t file_dims[ndims];
			H5Sget_simple_extent_dims(filespace_id, file_dims, NULL);

			// create target space
			hsize_t mem_dims[2] = {rows, stride_bytes / sizeof(FloatType)};
			hid_t memspace_id = H5Screate_simple(ndims, mem_dims, NULL);
			// select hyperslab of memory, for row by row traversal
			hsize_t mstart[2] = {0, 0};  // element offset for first block
			hsize_t mcount[2] = {rows, 1}; // # of blocks
			hsize_t mstride[2] = {1, stride_bytes / sizeof(FloatType)};  // element stride to get to next block
			hsize_t mblock[2] = {1, cols};  // block size  1xcols
			H5Sselect_hyperslab(memspace_id, H5S_SELECT_SET, mstart, mstride, mcount, mblock);

			// float type
			splash::utils::hdf5::datatype<FloatType> type;
			hid_t type_id = type.value;

			// read data.  use mem_type is variable length string.  memspace is same as file space. 
			//auto status = 
			H5Dread(dataset_id, type_id, memspace_id, filespace_id, H5P_DEFAULT, vectors);

			H5Sclose(memspace_id);
			H5Sclose(filespace_id);
			H5Dclose(dataset_id);

			return true;
		}

#ifdef USE_MPI
		bool readValues(hid_t file_id, std::string const & path,
			size_t const & rows, size_t const & cols, 
			FloatType * vectors, size_t const & stride_bytes,
			MPI_Comm const & comm) {

			if ((stride_bytes % sizeof(FloatType)) > 0) {
				// unsupported.  This means having to write row by row, and some procs may have to write 0 bytes - complicating PHDF5 write.
            	FMT_PRINT_ERR("ERROR: column stride not a multiple of element data type.  This is not support and will be deprecated.\n");
            	return false;
			}

			int procs, rank;
			MPI_Comm_size(comm, &procs);
			MPI_Comm_rank(comm, &rank);
			
			size_t row_offset = rows;
			MPI_Exscan(MPI_IN_PLACE, &row_offset, 1, MPI_UNSIGNED_LONG, MPI_SUM, comm);
			if (rank == 0) row_offset = 0;

			// open data set
			auto exists = H5Lexists(file_id, path.c_str(), H5P_DEFAULT);
			if (exists <= 0) return false;

			hid_t dataset_id = H5Dopen(file_id, path.c_str(), H5P_DEFAULT);

			// open data space and get dimensions
			hid_t filespace_id = H5Dget_space(dataset_id);
			const int ndims = H5Sget_simple_extent_ndims(filespace_id);
			hsize_t file_dims[ndims];
			H5Sget_simple_extent_dims(filespace_id, file_dims, NULL);
			
			// each process defines dataset in memory and hyperslab in file.
			hsize_t start[2] = {row_offset, 0};  // starting offset, row, then col.
			hsize_t count[2] = {rows, cols};   // number of row and col blocks.
			H5Sselect_hyperslab(filespace_id, H5S_SELECT_SET, start, NULL, count, NULL);

			hsize_t mem_dims[2] = {rows, stride_bytes / sizeof(FloatType) };
			hid_t memspace_id = H5Screate_simple(ndims, mem_dims, NULL);
			// select hyperslab of memory, for row by row traversal
			hsize_t mstart[2] = {0, 0};  // element offset for first block
			hsize_t mcount[2] = {rows, 1}; // # of blocks
			hsize_t mstride[2] = {1, stride_bytes / sizeof(FloatType)};  // element stride to get to next block
			hsize_t mblock[2] = {1, cols};  // block size  1xcols
			H5Sselect_hyperslab(memspace_id, H5S_SELECT_SET, mstart, mstride, mcount, mblock);

			// float type
			splash::utils::hdf5::datatype<FloatType> type;
			hid_t type_id = type.value;

			// read data.  use mem_type is variable length string.  memspace is same as file space. 
			hid_t plist_id = H5Pcreate(H5P_DATASET_XFER);
			H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);
			//auto status = 
			H5Dread(dataset_id, type_id, memspace_id, filespace_id, plist_id, vectors);

			// convert to string objects
			H5Sclose(memspace_id);
			H5Sclose(filespace_id);
			H5Dclose(dataset_id);

			H5Pclose(plist_id);

			return true;
		}
#endif


	public:

		HDF5MatrixReader(std::string const & _filename,
                         std::string const & _gdpath="axis1",
                         std::string const & _sdpath="axis0",
                         std::string const & _mtxpath="block0_values") : 
            filename(_filename), gene_data_path(_gdpath),
            samples_data_path(_sdpath), matrix_data_path(_mtxpath) {}
		virtual ~HDF5MatrixReader() {}

			// // open the file for reading only.
			// fid = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);


#ifdef USE_MPI
		bool getMatrixSize(std::string const & path, ssize_t& numVectors, ssize_t& vectorSize,
			MPI_Comm comm = MPI_COMM_WORLD) {
			int rank;
			int procs;
			MPI_Comm_rank(comm, &rank);
			MPI_Comm_size(comm, &procs);

			if (procs == 1) {
				return getMatrixSize_impl(path, numVectors, vectorSize);
			}

			bool res;
			if (rank == 0) {
				res = getMatrixSize_impl(path, numVectors, vectorSize);
			}
			splash::utils::mpi::datatype<ssize_t> size_type;
			MPI_Bcast(&numVectors, 1, size_type.value, 0, comm);
			MPI_Bcast(&vectorSize, 1, size_type.value, 0, comm);
			
			return res;
		}
#else 
		/*get gene expression matrix size*/
		bool getMatrixSize(std::string const & path, ssize_t& numVectors, ssize_t& vectorSize) {
			return getMatrixSize_impl(path, numVectors, vectorSize);
		}
#endif 

	protected:
		/*get gene expression matrix size*/
		bool getMatrixSize_impl(std::string const & path, 
			ssize_t& numVectors, ssize_t& vectorSize) {
			auto stime = getSysTime();

			// open the file for reading only.
			hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
			if (file_id < 0) return false;
			
			hid_t group_id;
            auto status = H5Lexists(file_id, path.c_str(), H5P_DEFAULT);
            if (status > 0) {
                group_id = H5Gopen(file_id, path.c_str(), H5P_DEFAULT);
            } else {
                FMT_PRINT_ERR("WARN: unable to get group {} in file {}\n", path, filename);
				H5Fclose(file_id);
                return false;
            }

			bool res = getSize(group_id, matrix_data_path, numVectors, vectorSize);


			H5Gclose(group_id);
			H5Fclose(file_id);

			auto etime = getSysTime();
			FMT_ROOT_PRINT("get matrix size {}x{} in {} sec\n", numVectors, vectorSize, get_duration_s(stime, etime));
			return res;
		}

	public:

#ifdef USE_MPI
		bool loadMatrixData(std::string const & path, std::vector<std::string>& genes,
				std::vector<std::string>& samples, FloatType* vectors, const ssize_t numVectors, const ssize_t vectorSize,
				const ssize_t stride_bytes, MPI_Comm comm) {
			// open the file for reading only.
			int procs, rank;
			MPI_Comm_size(comm, &procs);
			MPI_Comm_rank(comm, &rank);

			if (procs == 1) { // rank 0 only.
            	return HDF5MatrixReader::loadMatrixData(path, genes, samples, vectors, numVectors, vectorSize, stride_bytes);
        	}

			// read the names.
			auto stime = getSysTime();
			MPI_Info info = MPI_INFO_NULL;
			hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
			H5Pset_fapl_mpio(plist_id, comm, info);

			hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, plist_id);
			if (file_id < 0) {
				FMT_PRINT_ERR("ERROR: failed to open PHDF5 file {}\n", filename);
				H5Pclose(plist_id);
				return false;
			}

			hid_t group_id;
            auto status = H5Lexists(file_id, path.c_str(), H5P_DEFAULT);
            if (status > 0) {
                group_id = H5Gopen(file_id, path.c_str(), H5P_DEFAULT);
            } else {
                FMT_PRINT_ERR("WARN: unable to get group {} in file {}\n", path, filename);
				H5Fclose(file_id);
				H5Pclose(plist_id);
                return false;
            }

			readStrings(group_id, gene_data_path, genes);
			readStrings(group_id, samples_data_path, samples);
				
			auto etime = getSysTime();
			FMT_PRINT_RT("Read headers for {}/{} in {} sec\n", filename, path, get_duration_s(stime, etime));

			stime = getSysTime();

			// read the data.
			readValues(group_id, matrix_data_path, numVectors, vectorSize, vectors, stride_bytes);

			H5Gclose(group_id);
			H5Fclose(file_id);
	        H5Pclose(plist_id);

			etime = getSysTime();
			FMT_PRINT_RT("Read headers for {}{} in {} sec\n", filename, path, get_duration_s(stime, etime));

			return true;
		}
		bool loadMatrixData(std::string const & path, std::vector<std::string>& genes,
				std::vector<std::string>& samples, splash::ds::aligned_matrix<FloatType> & output, 
				MPI_Comm comm) {
			return loadMatrixData(path, genes, samples, 
				output.data(), output.rows(), output.columns(), output.column_bytes(), 
				comm);
		}
#endif
		bool loadMatrixData(std::string const & path, std::vector<std::string>& genes,
				std::vector<std::string>& samples, FloatType* vectors, const ssize_t numVectors, const ssize_t vectorSize,
				const ssize_t stride_bytes) {

			// open the file for reading only.
			hid_t file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
			if (file_id < 0) return false;

			hid_t group_id;
            auto status = H5Lexists(file_id, path.c_str(), H5P_DEFAULT);
            if (status > 0) {
                group_id = H5Gopen(file_id, path.c_str(), H5P_DEFAULT);
            } else {
                FMT_PRINT_ERR("WARN: unable to get group {} in file {}\n", path, filename);
				H5Fclose(file_id);
                return false;
            }

			// read the names.
			readStrings(group_id, gene_data_path, genes);
			readStrings(group_id, samples_data_path, samples);

			// read the data.
			readValues(group_id, matrix_data_path, numVectors, vectorSize, vectors, stride_bytes);

			H5Gclose(group_id);
			H5Fclose(file_id);
			return true;
		}
		bool loadMatrixData(std::string const & path, std::vector<std::string>& genes,
				std::vector<std::string>& samples, splash::ds::aligned_matrix<FloatType> & output) {
			return loadMatrixData(path, genes, samples, 
				output.data(), output.rows(), output.columns(), output.column_bytes());
		}



};




}}
