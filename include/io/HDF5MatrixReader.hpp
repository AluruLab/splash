/*
 *  HDFMatrixReader.hpp
 *
 *  Created on: Aug 21, 2020
 *  Author: Tony Pan
 *  Affiliation: Institute for Data Engineering and Science
 *  						Georgia Institute of Technology, Atlanta, GA 30332
 *  
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

#include <string>  // string
#include <algorithm> 
#include <vector>
#include <string>
#include "ds/aligned_matrix.hpp"  // matrix
#include "ds/char_array.hpp"  // char_array_template

#include "utils/benchmark.hpp"
#include "utils/report.hpp"
#include "utils/mpi_types.hpp"



#ifdef USE_MPI
#include <mpi.h>
#endif  // with mpi

#include "hdf5.h"
#include "utils/hdf5_types.hpp"


namespace splash { namespace io { 


template<typename FloatType>
class HDF5MatrixReader {

	protected:
		std::string filename;

		bool getSize(hid_t file_id, std::string const & path, ssize_t & rows, ssize_t & cols) {
			// https://stackoverflow.com/questions/15786626/get-the-dimensions-of-a-hdf5-dataset#:~:text=First%20you%20need%20to%20get,int%20ndims%20%3D%20H5Sget_simple_extent_ndims(dspace)%3B
			FMT_PRINT_RT("get size\n");
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

		bool readStrings(hid_t file_id, std::string const & path, std::vector<std::string> & out ) {
			// from https://stackoverflow.com/questions/581209/how-to-best-write-out-a-stdvector-stdstring-container-to-a-hdf5-dataset
			// MODIFIED to use C api.

			// open data set
			auto exists = H5Lexists(file_id, path.c_str(), H5P_DEFAULT);
			if (exists <= 0) return false;

			hid_t dataset_id = H5Dopen(file_id, path.c_str(), H5P_DEFAULT);

			hid_t filetype_id = H5Dget_type(dataset_id);
			size_t max_len = H5Tget_size(filetype_id);

			// open data space and get dimensions
			hid_t dataspace_id = H5Dget_space(dataset_id);
			const int ndims = H5Sget_simple_extent_ndims(dataspace_id);
			hsize_t dims[ndims];
			H5Sget_simple_extent_dims(dataspace_id, dims, NULL);
			
			// get continuous space now...
			char * data = reinterpret_cast<char *>(calloc( dims[0], max_len * sizeof(char)));
			//FMT_ROOT_PRINT("In read STRING dataset, got number of strings: [{}].  temp array at {:p}\n", dims[0], data );

			// prepare output

			// Variable length string type
			// read data.  use mem_type is variable length string.  memspace is same as file space. 
			// NOTE: data should be a continuous memory block.  
			// NOTE: Not sure why online docs show as char** with automatic allocation assumption. API behavior change?
			//auto status = 
			H5Dread(dataset_id, filetype_id, H5S_ALL, dataspace_id, H5P_DEFAULT, data);

			// convert to string objects
			out.clear();
			out.reserve(dims[0]);
			char * ptr = data;
			for(size_t x=0; x < dims[0]; ++x, ptr += max_len)
			{
				// auto l = strlen(ptr);
				// FMT_ROOT_PRINT("GOT STRING {} {:p} {} \"{}\"\n", x, ptr, l, std::string(ptr, l) );
				out.emplace_back(ptr, strlen(ptr));
			}
			// H5Dvlen_reclaim (filetype_id, dataspace_id, H5P_DEFAULT, data);
			free(data);
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
			hsize_t mem_dims[ndims] = {rows, stride_bytes / sizeof(FloatType)};
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

			hsize_t mem_dims[ndims] = {rows, stride_bytes / sizeof(FloatType) };
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

		HDF5MatrixReader(std::string const & _filename) : filename(_filename) {}
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
                return false;
            }

			bool res = getSize(group_id, "block0_values", numVectors, vectorSize);


			H5Gclose(group_id);
			H5Fclose(file_id);

			auto etime = getSysTime();
			FMT_ROOT_PRINT("get matrix size in {} sec\n", get_duration_s(stime, etime));
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
				return false;
			}

			hid_t group_id;
            auto status = H5Lexists(file_id, path.c_str(), H5P_DEFAULT);
            if (status > 0) {
                group_id = H5Gopen(file_id, path.c_str(), H5P_DEFAULT);
            } else {
                FMT_PRINT_ERR("WARN: unable to get group {} in file {}\n", path, filename);
                return false;
            }

			readStrings(group_id, "axis1", genes);
			readStrings(group_id, "axis0", samples);
				
			auto etime = getSysTime();
			FMT_PRINT_RT("Read headers for {}/{} in {} sec\n", filename, path, get_duration_s(stime, etime));

			stime = getSysTime();

			// read the data.
			readValues(group_id, "block0_values", numVectors, vectorSize, vectors, stride_bytes);

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
                return false;
            }

			// read the names.
			readStrings(group_id, "axis1", genes);
			readStrings(group_id, "axis0", samples);

			// read the data.
			readValues(group_id, "block0_values", numVectors, vectorSize, vectors, stride_bytes);

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