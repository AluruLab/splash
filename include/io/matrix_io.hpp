#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include "kernel/random.hpp"
#include "ds/aligned_matrix.hpp"
#include "utils/report.hpp"
#include "utils/partition.hpp"
#include "utils/string_utils.hpp"  // endsWith
#include "io/EXPMatrixReader.hpp"
#include "io/EXPMatrixReader2.hpp"
#include "io/CSVMatrixReader.hpp"
#include "io/CSVMatrixReader2.hpp"
#include "io/MatrixWriter.hpp"
#include "io/EXPMatrixWriter.hpp"
#include "io/HDF5MatrixReader.hpp"
#include "io/HDF5MatrixWriter.hpp"

#ifdef USE_MPI
#include <mpi.h>
#endif

template <typename T, typename S>
splash::ds::aligned_matrix<T> make_random_matrix(
	long const & seed, T const & rmin, T const & rmax,
	S const & rows, S const & cols,
	std::vector<std::string> & row_names, std::vector<std::string> & col_names
) {
	// allocate input.
	splash::ds::aligned_matrix<T> input(rows, cols);

	// random generate data.   output: every proc has full data.
	// OMP compatible, rows per thread.  MPI compatible, rows per proc.
	splash::kernel::random_number_generator<> generators(seed);
	splash::kernel::UniformRandomMatrixGenerator<T> mat_gen(generators, rmin, rmax);

	// split by proc
	splash::utils::partitioner1D<PARTITION_EQUAL> partitioner;

	int procs = 1;
	int rank = 0;

#ifdef USE_MPI
	MPI_Comm_size(MPI_COMM_WORLD, &procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif 

	splash::utils::partition<S> part = partitioner.get_partition(rows, procs, rank);
	part.print("random: ");

	mat_gen(input, part);
	input.allgather_inplace(part);

	row_names.resize(rows);
	for (size_t i = 0; i < rows; ++i) {
		row_names[i] = std::to_string(i);
	}
	col_names.resize(cols);
	for (size_t i = 0; i < cols; ++i) {
		col_names[i] = std::to_string(i);
	}

	return input;
};

template <typename T, typename S>
splash::ds::aligned_matrix<T> make_random_matrix_distributed(
	long const & seed, T const & rmin, T const & rmax,
	S const & rows, S const & cols,
	std::vector<std::string> & row_names, std::vector<std::string> & col_names
) {
	// random generate data.   output: every proc has full data.
	// OMP compatible, rows per thread.  MPI compatible, rows per proc.
	splash::kernel::random_number_generator<> generators(seed);
	splash::kernel::UniformRandomMatrixGenerator<T> mat_gen(generators, rmin, rmax);

	// split by proc
	splash::utils::partitioner1D<PARTITION_EQUAL> partitioner;

	int procs = 1;
	int rank = 0;

#ifdef USE_MPI
	MPI_Comm_size(MPI_COMM_WORLD, &procs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif 

	splash::utils::partition<S> part = partitioner.get_partition(rows, procs, rank);
	part.print("random: ");

	// allocate input.
	splash::ds::aligned_matrix<T> input(part.size, cols);

	mat_gen(input);

	row_names.resize(rows);
	for (size_t i = 0; i < rows; ++i) {
		row_names[i] = std::to_string(i);
	}
	col_names.resize(cols);
	for (size_t i = 0; i < cols; ++i) {
		col_names[i] = std::to_string(i);
	}

	return input;
};

template <typename T, typename S>
splash::ds::aligned_matrix<T> read_exp_matrix(std::string const & filename, S & rows, S & cols,
	std::vector<std::string> & genes, std::vector<std::string> & samples, bool skip = false) {
	ssize_t nvecs;
	ssize_t vecsize;
	
	// read file to get size (HAVE TO DO 2 PASS to get size.)
	// MPI compatible, not OpenMP enabled.
	splash::io::EXPMatrixReader<T>::getMatrixSize(filename, nvecs, vecsize, skip);

	// get the minimum array size.
	rows = (rows > 0) ? std::min(static_cast<S>(nvecs), rows) : nvecs;
	cols = (cols > 0) ? std::min(static_cast<S>(vecsize), cols) : vecsize;

	// allocate the data
	splash::ds::aligned_matrix<T> input(rows, cols);

	// now read the data.  // MPI compatible, not OpenMP enabled.
	splash::io::EXPMatrixReader<T>::loadMatrixData(filename, genes, samples, input, skip);

	return input;
};


template <typename T, typename S>
splash::ds::aligned_matrix<T> read_exp_matrix_fast(std::string const & filename, S & rows, S & cols,
	std::vector<std::string> & genes, std::vector<std::string> & samples,
	int const & atof_type = 1, bool skip = false
) {
	ssize_t nvecs = std::numeric_limits<ssize_t>::max();
	ssize_t vecsize = std::numeric_limits<ssize_t>::max();
	
	// read file to get size (HAVE TO DO 2 PASS to get size.)
	// MPI compatible, not OpenMP enabled.
	splash::io::EXPMatrixReader2<T> reader(filename.c_str(), atof_type);
	reader.getMatrixSize(nvecs, vecsize, skip);

	// get the minimum array size.
	rows = (rows > 0) ? std::min(static_cast<S>(nvecs), rows) : nvecs;
	cols = (cols > 0) ? std::min(static_cast<S>(vecsize), cols) : vecsize;

	// allocate the data
	splash::ds::aligned_matrix<T> input(rows, cols);

	// now read the data.  // MPI compatible, not OpenMP enabled.
	reader.loadMatrixData(genes, samples, input, skip);

	return input;
};

// template <typename T, typename S>
// splash::ds::aligned_matrix<T> read_exp_matrix_distributed(std::string const & filename, S & rows, S & cols,
// 	std::vector<std::string> & genes, std::vector<std::string> & samples,
// 	int const & atof_type = 1, bool skip = false
// ) {
// 	ssize_t nvecs = std::numeric_limits<ssize_t>::max();
// 	ssize_t vecsize = std::numeric_limits<ssize_t>::max();
	
// 	// read file to get size (HAVE TO DO 2 PASS to get size.)
// 	// MPI compatible, not OpenMP enabled.
// 	splash::io::EXPMatrixReader2<T> reader(filename.c_str(), atof_type);
// 	reader.getMatrixSize(nvecs, vecsize, skip);

// 	// get the minimum array size.
// 	rows = (rows > 0) ? std::min(static_cast<S>(nvecs), rows) : nvecs;
// 	cols = (cols > 0) ? std::min(static_cast<S>(vecsize), cols) : vecsize;

// #ifdef USE_MPI
// 	MPI_Comm_size(MPI_COMM_WORLD, &procs);
// 	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
// #endif 

// 	splash::utils::partition<S> part = partitioner.get_partition(rows, procs, rank);


// 	// allocate the data
// 	splash::ds::aligned_matrix<T> input(part.size, cols);

// 	// now read the data.  // MPI compatible, not OpenMP enabled.
// 	reader.loadMatrixData(genes, samples, input, skip);

// 	return input;
// };

template <typename T>
void write_exp_matrix(std::string const & filename, std::vector<std::string> & row_names, std::vector<std::string> & col_names,
	splash::ds::aligned_matrix<T> const & data) {
		// NOTE: rank 0 writes out.
			// write to file.  MPI enabled.  Not thread enabled.
#ifdef USE_MPI
		int rank = 0;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		auto allout = data.gather(0);
		FMT_ROOT_PRINT("writing file size: {}, {}\n", allout.rows(), allout.columns());
		if (rank == 0) {
			splash::io::EXPMatrixWriter<T>::storeMatrixData(filename, row_names, col_names, allout);
		}
			// std::stringstream ss;
			// ss << filename << "." << rank << ".exp";
			// std::string fn = ss.str();
			// splash::io::EXPMatrixWriter<T>::storeMatrixData(fn, row_names, col_names, data);
#else
		splash::io::EXPMatrixWriter<T>::storeMatrixData(filename, row_names, col_names, data);
#endif

};


template <typename T, typename S>
splash::ds::aligned_matrix<T> read_csv_matrix(std::string const & filename, S & rows, S & cols,
	std::vector<std::string> & genes, std::vector<std::string> & samples,
	int const & atof_type = 1
) {
	ssize_t nvecs = std::numeric_limits<ssize_t>::max();
	ssize_t vecsize = std::numeric_limits<ssize_t>::max();
	
	// read file to get size (HAVE TO DO 2 PASS to get size.)
	// MPI compatible, not OpenMP enabled.
	splash::io::CSVMatrixReader<T> reader(filename.c_str(), atof_type);
	reader.getMatrixSize(nvecs, vecsize);

	// get the minimum array size.
	rows = (rows > 0) ? std::min(static_cast<S>(nvecs), rows) : nvecs;
	cols = (cols > 0) ? std::min(static_cast<S>(vecsize), cols) : vecsize;

	// allocate the data
	splash::ds::aligned_matrix<T> input(rows, cols);

	// now read the data.  // MPI compatible, not OpenMP enabled.
	reader.loadMatrixData(genes, samples, input);

	return input;
};

template <typename T, typename S>
splash::ds::aligned_matrix<T> read_csv_matrix_fast(std::string const & filename, S & rows, S & cols,
	std::vector<std::string> & genes, std::vector<std::string> & samples,
	int const & atof_type = 1
) {
	ssize_t nvecs = std::numeric_limits<ssize_t>::max();
	ssize_t vecsize = std::numeric_limits<ssize_t>::max();
	
	// read file to get size (HAVE TO DO 2 PASS to get size.)
	// MPI compatible, not OpenMP enabled.
	splash::io::CSVMatrixReader2<T> reader(filename.c_str(), atof_type);
	reader.getMatrixSize(nvecs, vecsize);

	// get the minimum array size.
	rows = (rows > 0) ? std::min(static_cast<S>(nvecs), rows) : nvecs;
	cols = (cols > 0) ? std::min(static_cast<S>(vecsize), cols) : vecsize;

	// allocate the data
	splash::ds::aligned_matrix<T> input(rows, cols);

	// now read the data.  // MPI compatible, not OpenMP enabled.
	reader.loadMatrixData(genes, samples, input);

	return input;
};


// template <typename T, typename S>
// splash::ds::aligned_matrix<T> read_csv_matrix_distributed(std::string const & filename, S & rows, S & cols,
// 	std::vector<std::string> & genes, std::vector<std::string> & samples,
// 	int const & atof_type = 1
// ) {
// 	ssize_t nvecs = std::numeric_limits<ssize_t>::max();
// 	ssize_t vecsize = std::numeric_limits<ssize_t>::max();
	
// 	// read file to get size (HAVE TO DO 2 PASS to get size.)
// 	// MPI compatible, not OpenMP enabled.
// 	splash::io::CSVMatrixReader2<T> reader(filename.c_str(), atof_type);
// 	reader.getMatrixSize(nvecs, vecsize);

// 	// get the minimum array size.
// 	rows = (rows > 0) ? std::min(static_cast<S>(nvecs), rows) : nvecs;
// 	cols = (cols > 0) ? std::min(static_cast<S>(vecsize), cols) : vecsize;

// 	// allocate the data
// 	splash::ds::aligned_matrix<T> input(rows, cols);

// 	// now read the data.  // MPI compatible, not OpenMP enabled.
// 	reader.loadMatrixData(genes, samples, input);

// 	return input;
// };


template <typename T>
void write_csv_matrix(std::string const & filename, std::vector<std::string> & row_names, std::vector<std::string> & col_names,
	splash::ds::aligned_matrix<T> const & data) {
	if (filename.length() > 0) {
		// write to file.  MPI enabled.  Not thread enabled.
		// FMT_ROOT_PRINT("name sizes: {}, {}\n", row_names.size(), col_names.size());
		// FMT_ROOT_PRINT("outputing matrix size: {}, {}\n", data.rows(), data.columns());
#ifdef USE_MPI		
		int rank = 0;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		auto allout = data.gather(0);
		FMT_ROOT_PRINT("writing file size: {}, {}\n", allout.rows(), allout.columns());
		if (rank == 0) {
			splash::io::MatrixWriter<T>::storeMatrixData(filename, row_names, col_names, allout);
		}
#else
		splash::io::MatrixWriter<T>::storeMatrixData(filename, row_names, col_names, data);
#endif
	} else {
		// dump to console.  TODO: MPI
		data.print("matrix: ");
	}
};


template <typename T>
void write_csv_matrix_distributed(std::string const & filename, std::vector<std::string> & row_names, std::vector<std::string> & col_names,
	splash::ds::aligned_matrix<T> const & data, MPI_Comm const & comm = MPI_COMM_WORLD) {
	if (filename.length() > 0) {
		// write to file.  MPI enabled.  Not thread enabled.
		// FMT_ROOT_PRINT("name sizes: {}, {}\n", row_names.size(), col_names.size());
		// FMT_ROOT_PRINT("outputing matrix size: {}, {}\n", data.rows(), data.columns());
#ifdef USE_MPI
		splash::io::MatrixWriter<T>::storeMatrixDistributed(filename, row_names, col_names, data, comm);
#else
		splash::io::MatrixWriter<T>::storeMatrixData(filename, row_names, col_names, data);
#endif
	} else {
		// dump to console.  TODO: MPI
		data.print("matrix: ");
	}
};


template <typename T, typename S>
splash::ds::aligned_matrix<T> read_hdf5_matrix(std::string const & filename, 
	std::string const & datasetname, S & rows, S & cols, 
	std::vector<std::string> & row_names, std::vector<std::string> & col_names) {
	if (filename.length() <= 0)  return splash::ds::aligned_matrix<T>();

	ssize_t nvecs = std::numeric_limits<ssize_t>::max();
	ssize_t vecsize = std::numeric_limits<ssize_t>::max();
	
	// read file to get size (HAVE TO DO 2 PASS to get size.)
	// MPI compatible, not OpenMP enabled.
	splash::io::HDF5MatrixReader<T> reader(filename);
	reader.getMatrixSize(datasetname, nvecs, vecsize);

	// get the minimum array size.
	rows = (rows > 0) ? std::min(static_cast<S>(nvecs), rows) : nvecs;
	cols = (cols > 0) ? std::min(static_cast<S>(vecsize), cols) : vecsize;

	// allocate the data
	splash::ds::aligned_matrix<T> input(rows, cols);

	// now read the data.  // MPI compatible, not OpenMP enabled.
	reader.loadMatrixData(datasetname, row_names, col_names, input);

	return input;
};


template <typename T, typename S>
splash::ds::aligned_matrix<T> read_hdf5_matrix_distributed(std::string const & filename, 
	std::string const & datasetname, S & rows, S & cols,
	std::vector<std::string> & row_names, std::vector<std::string> & col_names,
	MPI_Comm const & comm = MPI_COMM_WORLD) {

	if (filename.length() <= 0)  return splash::ds::aligned_matrix<T>();

	ssize_t nvecs = std::numeric_limits<ssize_t>::max();
	ssize_t vecsize = std::numeric_limits<ssize_t>::max();
	
	// MPI compatible, not OpenMP enabled.
	splash::io::HDF5MatrixReader<T> reader(filename);
	reader.getMatrixSize(datasetname, nvecs, vecsize, comm);

	// get the minimum array size.
	rows = (rows > 0) ? std::min(static_cast<S>(nvecs), rows) : nvecs;
	cols = (cols > 0) ? std::min(static_cast<S>(vecsize), cols) : vecsize;

	// now split.
	int rank;
	int procs;
	MPI_Comm_rank(comm, &rank);
	MPI_Comm_size(comm, &procs);
	S extras = rows % procs;
	S local_rows = rows / procs + (rank < extras);

	// allocate the data
	splash::ds::aligned_matrix<T> input(local_rows, cols);

	// now read the data.  // MPI compatible, not OpenMP enabled.
	reader.loadMatrixData(datasetname, row_names, col_names, input, comm);

	return input;
};


template <typename T>
void write_hdf5_matrix(std::string const & filename, 
	std::string const & datasetname,
	std::vector<std::string> & row_names, std::vector<std::string> & col_names,
	splash::ds::aligned_matrix<T> const & data) {
	if (filename.length() > 0) {
		// write to file.  MPI enabled.  Not thread enabled.
		// FMT_ROOT_PRINT("name sizes: {}, {}\n", row_names.size(), col_names.size());
		// FMT_ROOT_PRINT("outputing matrix size: {}, {}\n", data.rows(), data.columns());
		splash::io::HDF5MatrixWriter<T> writer(filename);
#ifdef USE_MPI		
		int rank = 0;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		auto allout = data.gather(0);
		FMT_ROOT_PRINT("writing file size: {}, {}\n", allout.rows(), allout.columns());
		if (rank == 0) {
			writer.storeMatrixData(datasetname, row_names, col_names, allout);
		}
		MPI_Barrier(MPI_COMM_WORLD);
#else
		writer::storeMatrixData(datasetname, row_names, col_names, data);
#endif
	} else {
		// dump to console.  TODO: MPI
		data.print("matrix: ");
	}
};


template <typename T>
void write_hdf5_matrix_distributed(std::string const & filename, 
	std::string const & datasetname,
	std::vector<std::string> & row_names, std::vector<std::string> & col_names,
	splash::ds::aligned_matrix<T> const & data, MPI_Comm const & comm = MPI_COMM_WORLD) {
	if (filename.length() > 0) {
		// write to file.  MPI enabled.  Not thread enabled.
		// FMT_ROOT_PRINT("name sizes: {}, {}\n", row_names.size(), col_names.size());
		// FMT_ROOT_PRINT("outputing matrix size: {}, {}\n", data.rows(), data.columns());
		splash::io::HDF5MatrixWriter<T> writer(filename);
#ifdef USE_MPI
		writer.storeMatrixDistributed(datasetname, row_names, col_names, data, comm);

		MPI_Barrier(comm);
#else
		writer.storeMatrixData(datasetname, row_names, col_names, data);
#endif
	} else {
		// dump to console.  TODO: MPI
		data.print("matrix: ");
	}
};


template <typename T, typename S>
splash::ds::aligned_matrix<T> read_matrix(std::string const & filename, 
	std::string const & datasetname, S & rows, S & cols,
	std::vector<std::string> & genes, std::vector<std::string> & samples, 
	bool skip = false,
	int const & atof_type = 1 ) {
	
	if (splash::utils::endsWith(filename, ".csv")) {
		return read_csv_matrix_fast<T>(filename, rows, cols, genes, samples, atof_type);
	} else if (splash::utils::endsWith(filename, ".exp")) {
		return read_exp_matrix_fast<T>(filename, rows, cols, genes, samples, skip);
	} else if ((splash::utils::endsWith(filename, "hdf5")) || (splash::utils::endsWith(filename, ".h5"))) {
		return read_hdf5_matrix<T>(filename, datasetname, rows, cols, genes, samples);
	} else {
		FMT_PRINT_ERR("ERROR: unsupported file format {}.", filename);
		return splash::ds::aligned_matrix<T>();
	}
};


template <typename T>
void write_matrix(std::string const & filename, 
	std::string const & datasetname,
	std::vector<std::string> & row_names, std::vector<std::string> & col_names,
	splash::ds::aligned_matrix<T> const & data) {
	
	if (splash::utils::endsWith(filename, ".csv")) {
		write_csv_matrix(filename, row_names, col_names, data);
	} else if (splash::utils::endsWith(filename, ".exp")) {
		write_exp_matrix(filename, row_names, col_names, data);
	} else if ((splash::utils::endsWith(filename, "hdf5")) || (splash::utils::endsWith(filename, ".h5"))) {
		write_hdf5_matrix(filename, datasetname, row_names, col_names, data);
	} else {
		FMT_PRINT_ERR("ERROR: unsupported file format {}.", filename);
	}
};

template <typename T>
void write_matrix_distributed(std::string const & filename, 
	std::string const & datasetname,
	std::vector<std::string> & row_names, std::vector<std::string> & col_names,
	splash::ds::aligned_matrix<T> const & data, MPI_Comm const & comm = MPI_COMM_WORLD) {
	
	if (splash::utils::endsWith(filename, ".csv")) {
		write_csv_matrix_distributed(filename, row_names, col_names, data, comm);
	} else if (splash::utils::endsWith(filename, ".exp")) {
		write_exp_matrix(filename, row_names, col_names, data);
	} else if ((splash::utils::endsWith(filename, "hdf5")) || (splash::utils::endsWith(filename, ".h5"))) {
		write_hdf5_matrix_distributed(filename, datasetname, row_names, col_names, data, comm);
	} else {
		FMT_PRINT_ERR("ERROR: unsupported file format {}.", filename);
	}
};