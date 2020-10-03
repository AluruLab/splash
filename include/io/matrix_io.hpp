#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include "kernel/random.hpp"
#include "ds/aligned_matrix.hpp"
#include "utils/report.hpp"
#include "utils/partition.hpp"
#include "io/EXPMatrixReader.hpp"
#include "io/EXPMatrixReader2.hpp"
#include "io/CSVMatrixReader.hpp"
#include "io/CSVMatrixReader2.hpp"
#include "io/MatrixWriter.hpp"
#include "io/EXPMatrixWriter.hpp"

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

template <typename T>
void write_exp_matrix(std::string const & filename, std::vector<std::string> & row_names, std::vector<std::string> & col_names,
	splash::ds::aligned_matrix<T> const & data) {
		// NOTE: rank 0 writes out.
			// write to file.  MPI enabled.  Not thread enabled.
#ifdef USE_MPI
		int rank = 0;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		auto allout = data.gather(0);
		ROOT_PRINT("writing file size: %lu, %lu\n", allout.rows(), allout.columns());
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


template <typename T>
void write_csv_matrix(std::string const & filename, std::vector<std::string> & row_names, std::vector<std::string> & col_names,
	splash::ds::aligned_matrix<T> const & data) {
	if (filename.length() > 0) {
		// write to file.  MPI enabled.  Not thread enabled.
		// ROOT_PRINT("name sizes: %lu, %lu\n", row_names.size(), col_names.size());
		// ROOT_PRINT("outputing matrix size: %lu, %lu\n", data.rows(), data.columns());
#ifdef USE_MPI		
		int rank = 0;
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		auto allout = data.gather(0);
		ROOT_PRINT("writing file size: %lu, %lu\n", allout.rows(), allout.columns());
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