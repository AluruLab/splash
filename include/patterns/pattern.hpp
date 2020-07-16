#pragma once

#include "utils/partition.hpp"
#include "ds/aligned_tiles.hpp"
#include "ds/aligned_vector.hpp"
#include "ds/aligned_matrix.hpp"

#include <cassert>

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace splash { namespace pattern { 

// differs from kernel in that kernel is meant to be single thread, these account for distributed.

// TODO:
// [ ] change  IN, OUT to element type instead of container types. Op is operating on pointers.
// [ ] combine the MM2M and MVMV2M patterns.  
// [ ] work with partitioned input. and use shift when computing.

enum DIM_INDEX : int { ROW = 1, COLUMN = 2 };


// reduction along a particular DIM_INDEX.
template <typename IN, typename Op, typename OUT, int DIM>
class Reduce;
// possibilities - MAT->V along column, V->S,... 

// row wise reduction in matrix. 
template <typename IT, typename Op, typename OT>
class Reduce<splash::ds::aligned_matrix<IT>, Op, splash::ds::aligned_vector<OT>, DIM_INDEX::ROW> {

    protected:
	    splash::utils::partitioner1D<PARTITION_EQUAL> partitioner;
        using part1D_type = splash::utils::partition<size_t>;
        int procs;
        int rank;

    public:
        using InputType = splash::ds::aligned_matrix<IT>;
        using OutputType = splash::ds::aligned_vector<OT>;

#ifdef USE_MPI
        Reduce(MPI_Comm comm = MPI_COMM_WORLD) {
            MPI_Comm_size(comm, &procs);
            MPI_Comm_rank(comm, &rank);
        }
#else
        Reduce() : procs(1), rank(0) {};
#endif
        Reduce(int const & _procs, int const & _rank) :
            procs(_procs), rank(_rank) {};

        void operator()(InputType const & input, Op const & op, OutputType & output) const {
            assert((output.size() == input.rows()) && "Reduce requires output vector size to be same as input row count.");

            int threads = 1;
            int thread_id = 0;

#ifdef USE_OPENMP
#pragma omp parallel
            {
                threads = omp_get_num_threads();
                thread_id = omp_get_thread_num();
#endif
                // partition the local 2D tiles.  omp_tile_parts.offset is local to this processor.
                part1D_type omp_tile_parts = partitioner.get_partition(output.size(), threads, thread_id);
                // PRINT_RT("NORM thread %d partition: ", thread_id);
                // omp_tile_parts.print("OMP TILES: ");

                // iterate over rows.
                size_t rid = omp_tile_parts.offset;
                for (size_t i = 0; i < omp_tile_parts.size; ++i, ++rid) {
                    output[rid] = op(input.data(rid),  input.columns());
                }

#ifdef USE_OPENMP
            }
#endif
        }
};


// Transform each element individually.  same dimensionality.
template <typename IN, typename Op, typename OUT>
class Transform;


template <typename IT, typename Op, typename OT>
class Transform<splash::ds::aligned_matrix<IT>, Op, splash::ds::aligned_matrix<OT>> {

    protected:
	    splash::utils::partitioner1D<PARTITION_EQUAL> partitioner;
        using part1D_type = splash::utils::partition<size_t>;
        int procs;
        int rank;

    public:
        using InputType = splash::ds::aligned_matrix<IT>;
        using OutputType = splash::ds::aligned_matrix<OT>;

#ifdef USE_MPI
        Transform(MPI_Comm comm = MPI_COMM_WORLD) {
            MPI_Comm_size(comm, &procs);
            MPI_Comm_rank(comm, &rank);
        }
#else
        Transform() : procs(1), rank(0) {};
#endif
        Transform(int const & _procs, int const & _rank) :
            procs(_procs), rank(_rank) {};

        void operator()(InputType const & input, Op const & op, OutputType & output) const {
            assert((output.rows() == input.rows())  && "Transform requires output and input to have same number of rows.");

            int threads = 1;
            int thread_id = 0;

#ifdef USE_OPENMP
#pragma omp parallel
            {
                threads = omp_get_num_threads();
                thread_id = omp_get_thread_num();
#endif
                // partition the local 2D tiles.  omp_tile_parts.offset is local to this processor.
                part1D_type omp_tile_parts = partitioner.get_partition(output.rows(), threads, thread_id);
                omp_tile_parts.print("NORM");

                // iterate over rows.
                size_t rid = omp_tile_parts.offset;
                for (size_t i = 0; i < omp_tile_parts.size; ++i, ++rid) {
                    op(input.data(rid),  input.columns(), output.data(rid));
                }

#ifdef USE_OPENMP
            }
#endif
        }

};


// Multiply pattern, i.e. M1 * M2' -> M3
template <typename IN, typename Op, typename OUT>
class InnerProduct;

template <typename IT, typename Op, typename OT>
class InnerProduct<splash::ds::aligned_matrix<IT>, Op, splash::ds::aligned_matrix<OT>> {
    public:
        using InputType = splash::ds::aligned_matrix<IT>;
        using OutputType = splash::ds::aligned_matrix<OT>;

    protected:
        splash::utils::partitioner2D<PARTITION_FIXED> partitioner2d;
        splash::utils::partitioner1D<PARTITION_EQUAL> partitioner;
	    splash::utils::banded_diagonal_filter part_filter;
	    // splash::utils::upper_triangle_filter part_filter;
	
        using part1D_type = splash::utils::partition<size_t>;
        using part2D_type = splash::utils::partition2D<size_t>;
        using tiles_type = splash::ds::aligned_tiles<typename OutputType::data_type, part2D_type>;

        int procs;
        int rank;
	
    public:
#ifdef USE_MPI
        InnerProduct(MPI_Comm comm = MPI_COMM_WORLD) {
            MPI_Comm_size(comm, &procs);
            MPI_Comm_rank(comm, &rank);
        }
#else
        InnerProduct() : procs(1), rank(0) {};
#endif
        InnerProduct(int const & _procs, int const & _rank) :
            procs(_procs), rank(_rank) {};

        void operator()(InputType const & input1, InputType const & input2, Op const & op, OutputType & output) const {
            if ((output.rows() != input1.rows()) || (output.columns() != input2.rows())) 
                PRINT_RT("InnerProduct: input1 rows:  %lu, input2 rows: %lu, output rows: %lu, columns %lu\n",
                    input1.rows(), input2.rows(), output.rows(), output.columns());
            assert(((output.rows() == input1.rows()) && (output.columns() == input2.rows())) && "InnerProduct requires output rows and input1 rows to be same, and output columns and input2 rows to be same.");

            // ---- fixed-size partiton input and filter for tiles t
            auto stime = getSysTime();
            std::vector<part2D_type> all_tile_parts = partitioner2d.divide(input1.rows(), input2.rows(), 
                    static_cast<typename OutputType::size_type>(PARTITION_TILE_DIM), 
                    static_cast<typename OutputType::size_type>(PARTITION_TILE_DIM) );
            std::vector<part2D_type> tile_parts = part_filter.filter( all_tile_parts );
            // PRINT_RT("Partitions: 2D %lu -> filtered %lu\n", all_tile_parts.size(), tile_parts.size());

            // ---- partition the partitions for MPI
            part1D_type mpi_tile_parts = partitioner.get_partition(tile_parts.size(), this->procs, this->rank);
            // PRINT_RT("MPI Rank %d partition: ", this->rank);
            // mpi_tile_parts.print("MPI TILEs: ");

            auto etime = getSysTime();
            ROOT_PRINT("Correlation Partitioned in %f sec\n", get_duration_s(stime, etime));

            // ---- compute correlation
            stime = getSysTime();

        	// ---- set up the temporary output, tiled, contains the partitions to process.
	        // PRINT_RT("[pearson TILES] ");
	        tiles_type tiles(tile_parts.data() + mpi_tile_parts.offset, mpi_tile_parts.size);
            input1.print("INPUT: ");

            // OpenMP stuff.
            int threads = 1;
            int thread_id = 0;

#ifdef USE_OPENMP
#pragma omp parallel
            {
                threads = omp_get_num_threads();
                thread_id = omp_get_thread_num();
#endif
                // partition the local 2D tiles.  omp_tile_parts.offset is local to this processor.
		        part1D_type omp_tile_parts = partitioner.get_partition(mpi_tile_parts.size, threads, thread_id);
		        // PRINT_RT("thread %d partition: ", thread_id);
		        // omp_tile_parts.print("OMP PARTITION: ");

                // run
                // get range of global partitions for this thread.
                // tiles.part(omp_tile_parts.offset).print("OMP START PART");

                // size_t rmin = std::numeric_limits<size_t>::max();
                // size_t cmin = std::numeric_limits<size_t>::max();
                // size_t rmax = std::numeric_limits<size_t>::lowest();
                // size_t cmax = std::numeric_limits<size_t>::lowest();

                // iterate over all tiles
                size_t row, col, row_end, col_end;
                size_t id = omp_tile_parts.offset;
                auto i = omp_tile_parts.offset;
                for (i = 0; i < omp_tile_parts.size; ++i, ++id) {
                    auto part = tiles.part(id);  // id is a linear id.  tiles are filled sequentially within each proc-thread.
                    auto data = tiles.data(id);
                    // part.print("PART: ");

                    // work on 1 tile
                    row = part.r.offset;
                    row_end = row + part.r.size;

                    // rmin = std::min(rmin, row);
                    // cmin = std::min(cmin, part.c.offset);
                    // rmax = std::max(rmax, row_end);
                    // cmax = std::max(cmax, part.c.offset + part.c.size);

                    // work on 1 row
                    for (; row < row_end; ++row) {
                        // for row in tile
                        col = part.c.offset;
                        col_end = col + part.c.size;

                        // work on 1 column
                        for (; col < col_end; ++col) {
                            // no skipping entries within a tile, otherwise downstream copy into matrix would have missing entries. 

                            // compute correlation
                            *data = op(input1.data(row), input2.data(col), input1.columns());
                            ++data;
                        }
                    }
                }
        		// PRINT_RT("CORR BOUNDS r: %lu %lu, c %lu %lu\n", rmin, rmax-rmin, cmin, cmax-cmin);

#ifdef USE_OPENMP
        	}
#endif
        	etime = getSysTime();
	        ROOT_PRINT("Computed in %f sec\n", get_duration_s(stime, etime));


        	// ============= repartition output
	        stime = getSysTime();

            tiles.get_bounds().print("TILES BOUNDS: ");
            PRINT_RT("TILES COUNT: %lu\n", tiles.size());
            tiles.print("TILES: ");

        	part1D_type row_part = partitioner.get_partition(input1.rows(), this->procs, this->rank);
            row_part.print("ROW PARTITIONS: ");
            // with transpose.
            // PRINT_RT("[pearson TRANSPOSE] ");
            tiles_type transposed = tiles.transpose();
            // transposed.get_bounds().print("TRANSPOSED BOUNDS: ");
            // PRINT_RT("TRANSPOSED COUNT: %lu\n", transposed.size());
            // OKAY   transposed.print("TRANSPOSED TILES: ");

            // TODO: make a reflect function or something else that avoids processing the diagonal.
            // PRINT_RT("[pearson ADD] ");
            tiles_type all_tiles = tiles.merge(transposed);
            // all_tiles.get_bounds().print("MERGED BOUNDS: ");
            // PRINT_RT("MERGED COUNT: %lu\n", all_tiles.size());
            // OKAY all_tiles.print("ALL TILES: ");

            // PRINT_RT("[pearson ROW_PARTITION] ");
            tiles_type parted_tiles = all_tiles.row_partition(row_part);
            // PRINT_RT("[DEBUG] Tiles: %ld + %ld = %ld, partitioned %ld\n", tiles.size(), transposed.size(), all_tiles.size(), parted_tiles.size());
            // PRINT_RT("[DEBUG] Tiles: %ld + %ld = %ld, partitioned %ld\n", tiles.allocated(), transposed.allocated(), all_tiles.allocated(), parted_tiles.allocated());
            // parted_tiles.print("PARTED TILES: ");

            // get actual bounds.  set up first for MPI.
            part2D_type bounds = parted_tiles.get_bounds();
// #else
//  		bounds = part2D_type(part1D_type(0, input1.rows(), 0), part1D_type(0, input2.columns(), 0), 0, 1); 
// #endif
            // bounds.print("REPARTED BOUNDS: ");
            // PRINT_RT("REPARTED COUNT: %lu\n", parted_tiles.size());

            assert((output.rows() >= bounds.r.size) && "Output rows have to be at least equal to the bounds row size" );
            output.resize(bounds.r.size, input2.rows());
	        parted_tiles.copy_to(output, bounds.r.offset, 0);
            // TODO: fix copy to - we are partitioning rows equally and allocating output in the same way.
            //       however, a tile may straddle a boundary since we partition the tiles by offset and do not split tiles.
            // the output size needs to be allocated after tile partitioning to get the bounds properly.
	
        	etime = getSysTime();
	        PRINT_RT("Reorder tiles in %f sec\n", get_duration_s(stime, etime));
        }

};

/*
 * TODO: COMMENTED OUT FOR NOW BECAUSE NO GREAT WAY TO  PATTERN THIS YET.
// // tiles in 2D
// template <typename INM, typename INV, typename Op, typename OUT>
// class MVMV2MProcessor {
//     protected:
//         splash::utils::partitioner2D<PARTITION_FIXED> partitioner2d;
//         splash::utils::partitioner1D<PARTITION_EQUAL> partitioner;
// 	    splash::utils::banded_diagonal_filter part_filter;
// 	    // splash::utils::upper_triangle_filter part_filter;
	
//         using part1D_type = splash::utils::partition<size_t>;
//         using part2D_type = splash::utils::partition2D<size_t>;
//         using tiles_type = splash::ds::aligned_tiles<typename OutputType::data_type, part2D_type>;

//         int procs;
//         int rank;
	
//     public:
//         using InputType = INM;
//         using OutputType = OutputType;

// #ifdef USE_MPI
//         MVMV2MProcessor(MPI_Comm comm = MPI_COMM_WORLD) {
//             MPI_Comm_size(comm, &procs);
//             MPI_Comm_rank(comm, &rank);
//         }
// #else
//         MVMV2MProcessor() : procs(1), rank(0) {};
// #endif
//         MVMV2MProcessor(int const & _procs, int const & _rank) :
//             procs(_procs), rank(_rank) {};

//         OutputType operator()(INM const & input1, INV const & input1_aux, 
//             INM const & input2, INV const & input2_aux, Op const & op) const {
//             // ---- fixed-size partiton input and filter for tiles t
//             auto stime = getSysTime();
//             std::vector<part2D_type> all_tile_parts = partitioner2d.divide(input1.rows(), input2.rows(), 
//                     static_cast<typename OutputType::size_type>(PARTITION_TILE_DIM), 
//                     static_cast<typename OutputType::size_type>(PARTITION_TILE_DIM) );
//             std::vector<part2D_type> tile_parts = part_filter.filter( all_tile_parts );
//             PRINT_RT("Partitions: 2D %lu -> filtered %lu\n", all_tile_parts.size(), tile_parts.size());

//             // ---- partition the partitions for MPI
//             part1D_type mpi_tile_parts = partitioner.get_partition(tile_parts.size(), this->procs, this->rank);
//             PRINT_RT("MPI Rank %d partition: ", this->rank);
//             mpi_tile_parts.print("MPI PARTITION: ");

//             auto etime = getSysTime();
//             ROOT_PRINT("Correlation Partitioned in %f sec\n", get_duration_s(stime, etime));

//             // ---- compute correlation
//             stime = getSysTime();

//         	// ---- set up the temporary output, tiled, contains the partitions to process.
// 	        // PRINT_RT("[pearson TILES] ");
// 	        tiles_type tiles(tile_parts.data() + mpi_tile_parts.offset, mpi_tile_parts.size);

//             // OpenMP stuff.
//             int threads = 1;
//             int thread_id = 0;

// #ifdef USE_OPENMP
// #pragma omp parallel
//             {
//                 threads = omp_get_num_threads();
//                 thread_id = omp_get_thread_num();
// #endif
//                 // partition the local 2D tiles.  omp_tile_parts.offset is local to this processor.
// 		        part1D_type omp_tile_parts = partitioner.get_partition(mpi_tile_parts.size, threads, thread_id);
// 		        PRINT_RT("thread %d partition: ", thread_id);
// 		        omp_tile_parts.print(" OMP PARTITION");

//                 // run
//                 // get range of global partitions for this thread.
//                 // tiles.part(omp_tile_parts.offset).print("");

//                 // size_t rmin = std::numeric_limits<size_t>::max();
//                 // size_t cmin = std::numeric_limits<size_t>::max();
//                 // size_t rmax = std::numeric_limits<size_t>::lowest();
//                 // size_t cmax = std::numeric_limits<size_t>::lowest();

//                 // iterate over all tiles
//                 size_t row, col, row_end, col_end;
//                 size_t id = omp_tile_parts.offset;
//                 auto i = omp_tile_parts.offset;
//                 for (i = 0; i < omp_tile_parts.size; ++i, ++id) {
//                     auto part = tiles.part(id);  // id is a linear id.  tiles are filled sequentially within each proc-thread.
//                     auto data = tiles.data(id);
//                     // part.print("OMP PART: ");

//                     // work on 1 tile
//                     row = part.r.offset;
//                     row_end = row + part.r.size;

//                     // rmin = std::min(rmin, row);
//                     // cmin = std::min(cmin, part.c.offset);
//                     // rmax = std::max(rmax, row_end);
//                     // cmax = std::max(cmax, part.c.offset + part.c.size);

//                     // work on 1 row
//                     for (; row < row_end; ++row) {
//                         // for row in tile
//                         col = part.c.offset;
//                         col_end = col + part.c.size;

//                         // work on 1 column
//                         for (; col < col_end; ++col) {
//                             // no skipping entries within a tile, otherwise downstream copy into matrix would have missing entries. 

//                             // compute correlation
//                             *data = op(input1.data(row), input1_aux[row], 
//                                 input2.data(col), input2_aux[col], input1.columns());
//                             ++data;
//                         }
//                     }
//                 }
//         		// PRINT_RT("CORR BOUNDS r: %lu %lu, c %lu %lu\n", rmin, rmax-rmin, cmin, cmax-cmin);

// #ifdef USE_OPENMP
//         	}
// #endif
//         	etime = getSysTime();
// 	        ROOT_PRINT("Computed in %f sec\n", get_duration_s(stime, etime));


//         	// ============= repartition output
// 	        stime = getSysTime();
// 	        part2D_type bounds;

//             PRINT_RT("TILES BOUNDS: ");  tiles.get_bounds().print("TILES BOUNDS: ");
//             PRINT_RT("TILES COUNT: %lu\n", tiles.size());
//             // OKAY    tiles.print("TILES: ");

//         	part1D_type row_part = partitioner.get_partition(input1.rows(), this->procs, this->rank);
//             // row_part.print("ROW PARTITION: ");
//             // with transpose.
//             // PRINT_RT("[pearson TRANSPOSE] ");
//             tiles_type transposed = tiles.transpose();
//             PRINT_RT("TRANSPOSED BOUNDS: ");  transposed.get_bounds().print("TRANSPOSED BOUNDS: ");
//             PRINT_RT("TRANSPOSED COUNT: %lu\n", transposed.size());
//             // OKAY   transposed.print("TRANSPOSE: ");

//             // TODO: make a reflect function or something else that avoids processing the diagonal.
//             // PRINT_RT("[pearson ADD] ");
//             tiles_type all_tiles = tiles.merge(transposed);
//             PRINT_RT("MERGED BOUNDS: ");  all_tiles.get_bounds().print("MERGED BOUNDS: ");
//             PRINT_RT("MERGED COUNT: %lu\n", all_tiles.size());
//             // OKAY all_tiles.print("ADD: ");

//             // PRINT_RT("[pearson ROW_PARTITION] ");
//             tiles_type parted_tiles = all_tiles.row_partition(row_part);
//             // PRINT_RT("[DEBUG] Tiles: %ld + %ld = %ld, partitioned %ld\n", tiles.size(), transposed.size(), all_tiles.size(), parted_tiles.size());
//             // PRINT_RT("[DEBUG] Tiles: %ld + %ld = %ld, partitioned %ld\n", tiles.allocated(), transposed.allocated(), all_tiles.allocated(), parted_tiles.allocated());
//             // OKAY  parted_tiles.print("ROW PART: ");

//             // get actual bounds.  set up first for MPI.
//             bounds = parted_tiles.get_bounds();
// // #else
// //  		bounds = part2D_type(part1D_type(0, input1.rows(), 0), part1D_type(0, input2.columns(), 0), 0, 1); 
// // #endif
//             bounds.print("REPARTED BOUNDS: ");
//             PRINT_RT("REPARTED COUNT: %lu\n", parted_tiles.size());

//             output.resize(bounds.r.size, input2.rows());
// 	        parted_tiles.copy_to(output, bounds.r.offset, 0);
//             // TODO: fix copy to - we are partitioning rows equally and allocating output in the same way.
//             //       however, a tile may straddle a boundary since we partition the tiles by offset and do not split tiles.
//             // the output size needs to be allocated after tile partitioning to get the bounds properly.
	
//         	etime = getSysTime();
// 	        PRINT_RT("Reorder tiles in %f sec\n", get_duration_s(stime, etime));

//             return output;

//         }

// };
*/
}}