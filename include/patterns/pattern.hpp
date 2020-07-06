#pragma once

#include "utils/partition.hpp"
#include "ds/aligned_tiles.hpp"

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace splash { namespace pattern { 



// rows in 2D.  tested okay.
template <typename IN, typename Op, typename OUT>
class M2MProcessor {

    protected:
	    splash::utils::partitioner1D<PARTITION_EQUAL> partitioner;
        using part1D_type = splash::utils::partition<size_t>;
        int procs;
        int rank;

    public:
        using InputType = IN;
        using OutputType = OUT;

#ifdef USE_MPI
        M2MProcessor(MPI_Comm comm = MPI_COMM_WORLD) {
            MPI_Comm_size(comm, &procs);
            MPI_Comm_rank(comm, &rank);
        }
#else
        M2MProcessor() : procs(1), rank(0) {};
#endif
        M2MProcessor(int const & _procs, int const & _rank) :
            procs(_procs), rank(_rank) {};

        OUT operator()(IN const & input, Op const & op) const {
            int threads = 1;
            int thread_id = 0;

            OUT output(input.rows(), input.columns());
#ifdef USE_OPENMP
#pragma omp parallel
            {
                threads = omp_get_num_threads();
                thread_id = omp_get_thread_num();
#endif
                // partition the local 2D tiles.  omp_tile_parts.offset is local to this processor.
                part1D_type omp_tile_parts = partitioner.get_partition(output.rows(), threads, thread_id);
                PRINT_MPI("NORM thread %d partition: ", thread_id);
                omp_tile_parts.print();

                // iterate over rows.
                size_t rid = omp_tile_parts.offset;
                for (size_t i = 0; i < omp_tile_parts.size; ++i, ++rid) {
                    op(input.data(rid),  input.columns(), output.data(rid));
                }

#ifdef USE_OPENMP
            }
#endif
            return output;
        }

};


// tiles in 2D
template <typename IN, typename Op, typename OUT>
class MM2MProcessor {
    protected:
        splash::utils::partitioner2D<PARTITION_FIXED> partitioner2d;
        splash::utils::partitioner1D<PARTITION_EQUAL> partitioner;
	    splash::utils::banded_diagonal_filter part_filter;
	    // splash::utils::upper_triangle_filter part_filter;
	
        using part1D_type = splash::utils::partition<size_t>;
        using part2D_type = splash::utils::partition2D<size_t>;
        using tiles_type = splash::ds::aligned_tiles<typename OUT::data_type, part2D_type>;

        int procs;
        int rank;
	
    public:
        using InputType = IN;
        using OutputType = OUT;

#ifdef USE_MPI
        MM2MProcessor(MPI_Comm comm = MPI_COMM_WORLD) {
            MPI_Comm_size(comm, &procs);
            MPI_Comm_rank(comm, &rank);
        }
#else
        MM2MProcessor() : procs(1), rank(0) {};
#endif
        MM2MProcessor(int const & _procs, int const & _rank) :
            procs(_procs), rank(_rank) {};

        OUT operator()(IN const & input1, IN const & input2, Op const & op) const {
            // ---- fixed-size partiton input and filter for tiles t
            auto stime = getSysTime();
            std::vector<part2D_type> all_tile_parts = partitioner2d.divide(input1.rows(), input2.rows(), 
                    static_cast<typename OUT::size_type>(PARTITION_TILE_DIM), 
                    static_cast<typename OUT::size_type>(PARTITION_TILE_DIM) );
            std::vector<part2D_type> tile_parts = part_filter.filter( all_tile_parts );
            PRINT_MPI("Partitions: 2D %lu -> filtered %lu\n", all_tile_parts.size(), tile_parts.size());

            // ---- partition the partitions for MPI
            part1D_type mpi_tile_parts = partitioner.get_partition(tile_parts.size(), this->procs, this->rank);
            PRINT_MPI("MPI Rank %d partition: ", this->rank);
            mpi_tile_parts.print();

            auto etime = getSysTime();
            PRINT_MPI_ROOT("Correlation Partitioned in %f sec\n", get_duration_s(stime, etime));

            // ---- compute correlation
            stime = getSysTime();

        	// ---- set up the temporary output, tiled, contains the partitions to process.
	        // PRINT_MPI("[pearson TILES] ");
	        tiles_type tiles(tile_parts.data() + mpi_tile_parts.offset, mpi_tile_parts.size);

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
		        PRINT_MPI("thread %d partition: ", thread_id);
		        omp_tile_parts.print();

                // run
                // get range of global partitions for this thread.
                // tiles.part(omp_tile_parts.offset).print();

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
                    // part.print();

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
        		// PRINT_MPI("CORR BOUNDS r: %lu %lu, c %lu %lu\n", rmin, rmax-rmin, cmin, cmax-cmin);

#ifdef USE_OPENMP
        	}
#endif
        	etime = getSysTime();
	        PRINT_MPI_ROOT("Computed pearson in %f sec\n", get_duration_s(stime, etime));


        	// ============= repartition output
	        stime = getSysTime();
	        part2D_type bounds;

            PRINT_MPI("TILES BOUNDS: ");  tiles.get_bounds().print();
            PRINT_MPI("TILES COUNT: %lu\n", tiles.size());
            // OKAY    tiles.print();

        	part1D_type row_part = partitioner.get_partition(input1.rows(), this->procs, this->rank);
            // row_part.print();
            // with transpose.
            // PRINT_MPI("[pearson TRANSPOSE] ");
            tiles_type transposed = tiles.transpose();
            PRINT_MPI("TRANSPOSED BOUNDS: ");  transposed.get_bounds().print();
            PRINT_MPI("TRANSPOSED COUNT: %lu\n", transposed.size());
            // OKAY   transposed.print();

            // TODO: make a reflect function or something else that avoids processing the diagonal.
            // PRINT_MPI("[pearson ADD] ");
            tiles_type all_tiles = tiles.merge(transposed);
            PRINT_MPI("MERGED BOUNDS: ");  all_tiles.get_bounds().print();
            PRINT_MPI("MERGED COUNT: %lu\n", all_tiles.size());
            // OKAY all_tiles.print();

            // PRINT_MPI("[pearson ROW_PARTITION] ");
            tiles_type parted_tiles = all_tiles.row_partition(row_part);
            // PRINT_MPI("[DEBUG] Tiles: %ld + %ld = %ld, partitioned %ld\n", tiles.size(), transposed.size(), all_tiles.size(), parted_tiles.size());
            // PRINT_MPI("[DEBUG] Tiles: %ld + %ld = %ld, partitioned %ld\n", tiles.allocated(), transposed.allocated(), all_tiles.allocated(), parted_tiles.allocated());
            // OKAY  parted_tiles.print();

            // get actual bounds.  set up first for MPI.
            bounds = parted_tiles.get_bounds();
// #else
//  		bounds = part2D_type(part1D_type(0, input1.rows(), 0), part1D_type(0, input2.columns(), 0), 0, 1); 
// #endif
            PRINT_MPI("REPARTED BOUNDS: "); bounds.print();
            PRINT_MPI("REPARTED COUNT: %lu\n", parted_tiles.size());

            OUT output(bounds.r.size, input2.rows());
	        parted_tiles.copy_to(output, bounds.r.offset, 0);
            // TODO: fix copy to - we are partitioning rows equally and allocating output in the same way.
            //       however, a tile may straddle a boundary since we partition the tiles by offset and do not split tiles.
            // the output size needs to be allocated after tile partitioning to get the bounds properly.
	
        	etime = getSysTime();
	        PRINT_MPI("Reorder tiles in %f sec\n", get_duration_s(stime, etime));

            return output;

        }

};


}}