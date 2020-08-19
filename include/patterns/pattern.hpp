#pragma once

#include "utils/partition.hpp"
#include "ds/aligned_tiles.hpp"
#include "ds/aligned_vector.hpp"
#include "ds/aligned_matrix.hpp"

#include <vector>
#include <cassert>

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace splash { namespace pattern { 

/**
 * DATA ACCESS AND PROCESSING PATTERNS
 * note that any resources used by OMP parallel regions should be threadsafe.
 * This  includes kernel objects that may have internal buffers. - may be enough to just replicate these.
 */
// differs from kernel in that kernel is meant to be single thread, these account for distributed.


// OMP ISSUE:   making private operators in OMP by copy constructor does not actually call copy constructor.
//                       the reason may be: OMP thread makes a shallow copy because buffers are pointers and value copied. 
//                       OMP's op instances are temporary objects, which with compiler optimization were made as the thread private objects without actually calling the copy constructor.
//   firstprivate and private tags are not enough because they do not appear to be invoking the copy constructor.
//  alternative - create an array of ops outside of the parallel region, 1 per thread.  using vector for this, must use push_back which enforces copy construction.

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

    public:
        using InputType = splash::ds::aligned_matrix<IT>;
        using OutputType = splash::ds::aligned_vector<OT>;

    protected:
        // same input and output rows.
        void operator()(InputType const & input, part1D_type part, Op const & _op, OutputType & output) const {
            assert((output.size() == input.rows()) && "Reduce requires output vector size to be same as input row count.");

            // split the input amongst the processors.

            // ---- parallel compute
#ifdef USE_OPENMP
#pragma omp parallel
            {
                int threads = omp_get_num_threads();
                int thread_id = omp_get_thread_num();
#else 
                int threads = 1;
                int thread_id = 0;
#endif
                Op op;
                op.copy_parameters(_op);

                // partition the local 2D tiles.  omp_tile_parts.offset is local to this processor.
                // ROOT_PRINT_RT("partitioning info : %lu, %d, %d\n", output.size(), threads, thread_id);
                part1D_type omp_tile_parts = partitioner.get_partition(part, threads, thread_id);
                // PRINT_RT("NORM thread %d partition: ", thread_id);
                omp_tile_parts.print("OMP REDUC TILES: ");

                // iterate over rows.
                size_t rid = omp_tile_parts.offset;
                for (size_t i = 0; i < omp_tile_parts.size; ++i, ++rid) {
                    output[rid] = op(input.data(rid),  input.columns());
                }

#ifdef USE_OPENMP
#pragma omp barrier
            }
#endif
        }

    public:
        void operator()(InputType const & input, Op const & _op, OutputType & output) const {
            this->operator()(input, part1D_type(0, input.rows(), 0), _op, output);
        }
};

// // column wise reduction in matrix. 
// template <typename IT, typename Op, typename OT>
// class Reduce<splash::ds::aligned_matrix<IT>, Op, splash::ds::aligned_vector<OT>, DIM_INDEX::COLUMN> {

//     protected:
// 	    splash::utils::partitioner1D<PARTITION_EQUAL> partitioner;
//         using part1D_type = splash::utils::partition<size_t>;

//     public:
//         using InputType = splash::ds::aligned_matrix<IT>;
//         using OutputType = splash::ds::aligned_vector<OT>;

//     protected:
//         // same input and output rows.
//         void operator()(InputType const & input, part1D_type part, Op const & _op, OutputType & output) const {
//             assert((output.size() == input.columns()) && "Reduce requires output vector size to be same as input row count.");

//             // split the input amongst the processors.
//             splash::ds::aligned_matrix<OT> buffers(omp_get_max_threads(), input.columns());

//             // ---- parallel compute
// #ifdef USE_OPENMP
// #pragma omp parallel
//             {
//                 int threads = omp_get_num_threads();
//                 int thread_id = omp_get_thread_num();
// #else 
//                 int threads = 1;
//                 int thread_id = 0;
// #endif
//                 Op op;
//                 op.copy_parameters(_op);

//                 // partition the local 2D tiles.  omp_tile_parts.offset is local to this processor.
//                 // ROOT_PRINT_RT("partitioning info : %lu, %d, %d\n", output.size(), threads, thread_id);
//                 part1D_type omp_tile_parts = partitioner.get_partition(part, threads, thread_id);
//                 // PRINT_RT("NORM thread %d partition: ", thread_id);
//                 omp_tile_parts.print("OMP REDUC TILES: ");

//                 // iterate over rows.
//                 size_t rid = omp_tile_parts.offset;
//                 for (size_t i = 0; i < omp_tile_parts.size; ++i, ++rid) {
//                     op(input.data(rid), input.columns(), buffers.data(thread_id));
//                 }

// #ifdef USE_OPENMP
// #pragma omp barrier
//             }
// #endif
//             // now merge in thread safe way
//             for (size_t i = 0; i < omp_get_max_threads(); ++i) {
//                 op(buffers.data(i), input.columns(), output.data());
//             }


//         }

//     public:
//         void operator()(InputType const & input, Op const & _op, OutputType & output) const {
//             this->operator()(input, part1D_type(0, input.rows(), 0), op, output);
//         }
// };



// reduction along a particular DIM_INDEX.
template <typename IN, typename Op, typename OUT, int DIM>
class GlobalReduce;
// possibilities - MAT->V along column, V->S,... 

// row wise reduction in matrix. 
template <typename IT, typename Op, typename OT>
class GlobalReduce<splash::ds::aligned_matrix<IT>, Op, splash::ds::aligned_vector<OT>, DIM_INDEX::ROW> :
    public splash::pattern::Reduce<splash::ds::aligned_matrix<IT>, Op, splash::ds::aligned_vector<OT>, DIM_INDEX::ROW> {

    protected:
	    splash::utils::partitioner1D<PARTITION_EQUAL> partitioner;
        using part1D_type = splash::utils::partition<size_t>;
        int procs;
        int rank;
        using basetype = splash::pattern::Reduce<splash::ds::aligned_matrix<IT>, Op, splash::ds::aligned_vector<OT>, DIM_INDEX::ROW>;

    public:
        using InputType = splash::ds::aligned_matrix<IT>;
        using OutputType = splash::ds::aligned_vector<OT>;

#ifdef USE_MPI
        GlobalReduce(MPI_Comm _comm = MPI_COMM_WORLD) {
            MPI_Comm_size(_comm, &procs);
            MPI_Comm_rank(_comm, &rank);
        }
#else
        GlobalReduce() : procs(1), rank(0) {};
#endif
        GlobalReduce(int const & _procs, int const & _rank) :
            procs(_procs), rank(_rank) {};

        // FULL INPUT
        void operator()(InputType const & input, Op const & _op, OutputType & output) const {
            assert((output.size() == input.rows()) && "Reduce requires output vector size to be same as input row count.");

            // split the input amongst the processors.

            // ---- MPI partitioning.
            part1D_type mpi_tile_parts = partitioner.get_partition(input.rows(), procs, rank );
            
            basetype::operator()(input, mpi_tile_parts, _op, output);

            // ----- NO allgather in place. 
            output.allgather_inplace(mpi_tile_parts);

        }
};


// // row wise reduction in matrix. 
// template <typename IT, typename Op, typename OT>
// class GlobalReduce<splash::ds::aligned_matrix<IT>, Op, splash::ds::aligned_vector<OT>, DIM_INDEX::COLUMN> :
//     public splash::pattern::Reduce<splash::ds::aligned_matrix<IT>, Op, splash::ds::aligned_vector<OT>, DIM_INDEX::COLUMN> {

//     protected:
// 	    splash::utils::partitioner1D<PARTITION_EQUAL> partitioner;
//         using part1D_type = splash::utils::partition<size_t>;
//         int procs;
//         int rank;
//         using basetype = splash::pattern::Reduce<splash::ds::aligned_matrix<IT>, Op, splash::ds::aligned_vector<OT>, DIM_INDEX::COLUMN>;

//     public:
//         using InputType = splash::ds::aligned_matrix<IT>;
//         using OutputType = splash::ds::aligned_vector<OT>;

// #ifdef USE_MPI
//         GlobalReduce(MPI_Comm _comm = MPI_COMM_WORLD) {
//             MPI_Comm_size(_comm, &procs);
//             MPI_Comm_rank(_comm, &rank);
//         }
// #else
//         GlobalReduce() : procs(1), rank(0) {};
// #endif
//         GlobalReduce(int const & _procs, int const & _rank) :
//             procs(_procs), rank(_rank) {};

//         // FULL INPUT
//         void operator()(InputType const & input, Op const & _op, OutputType & output) const {
//             assert((output.size() == input.columns()) && "Reduce requires output vector size to be same as input row count.");

//             // split the input amongst the processors.

//             // ---- MPI partitioning.
//             part1D_type mpi_tile_parts = partitioner.get_partition(input.rows(), procs, rank );
            
//             basetype::operator()(input, mpi_tile_parts, op, output);

//             // ----- all reduce.   TODO:  need to implement output.allreduce.  specifically, custom operator support.
//             output.allreduce();

//         }
// };



// Transform each element individually.  same dimensionality.
template <typename IN, typename Op, typename OUT>
class Transform;

// FIX: [ ] non determinism, and at times segv.
template <typename IT, typename Op, typename OT>
class Transform<splash::ds::aligned_matrix<IT>, Op, splash::ds::aligned_matrix<OT>> {

    protected:
	    splash::utils::partitioner1D<PARTITION_EQUAL> partitioner;
        using part1D_type = splash::utils::partition<size_t>;

    public:
        using InputType = splash::ds::aligned_matrix<IT>;
        using OutputType = splash::ds::aligned_matrix<OT>;

    protected:
        // FULL INPUT.
        void operator()(InputType const & input, part1D_type part, Op const & _op, OutputType & output) const {
            assert((output.rows() == input.rows())  && "Transform requires output and input to have same number of rows.");

            // split the input amongst the processors.

            // ---- parallel compute

#ifdef USE_OPENMP
#pragma omp parallel 
            {
                int threads = omp_get_num_threads();
                int thread_id = omp_get_thread_num();
#else 
                int threads = 1;
                int thread_id = 0;
#endif
                // fprintf(stdout, "make Op copy: thread %d\n", omp_get_thread_num());
                Op op;
                op.copy_parameters(_op);

                // partition the local 2D tiles.  omp_tile_parts.offset is local to this processor.
                // ROOT_PRINT_RT("partitioning info : %lu, %d, %d, max thread %d\n", input.rows(), threads, thread_id, omp_get_max_threads());
                part1D_type omp_tile_parts = partitioner.get_partition(part, threads, thread_id);
                omp_tile_parts.print("OMP TRANSFORM TILES: ");

                // iterate over rows.
                size_t rid = omp_tile_parts.offset;
                for (size_t i = 0; i < omp_tile_parts.size; ++i, ++rid) {
                    op(input.data(rid),  input.columns(), output.data(rid));
                }

#ifdef USE_OPENMP
#pragma omp barrier
            }
#endif
        }
    
    public:
        void operator()(InputType const & input, Op const & _op, OutputType & output) const {
            this->operator()(input, part1D_type(0, input.rows(), 0), _op, output);
        }
};


// Transform each element individually.  same dimensionality.
template <typename IN, typename Op, typename OUT>
class GlobalTransform;

// FIX: [ ] non determinism, and at times segv.
template <typename IT, typename Op, typename OT>
class GlobalTransform<splash::ds::aligned_matrix<IT>, Op, splash::ds::aligned_matrix<OT>> :
 public splash::pattern::Transform<splash::ds::aligned_matrix<IT>, Op, splash::ds::aligned_matrix<OT>> {

    protected:
	    splash::utils::partitioner1D<PARTITION_EQUAL> partitioner;
        using part1D_type = splash::utils::partition<size_t>;
        int procs;
        int rank;
        using basetype = splash::pattern::Transform<splash::ds::aligned_matrix<IT>, Op, splash::ds::aligned_matrix<OT>>;

    public:
        using InputType = splash::ds::aligned_matrix<IT>;
        using OutputType = splash::ds::aligned_matrix<OT>;

#ifdef USE_MPI
        GlobalTransform(MPI_Comm comm = MPI_COMM_WORLD) {
            MPI_Comm_size(comm, &procs);
            MPI_Comm_rank(comm, &rank);
        }
#else
        GlobalTransform() : procs(1), rank(0) {};
#endif
        GlobalTransform(int const & _procs, int const & _rank) :
            procs(_procs), rank(_rank) {};

        // FULL INPUT.
        void operator()(InputType const & input, Op const & _op, OutputType & output) const {
            assert((output.rows() == input.rows())  && "Transform requires output and input to have same number of rows.");

            // split the input amongst the processors.

            // ---- MPI partitioning.  row by row.
            part1D_type mpi_tile_parts = partitioner.get_partition(input.rows(), procs, rank );

            // ---- parallel compute
            basetype::operator()(input, mpi_tile_parts, _op, output);

            // ----- allgather in place. 
            output.allgather_inplace(mpi_tile_parts);
        }
};



// reduction and transform
template <typename IN, typename Reduc, typename Op, typename OUT>
class ReduceTransform;

template <typename IT, typename Reduc, typename Op, typename OT>
class ReduceTransform<splash::ds::aligned_matrix<IT>, Reduc, Op, splash::ds::aligned_matrix<OT>> {
    protected:
	    splash::utils::partitioner1D<PARTITION_EQUAL> partitioner;
        using part1D_type = splash::utils::partition<size_t>;
        int procs;
        int rank;
        using MT = typename Reduc::OutputType;
        mutable splash::ds::aligned_vector<MT> __buffer;
        // using ROW_REDUC = splash::pattern::GlobalReduce<splash::ds::aligned_matrix<IT>, Reduc, splash::ds::aligned_vector<MT>, DIM_INDEX::ROW>;
        // using COL_REDUC = splash::pattern::GlobalReduce<splash::ds::aligned_matrix<IT>, Reduc, splash::ds::aligned_vector<MT>, DIM_INDEX::COLUMN>;
        

    public:
        using InputType = splash::ds::aligned_matrix<IT>;
        using OutputType = splash::ds::aligned_matrix<OT>;

#ifdef USE_MPI
        ReduceTransform(MPI_Comm comm = MPI_COMM_WORLD) {
            MPI_Comm_size(comm, &procs);
            MPI_Comm_rank(comm, &rank);
        }
#else
        ReduceTransform() : procs(1), rank(0) {};
#endif
        ReduceTransform(int const & _procs, int const & _rank) :
            procs(_procs), rank(_rank) {};

        // assume input is symmetric.
        // FULL INPUT
        void operator()(InputType const & input, 
            Reduc const & _reduc, Op const & _op, 
            OutputType & output) const {
            assert((output.rows() == input.rows())  && "Transform requires output and input to have same number of rows.");
            assert((output.columns() == input.columns())  && "Transform requires output and input to have same number of columns.");

            // for now, require symmetry.
            assert((input.rows() == input.columns())  && "Transform requires output and input to be symmetric.");


            // first perform a row-wise reduction
            auto stime = getSysTime();
            __buffer.resize(input.rows());

            // split the input amongst the processors.

            // ---- MPI partitioning.
            part1D_type mpi_tile_parts = partitioner.get_partition(input.rows(), procs, rank );

            // ---- parallel compute

#ifdef USE_OPENMP
#pragma omp parallel 
            {
                int threads = omp_get_num_threads();
                int thread_id = omp_get_thread_num();
#else 
                int threads = 1;
                int thread_id = 0;
#endif
                // fprintf(stdout, "make Op copy: thread %d\n", omp_get_thread_num());
                Reduc reduc;
                reduc.copy_parameters(_reduc);

                // partition the local 2D tiles.  omp_tile_parts.offset is local to this processor.
                // ROOT_PRINT_RT("partitioning info : %lu, %d, %d\n", output.size(), threads, thread_id);
                part1D_type omp_tile_parts = partitioner.get_partition(mpi_tile_parts, threads, thread_id);
                // PRINT_RT("NORM thread %d partition: ", thread_id);
                omp_tile_parts.print("OMP REDUC TILES: ");

                // iterate over rows.
                size_t rid = omp_tile_parts.offset;
                for (size_t i = 0; i < omp_tile_parts.size; ++i, ++rid) {
                    __buffer[rid] = reduc(input.data(rid), input.columns());
                }
#ifdef USE_OPENMP
#pragma omp barrier
            }
#endif

            // MPI to allgather results.
            __buffer.allgather_inplace(mpi_tile_parts);
            
            auto etime = getSysTime();
            ROOT_PRINT("ReduceTransform REDUCE phase in %f sec\n", get_duration_s(stime, etime));

            // ------------------ processing. reach element needs row and col
            // use the same partitioning
            output.resize(input.rows(), input.columns());

            // now do the transform using the intermediate results.
#ifdef USE_OPENMP
#pragma omp parallel 
            {
                int threads = omp_get_num_threads();
                int thread_id = omp_get_thread_num();
#else 
                int threads = 1;
                int thread_id = 0;
#endif
                // fprintf(stdout, "make Op copy: thread %d\n", omp_get_thread_num());
                Op op;
                op.copy_parameters(_op);

                // partition the local 2D tiles.  omp_tile_parts.offset is local to this processor.
                // ROOT_PRINT_RT("partitioning info : %lu, %d, %d, max thread %d\n", input.rows(), threads, thread_id, omp_get_max_threads());
                part1D_type omp_tile_parts = partitioner.get_partition(mpi_tile_parts, threads, thread_id);
                omp_tile_parts.print("OMP TRANSFORM TILES: ");

                // iterate over rows.
                size_t rid = omp_tile_parts.offset;
                for (size_t i = 0; i < omp_tile_parts.size; ++i, ++rid) {
                    for (unsigned int j = 0; j < input.columns(); ++j) {
                        output(rid, j) = op(input(rid, j), __buffer[rid], __buffer[j]);
                    }
                }

#ifdef USE_OPENMP
#pragma omp barrier
            }
#endif
            // ----- allgather in place. 
            output.allgather_inplace(mpi_tile_parts);
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


        // FULL INPUT, PARTITIONED OUTPUT.
        void operator()(InputType const & input1, InputType const & input2, Op const & _op, OutputType & output) const {
            if ((output.rows() != input1.rows()) || (output.columns() != input2.rows())) 
                PRINT_RT("InnerProduct: input1 rows:  %lu, input2 rows: %lu, output rows: %lu, columns %lu\n",
                    input1.rows(), input2.rows(), output.rows(), output.columns());
            assert(((output.rows() == input1.rows()) && (output.columns() == input2.rows())) && "InnerProduct requires output rows and input1 rows to be same, and output columns and input2 rows to be same.");

            PRINT_RT("InnerProduct: input1 %lu X %lu, input2 %lu X %lu, output %lu X %lu\n",
                input1.rows(), input1.columns(), input2.rows(), input2.columns(), output.rows(), output.columns());

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
            // input1.print("NORMED: ");

            // OpenMP stuff.
#ifdef USE_OPENMP
#pragma omp parallel
            {
                int threads = omp_get_num_threads();
                int thread_id = omp_get_thread_num();
#else 
                int threads = 1;
                int thread_id = 0;
#endif
                // Op op(_op);
                Op op;
                op.copy_parameters(_op);

                // partition the local 2D tiles.  omp_tile_parts.offset is local to this processor.
                // ROOT_PRINT_RT("partitioning info : %lu, %d, %d\n", mpi_tile_parts.size, threads, thread_id);
		        part1D_type omp_tile_parts = partitioner.get_partition(mpi_tile_parts.size, threads, thread_id);
		        // PRINT_RT("thread %d partition: ", thread_id);
		        omp_tile_parts.print("OMP PARTITION: ");

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

                    // if this tile is on the diagonal, do 1/2 comput.
                    // if it is not on the diagonal, then do full compute.

                    if (part.r.offset != part.c.offset) {
                        // off major diagonal.

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
                                // auto xy = op(input1.data(row), input2.data(col), input1.columns());
                                // auto yx = op(input2.data(col), input1.data(row), input2.columns());
                                // if (xy != yx)  printf("ERROR: distcorr not symmetric at row col (%lu, %lu), xy: %.18lf, yx %.18lf\n", row, col, xy, yx);
                                *data = op(input1.data(row), input2.data(col), input1.columns());
                                ++data;
                            }
                        }
                    } else {
                        // on major diagonal
                        // work on 1 tile
                        row = part.r.offset;
                        row_end = row + part.r.size;
                        auto data2c = data;   // transposed position.  addressed by data2[col, row]
                        auto data2 = data;

                        // rmin = std::min(rmin, row);
                        // cmin = std::min(cmin, part.c.offset);
                        // rmax = std::max(rmax, row_end);
                        // cmax = std::max(cmax, part.c.offset + part.c.size);

                        // work on 1 row
                        for (; row < row_end; ++row) {
                            // for row in tile
                            col = part.c.offset;
                            col_end = col + part.c.size;
                            data2 = data2c;

                            // work on 1 column
                            for (; col < col_end; ++col) {
                                // within a tile on the diagonal, skip compute if lower half.
                                if (row == col) {
                                    *data = 1.0;
                                } else if (row < col) {
                                    // upper.  so fill in.
                                    auto xy = op(input1.data(row), input2.data(col), input1.columns());
                                    *data = xy; 
                                    *data2 = xy;
                                }  // else lower half.  skip.

                                // compute correlation
                                // auto xy = op(input1.data(row), input2.data(col), input1.columns());
                                // auto yx = op(input2.data(col), input1.data(row), input2.columns());
                                // if (xy != yx)  printf("ERROR: distcorr not symmetric at row col (%lu, %lu), xy: %.18lf, yx %.18lf\n", row, col, xy, yx);
                                ++data;  // advance 1 col
                                data2 += part.c.size;  // advance 1 row
                            }
                            ++data2c;  // next column
                        }
                    }
                }
        		// PRINT_RT("CORR BOUNDS r: %lu %lu, c %lu %lu\n", rmin, rmax-rmin, cmin, cmax-cmin);

#ifdef USE_OPENMP
#pragma omp barrier
        	}
#endif
        	etime = getSysTime();
	        ROOT_PRINT("Computed in %f sec\n", get_duration_s(stime, etime));


        	// ============= repartition output
	        stime = getSysTime();

            tiles.get_bounds().print("TILES BOUNDS: ");
            PRINT_RT("TILES COUNT: %lu\n", tiles.size());
            // tiles.print("TILES: ");

        	part1D_type row_part = partitioner.get_partition(input1.rows(), this->procs, this->rank);
            row_part.print("ROW PARTITIONS: ");
            // // with transpose.
            // tiles_type transposed = tiles.transpose();
            // tiles_type all_tiles = tiles.merge(transposed);
            tiles_type all_tiles = tiles.reflect_diagonally();

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


}}