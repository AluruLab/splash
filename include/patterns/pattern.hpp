#pragma once

#include "utils/partition.hpp"
#include "ds/aligned_tiles.hpp"
#include "ds/aligned_vector.hpp"
#include "ds/aligned_matrix.hpp"

#include <vector>
#include <cassert>

#include "utils/report.hpp"

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

// SFINAE test
template <typename T, typename... Args>
class has_operator
{
    template <typename C,
            typename = decltype( std::declval<C>().operator()(std::declval<Args>()...) )>
    static std::true_type test(int);
    template <typename C>
    static std::false_type test(...);

public:
    static constexpr bool value = decltype(test<T>(0))::value;
};

class OpBase {
    public:
        mutable size_t processed;
        OpBase() : processed(0) {};
        virtual ~OpBase() {};
};


enum DIM_INDEX : int { ROW = 1, COLUMN = 2 };


// reduction along a particular DIM_INDEX.
template <typename IN, typename Op, typename OUT, int DIM>
class Reduce;
// possibilities - MAT->V along column, V->S,... 

// row wise reduction in matrix. 
template <typename IT, typename Op, typename OT>
class Reduce<splash::ds::aligned_matrix<IT>, Op, splash::ds::aligned_vector<OT>, DIM_INDEX::ROW> :
    public OpBase {

    protected:
	    splash::utils::partitioner1D<PARTITION_EQUAL> partitioner;
        using part1D_type = splash::utils::partition<size_t>;

        template <typename OO, typename I, typename std::enable_if<splash::pattern::has_operator<OO, size_t const &, I const *, size_t const &>::value, int>::type = 1>
        inline OT run(OO const & op, size_t const & r, I const * row, size_t const & count) const {
            return op(r, row, count);
        }
	
        template <typename OO, typename I, typename std::enable_if<!splash::pattern::has_operator<OO, size_t const &, I const *, size_t const &>::value, int>::type = 1>
        inline OT run(OO const & op, size_t const & r, I const * row, size_t const & count) const {
            return op(row, count);
        }


    public:
        using InputType = splash::ds::aligned_matrix<IT>;
        using OutputType = splash::ds::aligned_vector<OT>;

    protected:
        // same input and output rows.
        void operator()(InputType const & input, part1D_type part, Op const & _op, OutputType & output) const {
            assert((output.size() == input.rows()) && "Reduce requires output vector size to be same as input row count.");

            // split the input amongst the processors.

            // ---- parallel compute
            size_t count = 0;
#ifdef USE_OPENMP
#pragma omp parallel reduction(+: count)
            {
                int threads = omp_get_num_threads();
                int thread_id = omp_get_thread_num();
#else 
            {
                int threads = 1;
                int thread_id = 0;
#endif
                Op op;
                op.copy_parameters(_op);

                // partition the local 2D tiles.  omp_tile_parts.offset is local to this processor.
                // FMT_ROOT_PRINT_RT("partitioning info : {}, {}, {}\n", output.size(), threads, thread_id);
                part1D_type omp_tile_parts = partitioner.get_partition(part, threads, thread_id);
                // FMT_PRINT_RT("NORM thread {} partition: ", thread_id);
                omp_tile_parts.print("OMP REDUC TILES: ");

                // iterate over rows.
                size_t rid = omp_tile_parts.offset;
                for (size_t i = 0; i < omp_tile_parts.size; ++i, ++rid) {
                    output[rid] = run(op, rid, input.data(rid),  input.columns());
                }

                count += op.processed;
            }
            this->processed = count;
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
//              {
//                 int threads = 1;
//                 int thread_id = 0;
// #endif
//                 Op op;
//                 op.copy_parameters(_op);

//                 // partition the local 2D tiles.  omp_tile_parts.offset is local to this processor.
//                 // FMT_ROOT_PRINT_RT("partitioning info : {}, {}, {}\n", output.size(), threads, thread_id);
//                 part1D_type omp_tile_parts = partitioner.get_partition(part, threads, thread_id);
//                 // FMT_PRINT_RT("NORM thread {} partition: ", thread_id);
//                 omp_tile_parts.print("OMP REDUC TILES: ");

//                 // iterate over rows.
//                 size_t rid = omp_tile_parts.offset;
//                 for (size_t i = 0; i < omp_tile_parts.size; ++i, ++rid) {
//                     op(input.data(rid), input.columns(), buffers.data(thread_id));
//                 }

//             }
//
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
            // output.allgather_inplace(mpi_tile_parts);

            // allreduce
            splash::utils::mpi::datatype<size_t> dt;
            MPI_Allreduce(MPI_IN_PLACE, &(this->processed), 1, dt.value, MPI_SUM, MPI_COMM_WORLD );
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
class Transform<splash::ds::aligned_matrix<IT>, Op, splash::ds::aligned_matrix<OT>>  :
    public OpBase {

    protected:
	    splash::utils::partitioner1D<PARTITION_EQUAL> partitioner;
        using part1D_type = splash::utils::partition<size_t>;

    public:
        using InputType = splash::ds::aligned_matrix<IT>;
        using OutputType = splash::ds::aligned_matrix<OT>;

    protected:
        // FULL INPUT.
        void operator()(InputType const & input, part1D_type const & part, 
            Op const & _op, OutputType & output, part1D_type const & out_part) const {
            assert((out_part.size == part.size)  && "Transform requires output and input to have same number of rows.");

            // split the input amongst the processors.

            // ---- parallel compute
            size_t count = 0;
#ifdef USE_OPENMP
#pragma omp parallel reduction(+: count)
            {
                int threads = omp_get_num_threads();
                int thread_id = omp_get_thread_num();
#else 
            {
                int threads = 1;
                int thread_id = 0;
#endif
                // FMT_PRINT_RT("make Op copy: thread {}\n", omp_get_thread_num());
                Op op;
                op.copy_parameters(_op);

                // partition the local 2D tiles.  omp_tile_parts.offset is local to this processor.
                // FMT_ROOT_PRINT_RT("partitioning info : {}, {}, {}, max thread {}\n", input.rows(), threads, thread_id, omp_get_max_threads());
                part1D_type omp_tile_parts = partitioner.get_partition(part, threads, thread_id);
                part1D_type omp_out_part = partitioner.get_partition(out_part, threads, thread_id);
                // omp_tile_parts.print("OMP TRANSFORM TILES: ");

                // iterate over rows.
                size_t rid = omp_tile_parts.offset;
                size_t max_rid = rid + omp_tile_parts.size;
                size_t orid = omp_out_part.offset;
                for (; rid < max_rid; ++rid, ++orid) {
                    // DIRECTLY INCREMENT POINTER does not change time.
                    op(input.data(rid), input.columns(), output.data(orid));
                }

                count += op.processed;
            }
            this->processed = count;
        }
    
    public:
        void operator()(InputType const & input, Op const & _op, OutputType & output) const {
            part1D_type rows = part1D_type(0, input.rows(), 0);
            this->operator()(input, rows, _op, output, rows);
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

            // split the input amongst the processors.

            // ---- MPI partitioning.  row by row.
            part1D_type mpi_tile_parts = partitioner.get_partition(input.rows(), procs, rank );

            // ---- parallel compute
            output.resize(mpi_tile_parts.size, input.columns());
            basetype::operator()(input, mpi_tile_parts, _op, output, part1D_type(0, mpi_tile_parts.size, 0));

            // // ----- allgather in place. 
            // output.allgather_inplace(mpi_tile_parts);

            // allreduce
            splash::utils::mpi::datatype<size_t> dt;
            MPI_Allreduce(MPI_IN_PLACE, &(this->processed), 1, dt.value, MPI_SUM, MPI_COMM_WORLD );
        }
};




// Transform each element individually.  same dimensionality.
template <typename IN, typename IN2, typename Op, typename OUT>
class BinaryOp;

// this is COMPLETELY LOCAL
template <typename IT, typename IT2, typename Op, typename OT>
class BinaryOp<splash::ds::aligned_matrix<IT>, splash::ds::aligned_matrix<IT2>, Op, splash::ds::aligned_matrix<OT>> :
    public OpBase {

    protected:
	    splash::utils::partitioner1D<PARTITION_EQUAL> partitioner;
        using part1D_type = splash::utils::partition<size_t>;

    public:
        using InputType = splash::ds::aligned_matrix<IT>;
        using InputType2 = splash::ds::aligned_matrix<IT2>;
        using OutputType = splash::ds::aligned_matrix<OT>;

        // process specified region.  NOTE: MUST RESIZE OUTPUT OUTSIDE OF THIS CALL
        void operator()(InputType const & input, part1D_type const & in_part, 
            InputType2 const & input2, part1D_type const & in2_part,
            Op const & _op, OutputType & output, part1D_type const & out_part) const {
            assert((out_part.size == in_part.size) && (out_part.size == in2_part.size) && "Transform requires output and input to have same number of rows.");

            // ---- parallel compute
            size_t count = 0;
#ifdef USE_OPENMP
#pragma omp parallel reduction(+: count)
            {
                int threads = omp_get_num_threads();
                int thread_id = omp_get_thread_num();
#else 
            {
                int threads = 1;
                int thread_id = 0;
#endif
                // FMT_PRINT_RT("make Op copy: thread {}\n", omp_get_thread_num());
                Op op;
                op.copy_parameters(_op);

                // partition the local 2D tiles.  omp_tile_parts.offset is local to this processor.
                // FMT_ROOT_PRINT_RT("partitioning info : {}, {}, {}, max thread {}\n", input.rows(), threads, thread_id, omp_get_max_threads());
                part1D_type omp_in_part = partitioner.get_partition(in_part, threads, thread_id);
                part1D_type omp_in2_part = partitioner.get_partition(in2_part, threads, thread_id);
                part1D_type omp_out_part = partitioner.get_partition(out_part, threads, thread_id);
                // omp_tile_parts.print("OMP BINARY_OP TILES: ");

                // iterate over rows.
                size_t rid = omp_in_part.offset;
                size_t rid2 = omp_in2_part.offset;
                size_t orid = omp_out_part.offset;
                for (size_t i = 0; i < omp_out_part.size; ++i, ++rid, ++rid2, ++orid) {
                    op(input.data(rid), input2.data(rid2), input.columns(), output.data(orid));
                }

                count += op.processed;
            }
            this->processed = count;
        }

        void operator()(InputType const & input, InputType2 const & input2, Op const & _op, OutputType & output) const {
            part1D_type rows(0, input.rows(), 0);
            output.resize(input.rows(), input.columns());
            this->operator()(input, rows, input2, rows, _op, output, rows);
        }
};


// Transform each element individually.  input and output output distriibuted.
template <typename IN, typename IN2, typename Op, typename OUT>
class GlobalBinaryOp;

template <typename IT, typename IT2, typename Op, typename OT>
class GlobalBinaryOp<splash::ds::aligned_matrix<IT>, splash::ds::aligned_matrix<IT2>, Op, splash::ds::aligned_matrix<OT>> :
 public splash::pattern::BinaryOp<splash::ds::aligned_matrix<IT>, splash::ds::aligned_matrix<IT2>, Op, splash::ds::aligned_matrix<OT>> {

    protected:
	    splash::utils::partitioner1D<PARTITION_EQUAL> partitioner;
        using part1D_type = splash::utils::partition<size_t>;
        int procs;
        int rank;
        using basetype = splash::pattern::BinaryOp<splash::ds::aligned_matrix<IT>, splash::ds::aligned_matrix<IT2>,
                                                    Op, splash::ds::aligned_matrix<OT>>;

    public:
        using InputType = splash::ds::aligned_matrix<IT>;
        using InputType2 = splash::ds::aligned_matrix<IT2>;
        using OutputType = splash::ds::aligned_matrix<OT>;

#ifdef USE_MPI
        GlobalBinaryOp(MPI_Comm comm = MPI_COMM_WORLD) {
            MPI_Comm_size(comm, &procs);
            MPI_Comm_rank(comm, &rank);
        }
#else
        GlobalBinaryOp() : procs(1), rank(0) {};
#endif
        GlobalBinaryOp(int const & _procs, int const & _rank) :
            procs(_procs), rank(_rank) {};

        // FULL INPUT.
        void operator()(InputType const & input, InputType2 const & input2, Op const & _op, OutputType & output) const {

            // split the input amongst the processors.

            // ---- MPI partitioning.  row by row.
            part1D_type input_rows = partitioner.get_partition(input.rows(), procs, rank );

            // ---- parallel compute
            output.resize(input_rows.size, input.columns());
            basetype::operator()(input, input_rows, input2, input_rows, _op, output, part1D_type(0, input_rows.size, 0));

            // // ----- allgather in place. 
            // output.allgather_inplace(mpi_tile_parts);

            // allreduce
            splash::utils::mpi::datatype<size_t> dt;
            MPI_Allreduce(MPI_IN_PLACE, &(this->processed), 1, dt.value, MPI_SUM, MPI_COMM_WORLD );
        }
};





// reduction and transform
template <typename IN, typename Reduc, typename Op, typename OUT>
class ReduceTransform;

// input and output are distributed.  ONLY FOR SYMMETRIC MATRIX.
template <typename IT, typename Reduc, typename Op, typename OT>
class ReduceTransform<splash::ds::aligned_matrix<IT>, Reduc, Op, splash::ds::aligned_matrix<OT>> :
    public OpBase {
    protected:
	    splash::utils::partitioner1D<PARTITION_EQUAL> partitioner;
        using part1D_type = splash::utils::partition<size_t>;
        int procs;
        int rank;
        using MT = typename Reduc::OutputType;
        mutable splash::ds::aligned_vector<MT> __buffer;
        // using ROW_REDUC = splash::pattern::GlobalReduce<splash::ds::aligned_matrix<IT>, Reduc, splash::ds::aligned_vector<MT>, DIM_INDEX::ROW>;
        // using COL_REDUC = splash::pattern::GlobalReduce<splash::ds::aligned_matrix<IT>, Reduc, splash::ds::aligned_vector<MT>, DIM_INDEX::COLUMN>;
        
        template <typename OO, typename I, typename std::enable_if<splash::pattern::has_operator<OO, size_t const &, I const *, size_t const &>::value, int>::type = 1>
        inline MT reduc_run(OO const & op, size_t const & r, I const * row, size_t const & count) const {
            return op(r, row, count);
        }
	
        template <typename OO, typename I, typename std::enable_if<!splash::pattern::has_operator<OO, size_t const &, I const *, size_t const &>::value, int>::type = 1>
        inline MT reduc_run(OO const & op, size_t const & r, I const * row, size_t const & count) const {
            return op(row, count);
        }


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
            // assert((total_rows == input.columns())  && "Transform requires output and input to be symmetric.");            

            // first perform a row-wise reduction
            auto stime = getSysTime();

            // get sizes and dimensions
            size_t local_rows = input.rows();

            // THIS IMPLEMENTATION IS ONLY FOR SYMMETRIC MATRICES.
            splash::ds::aligned_vector<MT> buf(input.rows());

            // split the input amongst the processors.

            // ---- parallel compute
#ifdef USE_OPENMP
#pragma omp parallel 
            {
                int threads = omp_get_num_threads();
                int thread_id = omp_get_thread_num();
#else 
            {
                int threads = 1;
                int thread_id = 0;
#endif
                // FMT_PRINT_RT("make Op copy: thread {}\n", omp_get_thread_num());
                Reduc reduc;
                reduc.copy_parameters(_reduc);

                // partition the local 2D tiles.  omp_tile_parts.offset is local to this processor.
                // FMT_ROOT_PRINT_RT("partitioning info : {}, {}, {}\n", output.size(), threads, thread_id);
                part1D_type omp_tile_parts = partitioner.get_partition(input.rows(), threads, thread_id);
                // FMT_PRINT_RT("NORM thread {} partition: ", thread_id);
                // omp_tile_parts.print("OMP REDUC TILES: ");

                // iterate over rows.
                size_t rid = omp_tile_parts.offset;
                for (size_t i = 0; i < omp_tile_parts.size; ++i, ++rid) {
                    buf[rid] = reduc_run(reduc, rid, input.data(rid), input.columns());
                }
            }

            // MPI to allgather results.
            __buffer = buf.allgather();  // smallish vector, performance okay.
            
            auto etime = getSysTime();
            FMT_PRINT_RT("ReduceTransform REDUCE phase in {} sec\n", get_duration_s(stime, etime));
            FMT_ROOT_PRINT("ReduceTransform REDUCE phase in {} sec\n", get_duration_s(stime, etime));

            // ------------------ processing. reach element needs row and col
            // use the same partitioning
            output.resize(input.rows(), input.columns());

            // now do the transform using the intermediate results.
            size_t count = 0;
#ifdef USE_OPENMP
#pragma omp parallel reduction(+:count)
            {
                int threads = omp_get_num_threads();
                int thread_id = omp_get_thread_num();
#else 
            {
                int threads = 1;
                int thread_id = 0;
#endif
                // FMT_PRINT_RT("make Op copy: thread {}\n", omp_get_thread_num());
                Op op;
                op.copy_parameters(_op);

                // partition the local 2D tiles.  omp_tile_parts.offset is local to this processor.
                // FMT_ROOT_PRINT_RT("partitioning info : {}, {}, {}, max thread {}\n", input.rows(), threads, thread_id, omp_get_max_threads());
                part1D_type omp_tile_parts = partitioner.get_partition(input.rows(), threads, thread_id);
                // omp_tile_parts.print("OMP TRANSFORM TILES: ");

                // iterate over rows.
                size_t rid = omp_tile_parts.offset;
                for (size_t i = 0; i < omp_tile_parts.size; ++i, ++rid) {
                    op(input.data(rid), buf.data(rid), __buffer.data(), input.columns(), output.data(rid));
                    // for (unsigned int j = 0; j < input.columns(); ++j) {
                    //     output(rid - offset, j) = op(input(rid, j), __buffer[rid], __buffer[j]);
                    // }
                }

                count += op.processed;
            }
            this->processed = count;

            // ----- allgather in place. 
            // output.allgather_inplace(mpi_tile_parts);

            // allreduce
            splash::utils::mpi::datatype<size_t> dt;
            MPI_Allreduce(MPI_IN_PLACE, &(this->processed), 1, dt.value, MPI_SUM, MPI_COMM_WORLD );
        }

};



// reduction and transform
template <typename IN, typename Reduc, typename Op, typename OUT>
class GlobalReduceTransform;

// output is distributed.    ONLY FOR SYMMETRIC MATRIX.
template <typename IT, typename Reduc, typename Op, typename OT>
class GlobalReduceTransform<splash::ds::aligned_matrix<IT>, Reduc, Op, splash::ds::aligned_matrix<OT>> :
    public OpBase {
    protected:
	    splash::utils::partitioner1D<PARTITION_EQUAL> partitioner;
        using part1D_type = splash::utils::partition<size_t>;
        int procs;
        int rank;
        using MT = typename Reduc::OutputType;
        mutable splash::ds::aligned_vector<MT> __buffer;
        // using ROW_REDUC = splash::pattern::GlobalReduce<splash::ds::aligned_matrix<IT>, Reduc, splash::ds::aligned_vector<MT>, DIM_INDEX::ROW>;
        // using COL_REDUC = splash::pattern::GlobalReduce<splash::ds::aligned_matrix<IT>, Reduc, splash::ds::aligned_vector<MT>, DIM_INDEX::COLUMN>;
        
        template <typename OO, typename I, typename std::enable_if<splash::pattern::has_operator<OO, size_t const &, I const *, size_t const &>::value, int>::type = 1>
        inline MT reduc_run(OO const & op, size_t const & r, I const * row, size_t const & count) const {
            return op(r, row, count);
        }
	
        template <typename OO, typename I, typename std::enable_if<!splash::pattern::has_operator<OO, size_t const &, I const *, size_t const &>::value, int>::type = 1>
        inline MT reduc_run(OO const & op, size_t const & r, I const * row, size_t const & count) const {
            return op(row, count);
        }


    public:
        using InputType = splash::ds::aligned_matrix<IT>;
        using OutputType = splash::ds::aligned_matrix<OT>;

#ifdef USE_MPI
        GlobalReduceTransform(MPI_Comm comm = MPI_COMM_WORLD) {
            MPI_Comm_size(comm, &procs);
            MPI_Comm_rank(comm, &rank);
        }
#else
        GlobalReduceTransform() : procs(1), rank(0) {};
#endif
        GlobalReduceTransform(int const & _procs, int const & _rank) :
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
            {
                int threads = 1;
                int thread_id = 0;
#endif
                // FMT_PRINT_RT("make Op copy: thread {}\n", omp_get_thread_num());
                Reduc reduc;
                reduc.copy_parameters(_reduc);

                // partition the local 2D tiles.  omp_tile_parts.offset is local to this processor.
                // FMT_ROOT_PRINT_RT("partitioning info : {}, {}, {}\n", output.size(), threads, thread_id);
                part1D_type omp_tile_parts = partitioner.get_partition(mpi_tile_parts, threads, thread_id);
                // FMT_PRINT_RT("NORM thread {} partition: ", thread_id);
                // omp_tile_parts.print("OMP REDUC TILES: ");

                // iterate over rows.
                size_t rid = omp_tile_parts.offset;
                for (size_t i = 0; i < omp_tile_parts.size; ++i, ++rid) {
                    __buffer[rid] = reduc_run(reduc, rid, input.data(rid), input.columns());
                }
            }

            // MPI to allgather results.
            __buffer.allgather_inplace(mpi_tile_parts);  // smallish vector, performance okay.
            
            auto etime = getSysTime();
            FMT_PRINT_RT("GloblaReduceTransform REDUCE phase in {} sec\n", get_duration_s(stime, etime));
            FMT_ROOT_PRINT("GlobalReduceTransform REDUCE phase in {} sec\n", get_duration_s(stime, etime));

            // ------------------ processing. reach element needs row and col
            // use the same partitioning
            output.resize(mpi_tile_parts.size, input.columns());

            // now do the transform using the intermediate results.
            size_t count = 0;
#ifdef USE_OPENMP
#pragma omp parallel reduction(+:count)
            {
                int threads = omp_get_num_threads();
                int thread_id = omp_get_thread_num();
#else 
            {
                int threads = 1;
                int thread_id = 0;
#endif
                // FMT_PRINT_RT("make Op copy: thread {}\n", omp_get_thread_num());
                Op op;
                op.copy_parameters(_op);

                // partition the local 2D tiles.  omp_tile_parts.offset is local to this processor.
                // FMT_ROOT_PRINT_RT("partitioning info : {}, {}, {}, max thread {}\n", input.rows(), threads, thread_id, omp_get_max_threads());
                part1D_type omp_tile_parts = partitioner.get_partition(mpi_tile_parts, threads, thread_id);
                // omp_tile_parts.print("OMP TRANSFORM TILES: ");

                // iterate over rows.
                size_t rid = omp_tile_parts.offset;
                size_t offset = mpi_tile_parts.offset;
                for (size_t i = 0; i < omp_tile_parts.size; ++i, ++rid) {
                    op(input.data(rid), __buffer.data(rid), __buffer.data(), input.columns(), output.data(rid - offset));
                    // for (unsigned int j = 0; j < input.columns(); ++j) {
                    //     output(rid - offset, j) = op(input(rid, j), __buffer[rid], __buffer[j]);
                    // }
                }

                count += op.processed;
            }
            this->processed = count;

            // ----- allgather in place. 
            // output.allgather_inplace(mpi_tile_parts);

            // allreduce
            splash::utils::mpi::datatype<size_t> dt;
            MPI_Allreduce(MPI_IN_PLACE, &(this->processed), 1, dt.value, MPI_SUM, MPI_COMM_WORLD );
        }

};




// Multiply pattern, i.e. M1 * M2' -> M3
template <typename IN, typename Op, typename OUT, bool SYMMETRIC = true>
class InnerProduct;

template <typename IT, typename Op, typename OT, bool SYMMETRIC>
class InnerProduct<splash::ds::aligned_matrix<IT>, Op, splash::ds::aligned_tiles<OT, splash::utils::partition2D<size_t>>, SYMMETRIC> :
    public OpBase {
    public:
        using InputType = splash::ds::aligned_matrix<IT>;
        using OutputType = splash::ds::aligned_tiles<OT, splash::utils::partition2D<size_t>>;

    protected:
        splash::utils::partitioner2D<PARTITION_FIXED> partitioner2d;
        splash::utils::partitioner1D<PARTITION_EQUAL> partitioner;
	    splash::utils::banded_diagonal_filter part_filter;
	    // splash::utils::upper_triangle_filter part_filter;
	
        using part1D_type = splash::utils::partition<size_t>;
        using part2D_type = splash::utils::partition2D<size_t>;
        using tiles_type = OutputType;

        int procs;
        int rank;
        
        template <typename OO, typename I, typename std::enable_if<splash::pattern::has_operator<OO, size_t const &, size_t const &, I const *, I const *, size_t const &>::value, int>::type = 1>
        inline OT run(OO const & op, size_t const & r, size_t const & c, I const * row, I const * col, size_t const & count) const {
            return op(r, c, row, col, count);
        }
	
        template <typename OO, typename I, typename std::enable_if<!splash::pattern::has_operator<OO, size_t const &, size_t const &, I const *, I const *, size_t const &>::value, int>::type = 1>
        inline OT run(OO const & op, size_t const & r, size_t const & c, I const * row, I const * col, size_t const & count) const {
            return op(row, col, count);
        }

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
        OutputType operator()(InputType const & input1, InputType const & input2, Op const & _op) const {
            FMT_ROOT_PRINT("InnerProduct: input1 {} X {}, input2 {} X {} -> {} x {}\n",
                input1.rows(), input1.columns(), input2.rows(), input2.columns(), input1.rows(), input2.rows());

            // ---- fixed-size partiton input and filter for tiles t
            auto stime = getSysTime();
            std::vector<part2D_type> all_tile_parts = partitioner2d.divide(input1.rows(), input2.rows(), 
                    static_cast<typename OutputType::size_type>(PARTITION_TILE_DIM), 
                    static_cast<typename OutputType::size_type>(PARTITION_TILE_DIM) );
            std::vector<part2D_type> tile_parts;
            if (SYMMETRIC)
                tile_parts = part_filter.filter( all_tile_parts );
            else
                tile_parts = std::move(all_tile_parts);

            // FMT_PRINT_RT("Partitions: 2D {} -> filtered {}\n", all_tile_parts.size(), tile_parts.size());

            // ---- partition the partitions for MPI
            part1D_type mpi_tile_parts = partitioner.get_partition(tile_parts.size(), this->procs, this->rank);
            // FMT_PRINT_RT("MPI Rank {} partition: ", this->rank);
            // mpi_tile_parts.print("MPI TILEs: ");

            auto etime = getSysTime();
            FMT_ROOT_PRINT("Correlation Partitioned in {} sec\n", get_duration_s(stime, etime));

            // ---- compute correlation
            stime = getSysTime();

        	// ---- set up the temporary output, tiled, contains the partitions to process.
	        // FMT_PRINT_RT("[pearson TILES] ");
	        OutputType output(tile_parts.data() + mpi_tile_parts.offset, mpi_tile_parts.size);
            // input1.print("NORMED: ");

            std::vector<Op> ops(omp_get_max_threads());
#ifdef USE_OPENMP
#pragma omp parallel
            {
                int thread_id = omp_get_thread_num();
#else 
            {
                int thread_id = 0;
#endif
                ops[thread_id].copy_parameters(_op);
            }
            
            this->operator()(input1, input2, ops, output);

            // allreduce
            splash::utils::mpi::datatype<size_t> dt;
            MPI_Allreduce(MPI_IN_PLACE, &(this->processed), 1, dt.value, MPI_SUM, MPI_COMM_WORLD );

        	etime = getSysTime();
	        FMT_ROOT_PRINT("Computed in {} sec\n", get_duration_s(stime, etime));
            return output;
        }

        // compute for only tiles specified in the output.
        void operator()(InputType const & input1, InputType const & input2, std::vector<Op> const & _ops, OutputType & tiles) const {
            
            // OpenMP stuff.
            size_t count = 0;
#ifdef USE_OPENMP
#pragma omp parallel reduction(+: count)
            {
                int threads = omp_get_num_threads();
                int thread_id = omp_get_thread_num();
#else 
            {
                int threads = 1;
                int thread_id = 0;
#endif

                // partition the local 2D tiles.  omp_tile_parts.offset is local to this processor.
                // FMT_ROOT_PRINT_RT("partitioning info : {}, {}, {}\n", tiles.size(), threads, thread_id);
		        part1D_type omp_tile_parts = partitioner.get_partition(tiles.size(), threads, thread_id);
		        // FMT_PRINT_RT("thread {} partition: ", thread_id);
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
                size_t max_id = id + omp_tile_parts.size;
                for (; id < max_id; ++id) {
                    auto part = tiles.part(id);  // id is a linear id.  tiles are filled sequentially within each proc-thread.
                    auto data = tiles.data(id);
                    // part.print("PART: ");

                    // if this tile is on the diagonal, do 1/2 comput.
                    // if it is not on the diagonal, then do full compute, or if input1 != input2

                    if ((part.r.offset != part.c.offset) || !SYMMETRIC) {
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
                                // auto xy = _ops[thread_id](input1.data(row), input2.data(col), input1.columns());
                                // auto yx = _ops[thread_id](input2.data(col), input1.data(row), input2.columns());
                                // if (xy != yx)  FMT_PRINT_RT("ERROR: distcorr not symmetric at row col ({}, {}), xy: {}, yx {}\n", row, col, xy, yx);
                                // FMT_PRINT_RT("[r, c] = [{}, {}], cols = {} {}\n", row, col, input1.columns(), input2.columns());
                                *data = run(_ops[thread_id], row, col, input1.data(row), input2.data(col), input1.columns());
                                // if (row == col) FMT_PRINT_RT("Row col {},{} input1 row col {}x{} input2 row col {}x{} data = {}\n",
                                //     row, col, input1.rows(), input1.columns(), input2.rows(), input2.columns(), *data);
                                ++data;
                            }
                        }
                    } else {
                        // on major diagonal, for self-dot-product, so can skip a little.
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
                                if (row <= col) {
                                    // upper.  so fill in.
                                    *data = run(_ops[thread_id], row, col, input1.data(row), input2.data(col), input1.columns());
                                    // if (row == col) FMT_PRINT_RT("Row col {},{} input1 row col {}x{} input2 row col {}x{} data = {}\n",
                                    //     row, col, input1.rows(), input1.columns(), input2.rows(), input2.columns(), *data);
                                    if (row < col) *data2 = *data;
                                }  // else lower half.  skip.

                                // compute correlation
                                // auto xy = _ops[thread_id](input1.data(row), input2.data(col), input1.columns());
                                // auto yx = _ops[thread_id](input2.data(col), input1.data(row), input2.columns());
                                // if (xy != yx)  FMT_PRINT_RT("ERROR: distcorr not symmetric at row col ({}, {}), xy: {}, yx {}\n", row, col, xy, yx);
                                ++data;  // advance 1 col
                                data2 += part.c.size;  // advance 1 row
                            }
                            ++data2c;  // next column
                        }
                    }
                }
        		// FMT_PRINT_RT("CORR BOUNDS r: {} {}, c {} {}\n", rmin, rmax-rmin, cmin, cmax-cmin);

                count += _ops[thread_id].processed;
        	}

        }

        void copy_to_distributed_matrix(tiles_type const & tiles,
            splash::ds::aligned_matrix<OT> & output,
            part1D_type const & row_part, size_t const & cols) const {

        	// ============= repartition output
	        auto stime = getSysTime();
            // transpose if needed
            tiles_type all_tiles;
            if (SYMMETRIC) all_tiles = tiles.reflect_diagonally();
            else all_tiles = std::move(tiles);
            auto etime = getSysTime();
	        FMT_ROOT_PRINT("Reflected tiles in {} sec\n", get_duration_s(stime, etime));

            stime = getSysTime();
            MPI_Barrier(MPI_COMM_WORLD);
            etime = getSysTime();
            FMT_MIN_MAX_DOUBLE_PRINT("PV2M Barrier before partition (s):", get_duration_s(stime, etime));

            // partition tiles by row
	        stime = getSysTime();
            tiles_type parted_tiles;
            if (this->procs == 1)
                parted_tiles = std::move(all_tiles);
            else
                parted_tiles = all_tiles.row_partition(row_part);
            etime = getSysTime();
	        FMT_ROOT_PRINT("Partitioned tiles in {} sec\n", get_duration_s(stime, etime));

            stime = getSysTime();
            MPI_Barrier(MPI_COMM_WORLD);
            etime = getSysTime();
            FMT_MIN_MAX_DOUBLE_PRINT("PV2M Barrier after partition (s):", get_duration_s(stime, etime));


            // rows are partitioned equally, but a tile may cross that partition boundary.
            // for output allocation we want the actual offset and range.
	        stime = getSysTime();
            // get actual bounds.
            part1D_type bounds;
            if (this->procs == 1)
                bounds = row_part;
            else 
                bounds = parted_tiles.get_bounds().r;
            output.resize(bounds.size, cols);
            etime = getSysTime();
	        FMT_ROOT_PRINT("Resized output in {} sec\n", get_duration_s(stime, etime));

            // copy the tiles to matrix
	        stime = getSysTime();
	        parted_tiles.copy_to(output, bounds.offset, 0);	
        	etime = getSysTime();
	        FMT_ROOT_PRINT("Copied tiles in {} sec\n", get_duration_s(stime, etime));

        }

};





template <typename IT, typename Op, typename OT, bool SYMMETRIC>
class InnerProduct<splash::ds::aligned_matrix<IT>, Op, splash::ds::aligned_matrix<OT>, SYMMETRIC> :
    public OpBase {
    public:
        using InputType = splash::ds::aligned_matrix<IT>;
        using OutputType = splash::ds::aligned_matrix<OT>;

    protected:
        splash::utils::partitioner1D<PARTITION_EQUAL> partitioner;
	
        using part1D_type = splash::utils::partition<size_t>;
        using part2D_type = splash::utils::partition2D<size_t>;
        using tiles_type = splash::ds::aligned_tiles<typename OutputType::data_type, part2D_type>;

        using Delegate = InnerProduct<InputType, Op, tiles_type, SYMMETRIC>;
        Delegate delegate;
        int procs;
        int rank;


    public:
#ifdef USE_MPI
        InnerProduct(MPI_Comm comm = MPI_COMM_WORLD) : delegate(comm) {
            MPI_Comm_size(comm, &procs);
            MPI_Comm_rank(comm, &rank);
        }
#else
        InnerProduct() : procs(1), rank(0) {};
#endif
        InnerProduct(int const & _procs, int const & _rank) : delegate(_procs, _rank),
            procs(_procs), rank(_rank) {};


        // FULL INPUT, PARTITIONED OUTPUT.
        void operator()(InputType const & input1, InputType const & input2, Op const & _op, OutputType & output) const {
            
            tiles_type tiles = delegate(input1, input2, _op); // delegate computes and return the tiles.
            this->processed = delegate.processed;

            // get the partition
            part1D_type row_part = partitioner.get_partition(input1.rows(), this->procs, this->rank);
            // re-parttition and copy the tiles to the output matrix.  result is row_part.size x columns.

            delegate.copy_to_distributed_matrix(tiles, output, row_part, input2.rows());
        }

};


}}