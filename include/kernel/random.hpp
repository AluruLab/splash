#pragma once

#include "kernel/kernel_base.hpp"
#include <stdlib.h>  // srand, rand, for seeds
#include <time.h>    // time(0), for current time to use as seed.
#include <random>
#include <type_traits>  
#include <algorithm>  // generate_n
#include <vector>
#include "utils/partition.hpp"
#include "ds/aligned_vector.hpp"
#include "ds/aligned_matrix.hpp"

#if defined(USE_OPENMP)
#include <omp.h>
#endif

#if defined(USE_MPI)
#include <mpi.h>
#endif

// both Vector and Matrix generators only generate locally.
// for distributed initialization, first generate per proc, then gather or allgather.
// The random number generators handles distributed seeding.

namespace splash { namespace kernel { 

template <typename Generator = std::default_random_engine>
class random_number_generator {
    protected:
        std::vector<Generator> generators;
        
    public:
#ifdef USE_MPI
        random_number_generator(long const & _global_seed = 0, MPI_Comm comm = MPI_COMM_WORLD) {
#else
        random_number_generator(long const & _global_seed = 0) {
#endif
            // --------  set global seed.
            if (_global_seed == 0) srand48(time(0));
            else srand48(_global_seed);

            // -------- compute machine seed.
            int rank = 0;
#ifdef USE_MPI
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            // instead of having rank 0 generate all seeds, start all processes with same seed.
            // then get rand "rank" number of times.  the next one is the actual seed.
            //      if global seed is provided, this process should be deterministic and unlikely
            //          that 2 procs have the same value.
            //      if time is used as global seed, and in the very unlikely scenario that all have the same
            //          time stamp, then iterating over the first rank number of random numbers takes care of this.
#endif
            long seed = lrand48();  // this is the seed for the machine.
            for (int i = 0; i < rank; ++i) {
                seed ^= lrand48();   // make sure there is dependency so compiler does not optimize this out.
            }
            srand48(seed);

            // ------- compute thread seeds and generators.
            int threads = 1;
#ifdef USE_OPENMP
            threads = omp_get_max_threads();
#endif
            generators.clear();
            for (int i = 0; i < threads; ++i) {
                generators.push_back(Generator(lrand48())); // set generator (1) with seed _global_seed
            }
        }
        ~random_number_generator() {}

        Generator& get_generator(size_t const & thread_id = 0) {
            if (thread_id > generators.size()) {
                return generators[thread_id % generators.size()];  // thread safe?
            } else {
                return generators[thread_id];
            }
            
        }
};


template<typename OT, typename Distribution, typename Generator>
class RandomVectorGenerator : public splash::kernel::N2VOp<OT> {
    protected:
        random_number_generator<Generator> & generators;
        OT mn;
        OT mx;
        splash::utils::partitioner1D<PARTITION_EQUAL> partitioner;

	public:
        RandomVectorGenerator(random_number_generator<Generator> & _gen, OT const & min = 0.0, OT const & max = 1.0) : 
            generators(_gen), mn(min), mx(max) {}

		inline void operator()(splash::ds::aligned_vector<OT> & output) {
            this->operator()(output.size(), output.data());
        }

		inline void operator()(size_t const & count, OT * out_vector) {
            Distribution distribution(mn, mx);

#ifdef USE_OPENMP
#pragma omp parallel
            {
            size_t thread_id = omp_get_thread_num();
            size_t num_threads = omp_get_num_threads();
#else       
            size_t thread_id = 0;
            size_t num_threads = 1;
#endif

            // get the per-thread generator
            auto generator = generators.get_generator(thread_id);

            // get the partition
            splash::utils::partition<size_t> part = partitioner.get_partition(count, num_threads, thread_id);

            // compute. lambda capture is by reference.
            std::generate_n(out_vector + part.offset, part.size, [&generator, &distribution](){ return distribution(generator); });

#ifdef USE_OPENMP
            }
#endif
		};
};


template <typename OT>
using UniformRandomVectorGenerator = 
    splash::kernel::RandomVectorGenerator<OT, 
        typename ::std::conditional<
            ::std::is_floating_point<OT>::value,
            std::uniform_real_distribution<OT>,
            std::uniform_int_distribution<OT>>::type,
        std::default_random_engine>;

template <typename OT>
using NormalRandomVectorGenerator = 
    splash::kernel::RandomVectorGenerator<OT, 
        std::normal_distribution<OT>,
        std::default_random_engine>;



template<typename OT, typename Distribution, typename Generator>
class RandomMatrixGenerator : public splash::kernel::N2MOp<OT> {
    protected:
        random_number_generator<Generator> & generators;
        OT mn;
        OT mx;
        splash::utils::partitioner1D<PARTITION_EQUAL> partitioner;

	public:
        RandomMatrixGenerator(random_number_generator<Generator> & _gen, OT const & min = 0.0, OT const & max = 1.0) : 
            generators(_gen), mn(min), mx(max) {}

        inline void operator()(splash::ds::aligned_matrix<OT> & matrix) {
            splash::utils::partition<size_t> part(0, matrix.rows(), 0);
            this->operator()(part, matrix.columns(), matrix.column_bytes(), matrix.data());
        }
        inline void operator()(splash::ds::aligned_matrix<OT> & matrix,
            splash::utils::partition<size_t> const & part) {
            this->operator()(part, matrix.columns(), matrix.column_bytes(), matrix.data());
        }

		inline void operator()(size_t const & rows, size_t const & cols, size_t const & stride_bytes,
            OT * out_matrix) {
            splash::utils::partition<size_t> part(0, rows, 0);
            this->operator()(part, cols, stride_bytes, out_matrix);
        }

		inline void operator()(splash::utils::partition<size_t> const & part, size_t const & cols, size_t const & stride_bytes,
            OT * out_matrix) {

            Distribution distribution(mn, mx);

            size_t num_threads = 1;
            size_t thread_id = 0;

            // partition the rows by thread.
#ifdef USE_OPENMP
#pragma omp parallel
            {
            thread_id = omp_get_thread_num();
            num_threads = omp_get_num_threads();
#endif
            // get the per-thread generator
            auto generator = generators.get_generator(thread_id);

            // get the partition
            splash::utils::partition<size_t> p = partitioner.get_partition(part, num_threads, thread_id);

            // compute. lambda capture is by reference.
            OT * vec;
            size_t off = p.offset;
            for (size_t i = 0; i < p.size; ++i, ++off) {
                vec = reinterpret_cast<OT*>(reinterpret_cast<unsigned char*>(out_matrix) + off * stride_bytes);

                std::generate_n(vec, cols, [&generator, &distribution](){ return distribution(generator); });
            }
#ifdef USE_OPENMP
            }
#endif

		};

};


template <typename OT>
using UniformRandomMatrixGenerator = 
    splash::kernel::RandomMatrixGenerator<OT, 
        typename ::std::conditional<
            ::std::is_floating_point<OT>::value,
            std::uniform_real_distribution<OT>,
            std::uniform_int_distribution<OT>>::type,
        std::default_random_engine>;


template <typename OT>
using NormalRandomMatrixGenerator = 
    splash::kernel::RandomMatrixGenerator<OT, 
        std::normal_distribution<OT>,
        std::default_random_engine>;



}}