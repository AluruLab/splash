/*
 * Copyright 2020 Georgia Tech Research Corporation
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

#pragma once

#include "splash/kernel/kernel_base.hpp"
#include <stdlib.h>  // srand, rand, for seeds
#include <time.h>    // time(0), for current time to use as seed.
#include <random>
#include <type_traits>  
#include <algorithm>  // generate_n
#include <vector>
#include "splash/utils/partition.hpp"
#include "splash/ds/aligned_vector.hpp"
#include "splash/ds/aligned_matrix.hpp"

#include <sys/types.h>  // pid_t
#include <unistd.h>     // getpid()

#if defined(USE_OPENMP)
#include <omp.h>
#endif

#if defined(USE_MPI)
#include <mpi.h>
#endif

// both Vector and Matrix generators only generate locally.
// for distributed initialization, each rank calls lrand48 differnent number of times.
// The random_number_generator handles distributed seeding.
// TINGE:
//     random number seed is either user-specified (identical for all ranks, not okay).
//     "current time + process id in os".  (should be different on each process.  use this.).
//      
namespace splash { namespace kernel { 

template <typename Generator = std::default_random_engine>
class random_number_generator {
    protected:
        std::vector<Generator> generators;
        
        union seed_type {
            long s;
            int p[2];
        };

    public:
#ifdef USE_MPI
        random_number_generator(long const & _global_seed = 0, MPI_Comm comm = MPI_COMM_WORLD) {
#else
        random_number_generator(long const & _global_seed = 0) {
#endif
            // --------  set global seed.
            long global;
            if (_global_seed == 0) global = time(0) + getpid();
            else global = _global_seed;
                
            // -------- compute machine seed.
            seed_type seed;
#ifdef USE_MPI
            MPI_Comm_rank(MPI_COMM_WORLD, seed.p);
            // instead of having rank 0 generate all seeds, 
            //  deterministically compute a seed for each rank and thread (via composition);
#else 
            seed.p[0] = 0;
#endif

            // ------- compute thread seeds and generators.
            int threads = 1;
#ifdef USE_OPENMP
            threads = omp_get_max_threads();
#endif
            generators.clear();
            for (int i = 0; i < threads; ++i) {
                seed.p[1] = i;  // do
                generators.push_back(Generator(global ^ seed.s)); // set generator (1) with seed _global_seed
            }
        }
        ~random_number_generator() {}
        random_number_generator(random_number_generator const & other) : generators(other.generators) {};

        Generator& get_generator(size_t const & thread_id = 0) {
            if (thread_id > generators.size()) {
                return generators[thread_id % generators.size()];  // thread safe?
            } else {
                return generators[thread_id];
            }
            
        }
};



template<typename OT, typename Distribution, typename Generator>
class RandomNumberGenerator : public splash::kernel::generate<OT, splash::kernel::DEGREE::SCALAR> {
    protected:
        random_number_generator<Generator> & generators;
        OT mn;
        OT mx;
        Distribution distribution;

	public:
        using InputType = void;
        using OutputType = OT;

        RandomNumberGenerator(random_number_generator<Generator> & _gen, OT const & min = 0.0, OT const & max = 1.0) : 
            generators(_gen), mn(min), mx(max), distribution(min, max) {}
        virtual ~RandomNumberGenerator() {}
        void copy_parameters(RandomNumberGenerator const & other) {
            generators = other.generators;
            mn = other.mn;
            mx = other.mx;
            distribution = other.distribution;
        }

		inline virtual OT operator()() const {
            size_t num_threads = 1;
            size_t thread_id = 0;

#ifdef USE_OPENMP
#pragma omp parallel
            {
                thread_id = omp_get_thread_num();
                num_threads = omp_get_num_threads();
            }
#endif

            // get the per-thread generator
            auto generator = generators.get_generator(thread_id);

            // compute. lambda capture is by reference.
            return distribution(generator);
		};
};


template <typename OT>
using UniformRandomNumberGenerator = 
    splash::kernel::RandomNumberGenerator<OT, 
        typename ::std::conditional<
            ::std::is_floating_point<OT>::value,
            std::uniform_real_distribution<OT>,
            std::uniform_int_distribution<OT>>::type,
        std::default_random_engine>;

template <typename OT>
using NormalRandomNumberGenerator = 
    splash::kernel::RandomNumberGenerator<OT, 
        std::normal_distribution<OT>,
        std::default_random_engine>;


template<typename OT, typename Distribution, typename Generator>
class RandomVectorGenerator : public splash::kernel::generate<OT, splash::kernel::DEGREE::VECTOR> {
    protected:
        random_number_generator<Generator> & generators;
        OT mn;
        OT mx;
        splash::utils::partitioner1D<PARTITION_EQUAL> partitioner;

	public:
        using InputType = void;
        using OutputType = OT;

        RandomVectorGenerator(random_number_generator<Generator> & _gen, OT const & min = 0.0, OT const & max = 1.0) : 
            generators(_gen), mn(min), mx(max) {}
        virtual ~RandomVectorGenerator() {}
        void copy_parameters(RandomVectorGenerator const & other) {
            generators = other.generators;
            mn = other.mn;
            mx = other.mx;
        }

		inline void operator()(splash::ds::aligned_vector<OT> & output) const {
            this->operator()(output.size(), output.data());
        }

        inline void operator()(splash::ds::aligned_vector<OT> & vector,
            splash::utils::partition<size_t> const & part) const {
            this->operator()(part.size, vector.data(part.offset));
        }

		inline virtual void operator()(size_t const & count, OT * out_vector) const {
            Distribution distribution(mn, mx);

#ifdef USE_OPENMP
#pragma omp parallel
            {
            size_t thread_id = omp_get_thread_num();
            size_t num_threads = omp_get_num_threads();
#else
            size_t num_threads = 1;
            size_t thread_id = 0;
#endif


            // get the per-thread generator
            auto generator = generators.get_generator(thread_id);

            // get the partition
            splash::utils::partition<size_t> part = partitioner.get_partition(count, num_threads, thread_id);

            // compute. lambda capture is by reference.
            std::generate_n(out_vector + part.offset, part.size, [&generator, &distribution](){ return distribution(generator); });

#ifdef USE_OPENMP
#pragma omp barrier
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
class RandomMatrixGenerator : public splash::kernel::generate<OT, splash::kernel::DEGREE::MATRIX> {
    protected:
        random_number_generator<Generator> & generators;
        OT mn;
        OT mx;
        splash::utils::partitioner1D<PARTITION_EQUAL> partitioner;  // row partitioned.

	public:
        using InputType = void;
        using OutputType = OT;

        RandomMatrixGenerator(random_number_generator<Generator> & _gen, OT const & min = 0.0, OT const & max = 1.0) : 
            generators(_gen), mn(min), mx(max) {}
        virtual ~RandomMatrixGenerator() {}
        void copy_parameters(RandomMatrixGenerator const & other) {
            generators = other.generators;
            mn = other.mn;
            mx = other.mx;
        }

        inline void operator()(splash::ds::aligned_matrix<OT> & matrix) const {
            splash::utils::partition<size_t> part(0, matrix.rows(), 0);
            this->operator()(part, matrix.columns(), matrix.column_bytes(), matrix.data());
        }
        inline void operator()(splash::ds::aligned_matrix<OT> & matrix,
            splash::utils::partition<size_t> const & part) const {
            this->operator()(part, matrix.columns(), matrix.column_bytes(), matrix.data());
        }

		inline virtual void operator()(size_t const & rows, size_t const & cols, size_t const & stride_bytes,
            OT * out_matrix) const {
            splash::utils::partition<size_t> part(0, rows, 0);
            this->operator()(part, cols, stride_bytes, out_matrix);
        }

		inline void operator()(splash::utils::partition<size_t> const & part, size_t const & cols, size_t const & stride_bytes,
            OT * out_matrix) const {

            Distribution distribution(mn, mx);


            // partition the rows by thread.
#ifdef USE_OPENMP
#pragma omp parallel
            {
            size_t thread_id = omp_get_thread_num();
            size_t num_threads = omp_get_num_threads();
#else
            size_t num_threads = 1;
            size_t thread_id = 0;
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