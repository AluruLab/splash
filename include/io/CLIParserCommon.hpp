#pragma once

// review https://attractivechaos.wordpress.com/2018/08/31/a-survey-of-argument-parsing-libraries-in-c-c/
//  eval:  want typed, subcommand, multiple occurences, single file
// contenders:  TCLAP (Y, ?, Y, N), 
//              CLI11 1.8 (Y, Y, Y, Y)  
//              cxxopts (Y, cxxsubs, Y, Y),
//              Ketopts (N, Y, Y, Y)
//              argtable(Y, ?, Y, Y)
//              args(Y, ?, Y, Y)
//              argparse(Y, Y, N, 2 )
//  decision: CLI11, has most of the features, and is being actively updated.

#include "CLI/CLI.hpp"
#include "io/parameters_base.hpp"
#include <string>

#ifdef WITH_MPI
#include <mpi.h>
#endif

namespace splash { namespace io { 

class mpi_parameters : public parameter_base {
    public:
        int procs;
        int rank;

        mpi_parameters() :
            procs(1), rank(0) {}
        virtual ~mpi_parameters() {
#ifdef WITH_MPI
            MPI_Finalize();
#endif
        };

        virtual void config(CLI::App& app) {}

        virtual void post_parse() {
#ifdef WITH_MPI
	        MPI_Init(&argc, &argv);
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        	MPI_Comm_size(MPI_COMM_WORLD, &procs);

            if (rank == 0)
                fprintf(stderr, "Warming up MPI\n");

            int * data = (int*)malloc(procs * sizeof(int));
            MPI_Alltoall(MPI_IN_PLACE, 1, MPI_INT, data, 1, MPI_INT, MPI_COMM_WORLD);
            free(data);
#endif
        }
}

class common_parameters : public parametes_base {
    public:
        std::string input;
        std::string output;
        bool random;
        // bool use_single;   // use double only, for now.
        long num_vectors;
        long vector_size;
        size_t num_threads;
        long rseed;
        double rmin;
        double rmax;

        common_parameters() :
            input(""), random(true), // use_single(true),
            num_vectors(0), vector_size(0), 
            num_threads(1), rseed(11), rmin(0.0), rmax(1.0) {}
        virtual ~common_parameters() {}


        // CLI::App should be created in the main program
        // actual parsing should also happen there.
        virtual void config(CLI::App& app) {
            // allow input, or random input, or piped input later. 
            auto input_opt = app.add_option("-i,--input", input, "input EXP formatted file")->group(common)->check(CLI::ExistingFile());
			app.add_option("-o,--output", output, "output file")->group(common);

            auto random_opt = app.add_flag("-r,--random", random, "generate random data")->group(common);
            input_opt->excludes(random_opt);
            random_opt->excludes(input_opt);

            auto seed_opt = app.add_option("--random-seed", rseed, "random number generator seed")->group(common);
            auto min_opt = app.add_option("--random-min", rmin, "random number min")->group(common);
            auto max_opt = app.add_option("--random-max", rmax, "random number max")->group(common);
            seed_opt->needs(random_opt);
            min_opt->needs(random_opt);
            max_opt->needs(random_opt);

            auto nvec_opt = app.add_option("-n,--num-vectors", num_vectors, "number of samples")->group(common)->check(CLI::PositiveNumber());
            auto vsize_opt = app.add_option("-v,--vector-size", vector_size, "number of variables per sample")->group(common)->check(CLI::PositiveNumber());
            random_opt->needs(nvec_opt);
            random_opt->needs(vsize_opt);

            // must explicitly specify to use single precision.
            // app.add_flag("-s,--single", use_single, "use single precision")->group(common);

            // default to 1 thread.
            app.add_option("-t,--threads", num_threads, "number of CPU threads")->group(common)->check(CLI::PositiveNumber());

            // MPI parameters can be extracted from MPI runtime.
        }

        virtual void print() {
            size_t numPairs = (num_vectors + 1) * num_vectors / 2;	/*including self-vs-self*/
            // splash::utils::print("Single precision: %d\n", use_single ? 1 : 0);
            splash::utils::print("Input: %s\n", input);
            splash::utils::print("Random Input: %s\n", (random ? "Y" : "N"));
            splash::utils::print("Output: %s\n", output);
            splash::utils::print("Number of vectors: %ld\n", num_vectors);
            splash::utils::print("Vector size: %ld\n", vector_size);
            splash::utils::print("Number of threads: %lu\n", num_threads);
            splash::utils::print("random number generator seed: %ld\n", rseed);
            splash::utils::print("random number min: %f\n", rmin);
            splash::utils::print("random number max: %f\n", rmax);
        }

};




}}
