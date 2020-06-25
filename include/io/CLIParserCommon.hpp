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
#include "utils/report.hpp"
#include <string>

#ifdef USE_MPI
#include <mpi.h>
#endif

namespace splash { namespace io { 

class mpi_parameters : public parameters_base {
    public:
        int procs;
        int rank;

        mpi_parameters(int argc, char* argv[]) :
            procs(1), rank(0) {
#ifdef USE_MPI
	        MPI_Init(&argc, &argv);
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        	MPI_Comm_size(MPI_COMM_WORLD, &procs);
#endif
            }
        virtual ~mpi_parameters() {
#ifdef USE_MPI
            MPI_Finalize();
#endif
        }

        virtual void config(CLI::App& app) {
#ifdef USE_MPI
            PRINT("Config: warm up MPI with %d procs\n", procs);

            int * data = (int*)malloc(procs * sizeof(int));
            MPI_Alltoall(MPI_IN_PLACE, 1, MPI_INT, data, 1, MPI_INT, MPI_COMM_WORLD);
            free(data);
#endif
        }

        virtual void print() {
            PRINT("Number of MPI processes: %d\n", procs);
        }
};

class common_parameters : public parameters_base {
    public:
        std::string input;
        std::string output;
        bool random;
        // bool use_single;   // use double only, for now.
        size_t num_vectors;
        size_t vector_size;
        size_t num_threads;
        long rseed;
        double rmin;
        double rmax;

        common_parameters() :
            input(""), random(false), // use_single(true),
            num_vectors(0), vector_size(0), 
            num_threads(1), rseed(11), rmin(0.0), rmax(1.0) {}
        virtual ~common_parameters() {}


        // CLI::App should be created in the main program
        // actual parsing should also happen there.
        virtual void config(CLI::App& app) {
            // allow input, or random input, or piped input later. 
            auto input_opt = app.add_option("-i,--input", input, "input EXP formatted file")->check(CLI::ExistingFile);
            auto random_opt = app.add_flag("-r,--random", random, "generate random data");

            // require either -i input, or -r random.
            auto opt_group = app.add_option_group("Input Source");
            opt_group->add_option(input_opt);
            opt_group->add_option(random_opt);
            opt_group->require_option(1);

            // this should be required for random and optional for input file.
            auto nvec_opt = app.add_option("-n,--num-vectors", num_vectors, "number of samples")->group("Data")->check(CLI::PositiveNumber);
            auto vsize_opt = app.add_option("-v,--vector-size", vector_size, "number of variables per sample")->group("Data")->check(CLI::PositiveNumber);
            random_opt->needs(nvec_opt);
            random_opt->needs(vsize_opt);


            auto seed_opt = app.add_option("--random-seed", rseed, "random number generator seed")->group("RNG");
            auto min_opt = app.add_option("--random-min", rmin, "random number min")->group("RNG");
            auto max_opt = app.add_option("--random-max", rmax, "random number max")->group("RNG");
            seed_opt->needs(random_opt);
            min_opt->needs(random_opt);
            max_opt->needs(random_opt);

            // output
            app.add_option("-o,--output", output, "output file")->group("Output");

            // must explicitly specify to use single precision.
            // app.add_flag("-s,--single", use_single, "use single precision")->group("common");

            // default to 1 thread.
            app.add_option("-t,--threads", num_threads, "number of CPU threads")->group("Hardware")->check(CLI::PositiveNumber);

            // MPI parameters can be extracted from MPI runtime.
        }

        virtual void print() {
            size_t numPairs = (num_vectors + 1) * num_vectors / 2;	/*including self-vs-self*/
            // PRINT("Single precision: %d\n", use_single ? 1 : 0);
            PRINT("Input: %s\n", input.c_str());
            PRINT("Random Input: %s\n", (random ? "Y" : "N"));
            PRINT("Output: %s\n", output.c_str());
            PRINT("Number of vectors: %ld\n", num_vectors);
            PRINT("Vector size: %ld\n", vector_size);
            PRINT("number of pairs: %lu\n", numPairs);
            PRINT("Number of threads: %lu\n", num_threads);
            PRINT("random number generator seed: %ld\n", rseed);
            PRINT("random number min: %f\n", rmin);
            PRINT("random number max: %f\n", rmax);
        }

};




}}
