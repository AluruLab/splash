/*
 * Copyright 2021 Georgia Tech Research Corporation
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
#include "splash/io/parameters_base.hpp"
#include "splash/utils/report.hpp"
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
            FMT_ROOT_PRINT("Config: warm up MPI with {} procs\n", procs);

            int * data = (int*)malloc(procs * sizeof(int));
            MPI_Alltoall(MPI_IN_PLACE, 1, MPI_INT, data, 1, MPI_INT, MPI_COMM_WORLD);
            free(data);
#endif
        }

        virtual void print(const char* prefix) const {
            FMT_ROOT_PRINT("{} No. of MPI procs    : {}\n", prefix, procs);
        }
};

class common_parameters : public parameters_base {
    public:
        std::string input;
        std::string output;
        std::string h5_group;
        std::string h5_matrix_key;
        std::string h5_gene_key;
        std::string h5_samples_key;
        bool random;
        bool use_single;   // use double only, for now.
        size_t num_vectors;
        size_t vector_size;
        size_t num_threads;
        long rseed;
        double rmin;
        double rmax;
        bool skip;

        common_parameters() :
            input(""), output(""),
            h5_group("array"), h5_matrix_key("block0_values"),
            h5_gene_key("axis1"), h5_samples_key("axis0"),
            random(false), use_single(false),
            num_vectors(0), vector_size(0), 
            num_threads(1), rseed(11), rmin(0.0), rmax(1.0), skip(false) {}
        virtual ~common_parameters() {}


        // CLI::App should be created in the main program
        // actual parsing should also happen there.
        virtual void config(CLI::App& app) {
            // allow input, or random input, or piped input later. 
            auto input_opt = app.add_option("-i,--input", input, "input H5/EXP formatted file")->check(CLI::ExistingFile);
            auto random_opt = app.add_flag("-r,--random", random, "generate random data");

            // require either -i input, or -r random.
            auto opt_group = app.add_option_group("Input Source");
            opt_group->add_option(input_opt);
            opt_group->add_option(random_opt);
            opt_group->require_option(1);
            //

            // this should be required for random and optional for input file.
            auto nvec_opt = app.add_option("-n,--num-vectors", num_vectors, "No. of samples (Required for random generation; Optional for input file)")->group("Data")->check(CLI::PositiveNumber);
            auto vsize_opt = app.add_option("-v,--vector-size", vector_size, "No. of variables per sample (Required for random generation; Optional for input file)")->group("Data")->check(CLI::PositiveNumber);
            random_opt->needs(nvec_opt);
            random_opt->needs(vsize_opt);

            app.add_flag(
                "--skip", skip, "skip lines 2 and 3 (exp files only)"
                )->group("Data Input")->capture_default_str()->needs(input_opt);
            app.add_option(
                "--hdf_group", h5_group,
                "Group entry in HDF5 file under which the data matrix,"
                " gene names list and sample names list are stored (can be '/')"
                )->group("Data Input")->capture_default_str()->needs(input_opt);
            app.add_option(
                "--hdf_matrix", h5_matrix_key, 
                "Dataset Identfier for the data matrix in HDF5 Input file"
               )->group("Data Input")->capture_default_str()->needs(input_opt);
            app.add_option(
                "--hdf_gene", h5_gene_key, 
                "Dataset Identfier for Array of Gene Names in HDF5 Input File"
               )->group("Data Input")->capture_default_str()->needs(input_opt);
            app.add_option(
                "--hdf_sample", h5_samples_key, 
                "Dataset Identifier for Array of Sample Names in HDF5 Input File"
               )->group("Data Input")->capture_default_str()->needs(input_opt);

            auto seed_opt = app.add_option(
                "--random-seed", rseed, "Random number generator seed"
                )->group("Data Input")->capture_default_str();
            auto min_opt = app.add_option(
                "--random-min", rmin, "Random number min"
                )->group("Data Input")->capture_default_str();
            auto max_opt = app.add_option(
                "--random-max", rmax, "Random number max"
                )->group("Data Input")->capture_default_str();
            seed_opt->needs(random_opt);
            min_opt->needs(random_opt);
            max_opt->needs(random_opt);

            // output
            app.add_option("-o,--output", output, "output file")->group("Output");

            // must explicitly specify to use single precision.
            app.add_flag("-s,--single", use_single, "use single precision")->group("Common")->capture_default_str();

            // default to 1 thread.
            app.add_option("-t,--threads", num_threads, "No. of CPU threads")->group("Hardware")->check(CLI::PositiveNumber);

            // MPI parameters can be extracted from MPI runtime.
        }

        virtual void print(const char* prefix) const {
            size_t numPairs = (num_vectors + 1) * num_vectors / 2;	/*including self-vs-self*/
            FMT_ROOT_PRINT("{} File Input          : {}\n", prefix, input.c_str());
			FMT_ROOT_PRINT("{} ->HDF5 data root    : {}\n", prefix, h5_group);
			FMT_ROOT_PRINT("{} ->HDF5 genes key    : {}\n", prefix, h5_gene_key);
			FMT_ROOT_PRINT("{} ->HDF5 samples key  : {}\n", prefix, h5_samples_key);
			FMT_ROOT_PRINT("{} ->HDF5 matrix key   : {}\n", prefix, h5_matrix_key);
            FMT_ROOT_PRINT("{} ->Skip lines 2 & 3  : {}\n", prefix, (skip ? "Y" : "N"));
            FMT_ROOT_PRINT("{} Random Input        : {}\n", prefix, (random ? "Y" : "N"));
            FMT_ROOT_PRINT("{} ->RNG seed          : {}\n", prefix, rseed);
            FMT_ROOT_PRINT("{} ->Random number min : {}\n", prefix, rmin);
            FMT_ROOT_PRINT("{} ->Random number max : {}\n", prefix, rmax);
            FMT_ROOT_PRINT("{} Output              : {}\n", prefix, output.c_str());
            FMT_ROOT_PRINT("{} Data                : \n", prefix);
            FMT_ROOT_PRINT("{} ->Single precision  : {}\n", prefix, use_single ? "Y" : "N");
            FMT_ROOT_PRINT("{} ->Number of vectors : {}\n", prefix, num_vectors);
            FMT_ROOT_PRINT("{} ->Vector size       : {}\n", prefix, vector_size);
            FMT_ROOT_PRINT("{} ->No. of pairs      : {}\n", prefix, numPairs);
            FMT_ROOT_PRINT("{} No. of threads      : {}\n", prefix, num_threads);
        }

};




}}
