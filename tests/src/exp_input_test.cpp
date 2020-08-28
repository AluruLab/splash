/*
 * Pearson.cpp
 *
 *  Created on: June 29, 2020
 *      Author: Tony Pan
 *		  School of Computational Science & Engineering
 *		  Georgia Institute of Technology, USA
 */

#include <sstream>
#include <string>

#include "CLI/CLI.hpp"
#include "io/CLIParserCommon.hpp"
#include "io/parameters_base.hpp"
#include "utils/benchmark.hpp"
#include "utils/report.hpp"
#include "ds/aligned_matrix.hpp"

#include "io/matrix_io.hpp"

#ifdef USE_OPENMP
#include <omp.h>
#endif

class app_parameters : public parameters_base {
	public:
		enum atof_type : int { std = 0, fast = 1 };
		enum reader_type : int { lightpcc = 0, delim_str = 1, delim_char = 2 };

		atof_type atof_method;
		reader_type reader_method;

		app_parameters() : atof_method(fast), reader_method(delim_char) {}
		virtual ~app_parameters() {}

		virtual void config(CLI::App& app) {
            app.add_option("-a,--atof", atof_method, "atof impl: std=0, fast=1");
            app.add_option("-m,--method", reader_method, "reader impl: lightpcc=0, delim_str=1, delim_char=2");
		}
		virtual void print(const char * prefix) {
            ROOT_PRINT("%s atof method: %s\n", prefix, (atof_method == std ? "std::atof" : 
				"fast"));
            ROOT_PRINT("%s reader method: %s\n", prefix, (reader_method == lightpcc ? "lightpcc" : 
				reader_method == delim_str ? "multi-char delim" : "single-char delim"));
		}
};



int main(int argc, char* argv[]) {

	//==============  PARSE INPUT =====================
	CLI::App app{"EXP Reader Test"};

	// handle MPI (TODO: replace with MXX later)
	splash::io::mpi_parameters mpi_params(argc, argv);
	mpi_params.config(app);

	// set up CLI parsers.
	splash::io::common_parameters common_params;
	app_parameters app_params;

	common_params.config(app);
	app_params.config(app);

	// parse
	CLI11_PARSE(app, argc, argv);

	// print out, for fun.
	ROOT_PRINT_RT("command line: ");
	for (int i = 0; i < argc; ++i) {
		ROOT_PRINT("%s ", argv[i]);
	}
	ROOT_PRINT("\n");


#ifdef USE_OPENMP
	// omp_set_dynamic(0);
	omp_set_num_threads(common_params.num_threads);
	PRINT_RT("omp num threads %d.  user threads %lu\n", omp_get_max_threads(), common_params.num_threads);
#endif

	// =============== SETUP INPUT ===================
	// NOTE: input data is replicated on all MPI procs.
	splash::ds::aligned_matrix<double> input;
	std::vector<std::string> genes;
	std::vector<std::string> samples;

	auto stime = getSysTime();
	auto etime = getSysTime();

	stime = getSysTime();
	if (common_params.random) {
		input = make_random_matrix(common_params.rseed, 
			common_params.rmin, common_params.rmax, 
			common_params.num_vectors, common_params.vector_size,
			genes, samples);
	} else {
		if (app_params.reader_method == app_parameters::reader_type::lightpcc) {
			input = read_exp_matrix<double>(common_params.input, 
				common_params.num_vectors, common_params.vector_size,
				genes, samples);
		} else {
			input = read_exp_matrix_fast<double>(common_params.input, 
				common_params.num_vectors, common_params.vector_size,
				genes, samples, app_params.atof_method);
		}
	}
	etime = getSysTime();
	ROOT_PRINT("Load data in %f sec\n", get_duration_s(stime, etime));
	// input.print("INPUT: ");

	if (mpi_params.rank == 0) {
		mpi_params.print("[PARAM] ");
		common_params.print("[PARAM] ");
		app_params.print("[PARAM] ");
	}
	// ===== DEBUG ====== WRITE OUT INPUT =========
    {	// NOTE: rank 0 writes out.
        stime = getSysTime();
            // write to file.  MPI enabled.  Not thread enabled.
        write_exp_matrix(common_params.output, genes, samples, input);
        etime = getSysTime();
        ROOT_PRINT("dump input in %f sec\n", get_duration_s(stime, etime));
    }

	return 0;
}
