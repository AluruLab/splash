if (BUILD_TESTS)


set(splash_test_compile_defs "-DVERBOSE")
set(splash_test_compile_flags "")
set(splash_test_link_flags "")
set(splash_test_link_libs "m")


#if (USE_ZLIB)
#	set(splash_test_compile_defs "${splash_test_compile_defs};-DCOMPRESSED_INPUT")
#	set(splash_test_link_libs "z;${splash_test_link_libs}")
#endif(USE_ZLIB)

#if (USE_MKL)
#	set(splash_test_compile_defs "${splash_test_compile_defs};-DWITH_MKL;${MKL_DEFINES}")
#	set(splash_test_compile_flags "${MKL_COMPILE_FLAGS}")
#	set(splash_test_link_flags "${MKL_LINK_FLAGS}")
#	include_directories(${MKL_INCLUDE_DIRS})
#	link_directories(${MKL_LIB_DIRS})
#	set(splash_test_link_libs "${MKL_LIBS};${splash_test_link_libs}")
#endif(USE_MKL)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
	set(splash_test_compile_defs "${splash_test_compile_defs};-DHAVE_INTEL_ICC")
	if (PRINT_VEC_REPORT)
		set(splash_test_compile_flags "${splash_test_compile_flags};-qopt-report=5;-qopt-report-phase=vec;-qopt-report-file=stdout")
	endif(PRINT_VEC_REPORT)

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	if (PRINT_VEC_REPORT)
		set(splash_test_compile_flags "${splash_test_compile_flags};-ftree-vectorizer-verbose=4")
	endif(PRINT_VEC_REPORT)

endif()


set(CORRELATION_APPS csv_input_test exp_input_test)

foreach( exec IN LISTS CORRELATION_APPS)
	add_executable(${exec} ${exec}.cpp)
	foreach (cdef IN LISTS splash_test_compile_defs)
		target_compile_definitions(${exec} PRIVATE ${cdef})
	endforeach(cdef)
	foreach (cflgs IN LISTS splash_test_compile_flags)
		target_compile_options(${exec} PRIVATE ${cflgs})
	endforeach(cflgs)
	target_link_libraries(${exec} ${splash_test_link_flags} ${splash_test_link_libs} ${EXTRA_LIBS})

	if (USE_MPI)
		target_compile_definitions(${exec} PRIVATE -DUSE_MPI)
		# MPI_LIBRARIES already in EXTRA_LIBS
	endif(USE_MPI)
endforeach(exec)




endif(BUILD_TESTS)


