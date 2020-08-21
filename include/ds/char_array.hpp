/*
 * char_array.hpp
 *
 *  Created on: Aug 20, 2020
 *  Author: Tony Pan
 *  Affiliation: Institute for Data Engineering and Science
 *  						Georgia Institute of Technology, Atlanta, GA 30332
 *  
 */
#pragma once

#include <algorithm>  // find, find_if

// #include "utils/benchmark.hpp"
// #include "utils/report.hpp"


namespace splash { namespace ds { 

struct char_array_stl {
	// strtok modifies the buffer in memory.  avoid.

	static const char EOL[];
	static constexpr char CR = '\r';
	static constexpr char COMMA = ',';
	static constexpr size_t EMPTY = 0;

	char * ptr;
	size_t size;

	// search buffer to get to the first non-delim character, and return size.
	template <size_t N>
	char_array_stl trim_left(const char (&delim)[N]) {
		// auto stime = getSysTime();	

		char_array_stl output = {ptr, EMPTY};
		if ((size == EMPTY) || (ptr == nullptr)) return output;  // buffer is empty, short circuit.
		auto delim_end = delim + N;

		// find first entry that is NOT in delim
		ptr = std::find_if(ptr, ptr + size,
			[&delim, &delim_end](char const & c) {
				// by failing to find a match
				return (std::find(delim, delim_end, c) == delim_end);
			});
		
		output.size = std::distance(output.ptr, ptr);
		size -= output.size;

		// auto etime = getSysTime();
		// ROOT_PRINT("trim_left x in %f sec\n", get_duration_s(stime, etime));
		return output;
	}
	char_array_stl trim_left(const char delim) {
		// auto stime = getSysTime();	
		char_array_stl output = {ptr, EMPTY};
		if ((size == EMPTY) || (ptr == nullptr)) return output;  // buffer is empty, short circuit.

		// find first entry that is NOT in delim
		ptr = std::find_if(ptr, ptr + size,
			[&delim](char const & c) {
				// by failing to find a match
				return delim != c;
			});
		output.size = std::distance(output.ptr, ptr);
		size -= output.size;

		// auto etime = getSysTime();	
		// ROOT_PRINT("trim_left in %f sec\n", get_duration_s(stime, etime));
		return output;
	}

	// return token (ptr and length). does not treat consecutive delimiters as one.
	// assumes buffer points to start of token.
	// if there is no more token, then return nullptr
	template <size_t N>
	char_array_stl extract_token(const char (&delim)[N]) {
		// auto stime = getSysTime();	
		if ((size == EMPTY) || (ptr == nullptr)) return char_array_stl{nullptr, EMPTY};  // buffer is empty, short circuit.
		char_array_stl output = {ptr, EMPTY};
		
		auto delim_end = delim + N;

		// search for first match to delim
		ptr = std::find_if(ptr, ptr + size,
		[&delim, &delim_end](char const & c){
			// by finding a match
			return (std::find(delim, delim_end, c) != delim_end);
		});

		// update the output and buffer sizes.
		output.size = std::distance(output.ptr, ptr);
		size -= output.size;
		// printf("extract_token: %lu\n", output.size);
		// auto etime = getSysTime();	
		// ROOT_PRINT("extract token x in %f sec\n", get_duration_s(stime, etime));
		return output;
	}
	char_array_stl extract_token(const char delim) {
		// auto stime = getSysTime();	
		if ((size == EMPTY) || (ptr == nullptr)) return char_array_stl{nullptr, EMPTY};  // buffer is empty, short circuit.
		char_array_stl output = {ptr, EMPTY};
		
		// search for first match to delim
		ptr = std::find_if(ptr, ptr + size,
		[&delim](char const & c){
			return (delim == c);
			// return (std::find(delim, delim_end, c) != delim_end);
		});
		// update the output and buffer sizes.
		output.size = std::distance(output.ptr, ptr);
		size -= output.size;
		// printf("extract_token: %lu\n", output.size);
		// auto etime = getSysTime();	
		// ROOT_PRINT("extract token in %f sec\n", get_duration_s(stime, etime));
		return output;
	}
	// remove 1 delim character from buffer (or \r\n)
	template <size_t N>
	char_array_stl trim_left_1(const char (&delim)[N]) {
		// auto stime = getSysTime();	

		char_array_stl output = {ptr, EMPTY};
		if ((size == EMPTY) || (ptr == nullptr)) return output;  // buffer is empty, short circuit.
		
		// check if CR and in delim.  if yes, skip.
		if ((ptr[0] == CR) &&
			(std::find(delim, delim + N, CR) != delim + N)) {
			--size;	
			++ptr;
		}

		if ((size > EMPTY) && 
			(std::find(delim, delim + N, ptr[0]) != delim + N)) {
			--size;	
			++ptr;
		}
		output.size = std::distance(output.ptr, ptr);
		// printf("trim_left_1: %lu\n", output.size);
		// auto etime = getSysTime();	
		// ROOT_PRINT("trim_left_1 x in %f sec\n", get_duration_s(stime, etime));
		return output;
	}
	char_array_stl trim_left_1(const char delim) {
		// auto stime = getSysTime();	
		char_array_stl output = {ptr, EMPTY};
		if ((size == EMPTY) || (ptr == nullptr)) return output;  // buffer is empty, short circuit.
		
		if ((size > EMPTY) && (ptr[0] == delim)) {
			--size;	
			++ptr;
		}
		output.size = std::distance(output.ptr, ptr);
		// printf("trim_left_1: %lu\n", output.size);
		// auto etime = getSysTime();	
		// ROOT_PRINT("trim_left_1 in %f sec\n", get_duration_s(stime, etime));
		return output;
	}

	// get a token. does not treat consecutive delimiters as one so output may be empty.
	template <size_t N>
	char_array_stl get_token_or_empty(const char (&delim)[N]) {
		char_array_stl output = extract_token(delim);
		trim_left_1(delim);
		return output;
	}
	char_array_stl get_token_or_empty(const char delim) {
		char_array_stl output = extract_token(delim);
		trim_left_1(delim);
		return output;
	}
	// return token (ptr and length). treats consecutive delimiters as one.
	template <size_t N>
	char_array_stl get_token(const char (&delim)[N]) {
		char_array_stl output = extract_token(delim);
		trim_left(delim);
		return output;
	}		// return token (ptr and length). treats consecutive delimiters as one.
	char_array_stl get_token(const char delim) {
		char_array_stl output = extract_token(delim);
		trim_left(delim);
		return output;
	}

	template <size_t N>
	size_t count_token_or_empty(const char (&delim)[N]) {
		// auto stime = getSysTime();
		if ((size == EMPTY) || (ptr == nullptr)) return 0;

		auto delim_end = delim + N;
		// now count delimiters
		size_t count = std::count_if(ptr, ptr + size,
			[&delim, &delim_end](const char & c){
				return (std::find(delim, delim_end, c) != delim_end);
			}) + 1;

		// auto etime = getSysTime();
		// ROOT_PRINT("count tokens or empty in %f sec\n", get_duration_s(stime, etime));
		return count;
	}
	size_t count_token_or_empty(const char delim) {
		// auto stime = getSysTime();
		if ((size == EMPTY) || (ptr == nullptr)) return 0;

		// now count delimiters
		size_t count = std::count_if(ptr, ptr + size,
			[&delim](const char & c){
				return (delim == c);
				// return (std::find(delim, delim_end, c) != delim_end);
			}) + 1;

		// auto etime = getSysTime();
		// ROOT_PRINT("count tokens or empty in %f sec\n", get_duration_s(stime, etime));
		return count;
	}
	template <size_t N>
	size_t count_token(const char (&delim)[N]) {
		// auto stime = getSysTime();
		if ((size == EMPTY) || (ptr == nullptr)) return 0;

		size_t count = 0;

		// set up the adjacent find predicate
		auto delim_end = delim + N;
		auto adj_find_pred = [&delim, &delim_end](const char & x, const char & y){
			bool x_no_delim = std::find(delim, delim_end, x) == delim_end;
			bool y_delim = std::find(delim, delim_end, y) != delim_end;
			return x_no_delim && y_delim; 
		};

		// now count non-delimiter to delimiter transitions
		auto start = ptr;
		auto end = ptr + size;
		while ((start = std::adjacent_find(start, end, adj_find_pred)) != end) {
			// found a transition
			++start; // go to next char
			++count; // increment count
		}

		// check the last char to see if we we have a non-delim to null transition.
		if (std::find(delim, delim_end, ptr[size - 1]) == delim_end) {
			++count;
		}
		// auto etime = getSysTime();
		// ROOT_PRINT("count tokens in %f sec\n", get_duration_s(stime, etime));
		return count;
	}

	size_t count_token(const char delim) {
		// auto stime = getSysTime();
		if ((size == EMPTY) || (ptr == nullptr)) return 0;

		size_t count = 0;

		// set up the adjacent find predicate
		auto adj_find_pred = [&delim](const char & x, const char & y){
			return (x != delim) && (y == delim); 
		};

		// now count non-delimiter to delimiter transitions
		auto start = ptr;
		auto end = ptr + size;
		while ((start = std::adjacent_find(start, end, adj_find_pred)) != end) {
			// found a transition
			++start; // go to next char
			++count; // increment count
		}
		// check the last char to see if we we have a non-delim to null transition.
		count += (ptr[size - 1] != delim);
		
		// auto etime = getSysTime();
		// ROOT_PRINT("count tokens in %f sec\n", get_duration_s(stime, etime));
		return count;
	}

};
const char char_array_stl::EOL[] = "\r\n";


class char_array_loop {
	// strtok modifies the buffer in memory.  avoid.

	public:
		static const char EOL[];
		static constexpr char CR = '\r';
		static constexpr char COMMA = ',';
		static constexpr size_t EMPTY = 0;

		char * ptr;
		size_t size;

	protected:
		template <size_t N>
		inline bool is_delimiter(const char (&delim)[N], const char & c) {
			for (size_t i = 0; i < N; ++i) {
				if (delim[i] == c) return true;
			}
			return false;
		}
		inline bool is_delimiter(const char & delim, const char & c) {
			return c == delim;
		}
		template <size_t N>
		inline bool is_token_end(const char (&delim)[N], const char & x, const char & y) {
			bool x_is_not = true;
			bool y_is = false;
			for (size_t i = 0; i < N; ++i) {
				x_is_not &= (x != delim[i]);
				y_is |= (y == delim[i]);
			}
			return x_is_not && y_is;
		}
		inline bool is_token_end(const char & delim, const char & x, const char & y) {
			return (x != delim) && (y == delim);
		}

	public:
		// search buffer to get to the first non-delim character, and return size.
		template <typename D>
		char_array_loop trim_left(D delim) {
			// auto stime = getSysTime();	

			char_array_loop output = {ptr, EMPTY};
			if ((size == EMPTY) || (ptr == nullptr)) return output;  // buffer is empty, short circuit.
			
			// find first entry that is NOT in delim
			ptr = std::find_if_not(ptr, ptr + size,
				[&delim](char const & c) {
					return is_delimiter(delim, c);
				});
			
			output.size = std::distance(output.ptr, ptr);
			size -= output.size;

			// auto etime = getSysTime();
			// ROOT_PRINT("trim_left x in %f sec\n", get_duration_s(stime, etime));
			return output;
		}

		// return token (ptr and length). does not treat consecutive delimiters as one.
		// assumes buffer points to start of token.
		// if there is no more token, then return nullptr
		template <typename D>
		char_array_loop extract_token(D delim) {
			// auto stime = getSysTime();	
			if ((size == EMPTY) || (ptr == nullptr)) return char_array_loop{nullptr, EMPTY};  // buffer is empty, short circuit.
			char_array_loop output = {ptr, EMPTY};
			
			// search for first match to delim
			ptr = std::find_if(ptr, ptr + size,
				[&delim](char const & c){
					return is_delimiter(delim, c);
				});

			// update the output and buffer sizes.
			output.size = std::distance(output.ptr, ptr);
			size -= output.size;
			// printf("extract_token: %lu\n", output.size);
			// auto etime = getSysTime();	
			// ROOT_PRINT("extract token x in %f sec\n", get_duration_s(stime, etime));
			return output;
		}
		// remove 1 delim character from buffer (or \r\n)
		template <typename D>
		char_array_loop trim_left_1(D delim) {
			// auto stime = getSysTime();	

			char_array_loop output = {ptr, EMPTY};
			if ((size == EMPTY) || (ptr == nullptr)) return output;  // buffer is empty, short circuit.
			
			// check if CR and in delim.  if yes, skip.
			if ((ptr[0] == CR) && is_delimiter(delim, CR)) {
				--size;	
				++ptr;
			}

			if ((size > EMPTY) && is_delimiter(delim, ptr[0])) {
				--size;	
				++ptr;
			}
			output.size = std::distance(output.ptr, ptr);
			// printf("trim_left_1: %lu\n", output.size);
			// auto etime = getSysTime();	
			// ROOT_PRINT("trim_left_1 x in %f sec\n", get_duration_s(stime, etime));
			return output;
		}
		char_array_loop trim_left_1(const char & delim) {
			// auto stime = getSysTime();	
			char_array_loop output = {ptr, EMPTY};
			if ((size == EMPTY) || (ptr == nullptr)) return output;  // buffer is empty, short circuit.
			
			if ((size > EMPTY) && is_delimiter(delim, ptr[0])) {
				--size;	
				++ptr;
			}
			output.size = std::distance(output.ptr, ptr);
			// printf("trim_left_1: %lu\n", output.size);
			// auto etime = getSysTime();	
			// ROOT_PRINT("trim_left_1 in %f sec\n", get_duration_s(stime, etime));
			return output;
		}

		// get a token. does not treat consecutive delimiters as one so output may be empty.
		template <size_t N>
		char_array_loop get_token_or_empty(const char (&delim)[N]) {
			char_array_loop output = extract_token(delim);
			trim_left_1(delim);
			return output;
		}
		char_array_loop get_token_or_empty(const char delim) {
			char_array_loop output = extract_token(delim);
			trim_left_1(delim);
			return output;
		}
		// return token (ptr and length). treats consecutive delimiters as one.
		template <size_t N>
		char_array_loop get_token(const char (&delim)[N]) {
			char_array_loop output = extract_token(delim);
			trim_left(delim);
			return output;
		}		// return token (ptr and length). treats consecutive delimiters as one.
		char_array_loop get_token(const char delim) {
			char_array_loop output = extract_token(delim);
			trim_left(delim);
			return output;
		}

		template <size_t N>
		size_t count_token_or_empty(const char (&delim)[N]) {
			// auto stime = getSysTime();
			if ((size == EMPTY) || (ptr == nullptr)) return 0;

			auto delim_end = delim + N;
			// now count delimiters
			size_t count = std::count_if(ptr, ptr + size,
				[&delim, &delim_end](const char & c){
					for (size_t i = 0; i < N; ++i) {
						if (delim[i] == c) return true;
					}
					return false;

				}) + 1;

			// auto etime = getSysTime();
			// ROOT_PRINT("count tokens or empty in %f sec\n", get_duration_s(stime, etime));
			return count;
		}
		size_t count_token_or_empty(const char delim) {
			// auto stime = getSysTime();
			if ((size == EMPTY) || (ptr == nullptr)) return 0;

			// now count delimiters
			size_t count = 0;
			for (size_t i = 0; i < size; ++i) {
				if (ptr[i] == delim) ++count;
			}
			++count;

			// auto etime = getSysTime();
			// ROOT_PRINT("count tokens or empty in %f sec\n", get_duration_s(stime, etime));
			return count;
		}
		template <size_t N>
		size_t count_token(const char (&delim)[N]) {
			// auto stime = getSysTime();
			if ((size == EMPTY) || (ptr == nullptr)) return 0;

			size_t count = 0;

			// set up the adjacent find predicate
			auto delim_end = delim + N;
			auto adj_find_pred = [&delim, &delim_end](const char & x, const char & y){
				// bool x_no_delim = std::find(delim, delim_end, x) == delim_end;
				// bool y_delim = std::find(delim, delim_end, y) != delim_end;
				bool x_no_delim = true;
				bool y_delim = false;
				for (size_t i = 0; i < N; ++i) {
					x_no_delim &= (delim[i] != x);
					y_delim |= (delim[i] == y);
				}
				return x_no_delim && y_delim; 
			};

			// now count non-delimiter to delimiter transitions
			auto start = ptr;
			auto end = ptr + size;
			while ((start = std::adjacent_find(start, end, adj_find_pred)) != end) {
				// found a transition
				++start; // go to next char
				++count; // increment count
			}

			// check the last char to see if we we have a non-delim to null transition.
			bool z_no_delim = true;
			for (size_t i = 0; i < N; ++i) {
				z_no_delim &= (delim[i] != ptr[size - 1]);
			}
			count += z_no_delim;
			

			// auto etime = getSysTime();
			// ROOT_PRINT("count tokens in %f sec\n", get_duration_s(stime, etime));
			return count;
		}

		size_t count_token(const char delim) {
			// auto stime = getSysTime();
			if ((size == EMPTY) || (ptr == nullptr)) return 0;

			size_t count = 0;

			// now count non-delimiter to delimiter transitions
			for (size_t i = 1; i < size; ++i) {
				if ((ptr[i-1] != delim) && (ptr[i] == delim)) ++count;
			}
			// check the last char to see if we we have a non-delim to null transition.
			count += (ptr[size - 1] != delim);
			
			// auto etime = getSysTime();
			// ROOT_PRINT("count tokens in %f sec\n", get_duration_s(stime, etime));
			return count;
		}

};
const char char_array_loop::EOL[] = "\r\n";

using char_array = char_array_stl;
}}