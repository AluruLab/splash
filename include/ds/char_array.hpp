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

#define COMMA ','
#define CR '\r'
#define EOL "\r\n"
#define LF '\n'
#define TAB '\t'

// type deduction of D with func(D delim) would result in D="const char*" when delim is declared "const char delim[]"
// to get D="const char (&)[N]", parameter should be (D & delim) or (D const & delim)
// since we may use literals, we need to use (D const & delim)
// see https://stackoverflow.com/questions/63371807/why-is-type-deduction-on-const-char-different-to-const-char

// use stl find, find_if, find_if_not, and adjacent_find
// performance: 
//	g++: 		get matrix size in 0.145s, load fist line in 0.000016s. load values in 0.141 s.  total 0.290s
// 	clang++: 	get matrix size in 0.0575s, load fist line in 0.000003s. load values in 0.076 s.  total 0.135s
class char_array_stl {
	// strtok modifies the buffer in memory.  avoid.

	public:
		static constexpr size_t EMPTY = 0;

		char * ptr;
		size_t size;

	protected:
		template <size_t N>
		inline bool is_delimiter(const char (&delim)[N], const char & c) const {
			return (std::find(delim, delim + N, c) != delim + N);
		}
		inline bool is_delimiter(const char & delim, const char & c) const {
			return c == delim;
		}
		template <size_t N>
		inline bool is_token_end(const char (&delim)[N], const char & x, const char & y) const {
			bool x_is_not = std::find(delim, delim + N, x) == delim + N;
			bool y_is = std::find(delim, delim + N, y) != delim + N;
			return x_is_not && y_is;
		}
		inline bool is_token_end(const char & delim, const char & x, const char & y) const {
			return (x != delim) && (y == delim);
		}


	public:
		// search buffer to get to the first non-delim character, and return size.
		template <typename D>
		char_array_stl trim_left(D const & delim) {
			// auto stime = getSysTime();	

			char_array_stl output = {ptr, EMPTY};
			if ((size == EMPTY) || (ptr == nullptr)) return output;  // buffer is empty, short circuit.
			
			// find first entry that is NOT in delim
			ptr = std::find_if_not(ptr, ptr + size,
				[this, &delim](char const & c) {
					return this->is_delimiter(delim, c);
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
		char_array_stl extract_token(D const & delim) {
			// auto stime = getSysTime();	
			if ((size == EMPTY) || (ptr == nullptr)) return char_array_stl{nullptr, EMPTY};  // buffer is empty, short circuit.
			char_array_stl output = {ptr, EMPTY};
			
			// search for first match to delim
			ptr = std::find_if(ptr, ptr + size,
				[this, &delim](char const & c){
					return this->is_delimiter(delim, c);
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
		char_array_stl trim_left_1(D const & delim) {
			// auto stime = getSysTime();	

			char_array_stl output = {ptr, EMPTY};
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
		char_array_stl trim_left_1(const char & delim) {
			// auto stime = getSysTime();	
			char_array_stl output = {ptr, EMPTY};
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
		template <typename D>
		char_array_stl get_token_or_empty(D const & delim) {
			char_array_stl output = extract_token(delim);
			trim_left_1(delim);
			return output;
		}
		// return token (ptr and length). treats consecutive delimiters as one.
		template <typename D>
		char_array_stl get_token(D const & delim) {
			char_array_stl output = extract_token(delim);
			trim_left(delim);
			return output;
		}		

		template <typename D>
		size_t count_token_or_empty(D const & delim) const {
			// auto stime = getSysTime();
			if ((size == EMPTY) || (ptr == nullptr)) return 0;

			// now count delimiters
			size_t count = std::count_if(ptr, ptr + size,
				[this, &delim](const char & c){
					return this->is_delimiter(delim, c);
				}) + 1;

			// auto etime = getSysTime();
			// ROOT_PRINT("count tokens or empty in %f sec\n", get_duration_s(stime, etime));
			return count;
		}
		// count non-empty tokens.
		template <typename D>
		size_t count_token(D const & delim) const {
			// auto stime = getSysTime();
			if ((size == EMPTY) || (ptr == nullptr)) return 0;

			size_t count = 0;

			// now count non-delimiter to delimiter transitions
			auto start = ptr;
			auto end = ptr + size;
			while ((start = std::adjacent_find(start, end, 
								[this, &delim](const char & x, const char & y){
									return this->is_token_end(delim, x, y);
								}))
					!= end) {
				// found a transition
				++start; // go to next char
				++count; // increment count
			}

			// check the last char to see if we we have a non-delim to null transition.
			count += !(is_delimiter(delim, ptr[size - 1]));
			
			// auto etime = getSysTime();
			// ROOT_PRINT("count tokens in %f sec\n", get_duration_s(stime, etime));
			return count;
		}


};


// comparison kernels use loops
// performance: 
//	g++: 		get matrix size in 0.0234s, load fist line in 0.000004s. load values in 0.0867 s.  total 0.112s
// 	clang++: 	get matrix size in 0.0211s, load fist line in 0.000003s. load values in 0.0672 s.  total 0.091s
class char_array_hybrid {
	// strtok modifies the buffer in memory.  avoid.

	public:
		static constexpr size_t EMPTY = 0;

		char * ptr;
		size_t size;

	protected:
		template <size_t N>
		inline bool is_delimiter(const char (&delim)[N], const char & c) const {
			for (size_t i = 0; i < N; ++i) {
				if (delim[i] == c) return true;
			}
			return false;
		}
		inline bool is_delimiter(const char & delim, const char & c) const {
			return c == delim;
		}
		template <size_t N>
		inline bool is_token_end(const char (&delim)[N], const char & x, const char & y) const {
			bool x_is_not = (x != delim[0]);
			bool y_is = (y == delim[0]);
			for (size_t i = 1; i < N; ++i) {
				x_is_not &= (x != delim[i]);
				y_is |= (y == delim[i]);
			}
			return x_is_not && y_is;
		}
		inline bool is_token_end(const char & delim, const char & x, const char & y) const {
			return (x != delim) && (y == delim);
		}

	public:
		// search buffer to get to the first non-delim character, and return size.
		template <typename D>
		char_array_hybrid trim_left(D const & delim) {
			// auto stime = getSysTime();	

			char_array_hybrid output = {ptr, EMPTY};
			if ((size == EMPTY) || (ptr == nullptr)) return output;  // buffer is empty, short circuit.
			
			// find first entry that is NOT in delim
			ptr = std::find_if_not(ptr, ptr + size,
				[this, &delim](char const & c) {
					return this->is_delimiter(delim, c);
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
		char_array_hybrid extract_token(D const & delim) {
			// auto stime = getSysTime();	
			if ((size == EMPTY) || (ptr == nullptr)) return char_array_hybrid{nullptr, EMPTY};  // buffer is empty, short circuit.
			char_array_hybrid output = {ptr, EMPTY};
			
			// search for first match to delim
			ptr = std::find_if(ptr, ptr + size,
				[this, &delim](char const & c){
					return this->is_delimiter(delim, c);
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
		char_array_hybrid trim_left_1(D const & delim) {
			// auto stime = getSysTime();	

			char_array_hybrid output = {ptr, EMPTY};
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
		char_array_hybrid trim_left_1(const char & delim) {
			// auto stime = getSysTime();	
			char_array_hybrid output = {ptr, EMPTY};
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
		template <typename D>
		char_array_hybrid get_token_or_empty(D const & delim) {
			char_array_hybrid output = extract_token(delim);
			trim_left_1(delim);
			return output;
		}
		// return token (ptr and length). treats consecutive delimiters as one.
		template <typename D>
		char_array_hybrid get_token(D const & delim) {
			char_array_hybrid output = extract_token(delim);
			trim_left(delim);
			return output;
		}		

		template <typename D>
		size_t count_token_or_empty(D const & delim) const {
			// auto stime = getSysTime();
			if ((size == EMPTY) || (ptr == nullptr)) return 0;

			// now count delimiters
			size_t count = std::count_if(ptr, ptr + size,
				[this, &delim](const char & c){
					return this->is_delimiter(delim, c);
				}) + 1;

			// auto etime = getSysTime();
			// ROOT_PRINT("count tokens or empty in %f sec\n", get_duration_s(stime, etime));
			return count;
		}
		// count non-empty tokens.
		template <typename D>
		size_t count_token(D const & delim) const {
			// auto stime = getSysTime();
			if ((size == EMPTY) || (ptr == nullptr)) return 0;

			size_t count = 0;

			// now count non-delimiter to delimiter transitions
			auto start = ptr;
			auto end = ptr + size;
			while ((start = std::adjacent_find(start, end, 
								[this, &delim](const char & x, const char & y){
									return this->is_token_end(delim, x, y);
								}))
					!= end) {
				// found a transition
				++start; // go to next char
				++count; // increment count
			}

			// check the last char to see if we we have a non-delim to null transition.
			count += !(is_delimiter(delim, ptr[size - 1]));
			
			// auto etime = getSysTime();
			// ROOT_PRINT("count tokens in %f sec\n", get_duration_s(stime, etime));
			return count;
		}

};

// using raw loops for outer loop and inner kernels, no lambdas.
// performance: 
//	g++: 		get matrix size in 0.0203s, load fist line in 0.000003s. load values in 0.0893 s.  total 0.112s
// 	clang++: 	get matrix size in 0.0348s, load fist line in 0.000004s. load values in 0.0715 s.  total 0.108s
class char_array_loop {
	// strtok modifies the buffer in memory.  avoid.

	public:
		static constexpr size_t EMPTY = 0;

		char * ptr;
		size_t size;

	protected:
		template <size_t N>
		inline bool is_delimiter(const char (&delim)[N], const char & c) const {
			for (size_t i = 0; i < N; ++i) {
				if (delim[i] == c) return true;
			}
			return false;
		}
		inline bool is_delimiter(const char & delim, const char & c) const {
			return c == delim;
		}
		template <size_t N>
		inline bool is_token_end(const char (&delim)[N], const char & x, const char & y) const {
			bool x_is_not = (x != delim[0]);
			bool y_is = (y == delim[0]);
			for (size_t i = 1; i < N; ++i) {
				x_is_not &= (x != delim[i]);
				y_is |= (y == delim[i]);
			}
			return x_is_not && y_is;
		}
		inline bool is_token_end(const char & delim, const char & x, const char & y) const {
			return (x != delim) && (y == delim);
		}

	public:
		// search buffer to get to the first non-delim character, and return size.
		template <typename D>
		char_array_loop trim_left(D const & delim) {
			// auto stime = getSysTime();	

			char_array_loop output = {ptr, EMPTY};
			if ((size == EMPTY) || (ptr == nullptr)) return output;  // buffer is empty, short circuit.
			
			// find first entry that is NOT in delim
			size_t i = 0;
			for (; (i < size) && is_delimiter(delim, ptr[i]); ++i);
			ptr += i;
			size -= i;
			output.size = i;

			// auto etime = getSysTime();
			// ROOT_PRINT("trim_left x in %f sec\n", get_duration_s(stime, etime));
			return output;
		}

		// return token (ptr and length). does not treat consecutive delimiters as one.
		// assumes buffer points to start of token.
		// if there is no more token, then return nullptr
		template <typename D>
		char_array_loop extract_token(D const & delim) {
			// auto stime = getSysTime();	
			if ((size == EMPTY) || (ptr == nullptr)) return char_array_loop{nullptr, EMPTY};  // buffer is empty, short circuit.
			char_array_loop output = {ptr, EMPTY};
			
			// search for first match to delim
			size_t i = 0;
			for (; (i < size) && !is_delimiter(delim, ptr[i]); ++i);
			ptr += i;
			size -= i;
			output.size = i;

			// printf("extract_token: %lu\n", output.size);
			// auto etime = getSysTime();	
			// ROOT_PRINT("extract token x in %f sec\n", get_duration_s(stime, etime));
			return output;
		}
		// remove 1 delim character from buffer (or \r\n)
		template <typename D>
		char_array_loop trim_left_1(D const & delim) {
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
		template <typename D>
		char_array_loop get_token_or_empty(D const & delim) {
			char_array_loop output = extract_token(delim);
			trim_left_1(delim);
			return output;
		}
		// return token (ptr and length). treats consecutive delimiters as one.
		template <typename D>
		char_array_loop get_token(D const & delim) {
			char_array_loop output = extract_token(delim);
			trim_left(delim);
			return output;
		}		

		template <typename D>
		size_t count_token_or_empty(D const & delim) const {
			// auto stime = getSysTime();
			if ((size == EMPTY) || (ptr == nullptr)) return 0;

			// now count delimiters
			size_t count = 1;
			for (size_t i = 0; i < size; ++i) 
				if (is_delimiter(delim, ptr[i])) ++count;

			// auto etime = getSysTime();
			// ROOT_PRINT("count tokens or empty in %f sec\n", get_duration_s(stime, etime));
			return count;
		}
		// count non-empty tokens.
		template <typename D>
		size_t count_token(D const & delim) const {
			// auto stime = getSysTime();
			if ((size == EMPTY) || (ptr == nullptr)) return 0;

			size_t count = 0;
			// now count non-delimiter to delimiter transitions
			for (size_t i = 1; i < size; ++i)
				if ( is_token_end(delim, ptr[i-1], ptr[i]) ) ++count;
			// check the last char to see if we we have a non-delim to null transition.
			count += !(is_delimiter(delim, ptr[size - 1]));
			
			// auto etime = getSysTime();
			// ROOT_PRINT("count tokens in %f sec\n", get_duration_s(stime, etime));
			return count;
		}

};


// version with templated delimiters.  Goal is avoid iteration of delim, at cost of no delimiter strings.
// ASSUME "\r\n" always together, and are essentially treated as a single char.
// performance: 
//	g++: 		get matrix size in 0.0140s, load fist line in 0.000003s. load values in 0.0840 s.  total 0.0998s
// 	clang++: 	get matrix size in 0.0149s, load fist line in 0.000003s. load values in 0.0780 s.  total 0.0948s
class char_array_template {
	// strtok modifies the buffer in memory.  avoid

	public:
		static constexpr size_t EMPTY = 0;

		char * ptr;
		size_t size;

	protected:

		template <char delim, typename = void>
		inline bool is_delimiter(size_t const & x) const {
			return ptr[x] == delim;
		}
		// assumption:  \r\n always together.
		template <char delim, typename std::enable_if<delim == LF>::type>
		inline bool is_delimiter(size_t const & x) const {
			return (ptr[x] == LF) || (ptr[x] == CR);
		}
		template <char delim, typename = void>
		inline bool is_not_delimiter(size_t const & x) const {
			return ptr[x] != delim;
		}
		// assumption:  \r\n always together.
		template <char delim, typename std::enable_if<delim == LF>::type>
		inline bool is_not_delimiter(size_t const & x) const {
			return (ptr[x] != LF) && (ptr[x] != CR); 
		}
		template <char delim, typename = int>
		inline size_t& advance(size_t & x) const {
			return ++x;
		}
		// assumption:  \r\n always together.
		template <char delim, typename std::enable_if<delim == LF>::type>
		inline size_t& advance(size_t & x) const {
			if (ptr[x] == CR) x += 2;
			else ++x;
			return x;
		}
		template <char delim>
		inline bool is_token_end(const size_t & x, const size_t & y) const {
			return is_not_delimiter<delim>(x) && is_delimiter<delim>(y);
		}

	public:
		// search buffer to get to the first non-delim character, and return size.
		template <char delim>
		char_array_template trim_left() {
			// auto stime = getSysTime();

			char_array_template output = {ptr, EMPTY};
			if ((size == EMPTY) || (ptr == nullptr)) return output;  // buffer is empty, short circuit.
			
			// find first entry that is NOT in delim
			size_t i = 0;
			for (; (i < size) && is_delimiter<delim>(i); advance<delim>(i));
			i = std::min(i, size);
			ptr += i;
			size -= i;
			output.size = i;

			// auto etime = getSysTime();
			// ROOT_PRINT("trim_left x in %f sec\n", get_duration_s(stime, etime));
			return output;
		}



		// return token (ptr and length). does not treat consecutive delimiters as one.
		// assumes buffer points to start of token.
		// if there is no more token, then return nullptr
		template <char delim>
		char_array_template extract_token() {
			// auto stime = getSysTime();	
			if ((size == EMPTY) || (ptr == nullptr)) return char_array_template{nullptr, EMPTY};  // buffer is empty, short circuit.
			char_array_template output = {ptr, EMPTY};
			
			// search for first match to delim
			size_t i = 0;
			for (; (i < size) && !is_delimiter<delim>(i); advance<delim>(i));
			ptr += i;
			size -= i;
			output.size = i;

			// printf("extract_token: %lu\n", output.size);
			// auto etime = getSysTime();	
			// ROOT_PRINT("extract token x in %f sec\n", get_duration_s(stime, etime));
			return output;
		}

		// remove 1 delim character from buffer (or \r\n)
		template <char delim>
		char_array_template trim_left_1() {
			// auto stime = getSysTime();	

			char_array_template output = {ptr, EMPTY};
			if ((size == EMPTY) || (ptr == nullptr)) return output;  // buffer is empty, short circuit.
			
			// check if CR and in delim.  if yes, skip.
			size_t i = 0;
			if (is_delimiter<delim>(i)) {
				advance<delim>(i);
				output.size = std::min(size, i);
				size -= output.size;
				ptr += output.size;
			}
			// printf("trim_left_1: %lu\n", output.size);
			// auto etime = getSysTime();	
			// ROOT_PRINT("trim_left_1 x in %f sec\n", get_duration_s(stime, etime));
			return output;
		}

		// get a token. does not treat consecutive delimiters as one so output may be empty.
		template <char delim>
		char_array_template get_token_or_empty() {
			char_array_template output = extract_token<delim>();
			trim_left_1<delim>();
			return output;
		}
		// return token (ptr and length). treats consecutive delimiters as one.
		template <char delim>
		char_array_template get_token() {
			char_array_template output = extract_token<delim>();
			trim_left<delim>();
			return output;
		}		

		template <char delim>
		size_t count_token_or_empty() const {
			// auto stime = getSysTime();
			if ((size == EMPTY) || (ptr == nullptr)) return 0;

			// now count delimiters
			size_t count = 1;
			for (size_t i = 0; i < size; advance<delim>(i)) 
				if (is_delimiter<delim>(i)) ++count;

			// auto etime = getSysTime();
			// ROOT_PRINT("count tokens or empty in %f sec\n", get_duration_s(stime, etime));
			return count;
		}
		// count non-empty tokens.
		template <char delim>
		size_t count_token() const {
			// auto stime = getSysTime();
			if ((size == EMPTY) || (ptr == nullptr)) return 0;

			size_t count = 0;
			// now count non-delimiter to delimiter transitions
			for (size_t i = 1; i < size; advance<delim>(i))
				if ( is_token_end<delim>(i-1, i) ) ++count;
			// check the last char to see if we we have a non-delim to null transition.
			count += is_not_delimiter<delim>(size - 1);
			
			// auto etime = getSysTime();
			// ROOT_PRINT("count tokens in %f sec\n", get_duration_s(stime, etime));
			return count;
		}

};


using char_array = char_array_hybrid;
}}