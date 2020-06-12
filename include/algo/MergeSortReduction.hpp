/*
 *  MergeSortReduction.hpp
 *
 *  Created on: June 12, 2020
 *  Author: Tony C. Pan
 *  Affiliation: School of Computational Science & Engineering
 *  						Georgia Institute of Technology, Atlanta, GA 30332
 */

#pragma once


#include <xmmintrin.h>

#define DC_MERGESORT_ASCEND_CONCORDANT  0
#define DC_MERGESORT_DESC_CONCORDANT  1
#define DC_MERGESORT_ASCEND_DISCORDANT  2


// TODO: hybrid insertion-merge sort.   size:  at most 32, or k * (64/datasize), where k is the set associativity.  
//      binary search may not help as it is non-linear access to memory and cachelines.

// compute reduction via Modified MergeSort.  example of reduction include partial sum, inversion count, etc.
// useful for kendall's tau and distance correlation, and possibly others.
// relies on ElemType providing the following operations:
//    operator=(ElemType const &)
//    init(ElemType &):  initialize 1 element.
//    operator<(ElemType const &): sort comparison operator.
//    scan_update(ScanElemType const &, ElemType const &): for inclusive scan using plus operator.
//    merge_update(ElemType const &, ElemType const &, ElemType &): for updating the inverted elements during merge. (otherwise merge is just selecting left and right.)
template <typename ElemType, typename SortElemType>
class MergeSortAndReduce {

    protected:
        size_t vecSize;
        SortElemType * buffer;
        ElemType * prefix;

        int variant;
    public:
        MergeSortAndReduce(size_t const & count, int const & _variant) : vecSize(count), variant(_variant) {
            buffer = reinterpret_cast<SortElemType*>(_mm_malloc(count * sizeof(SortElemType), 64));
            prefix = reinterpret_cast<ElemType*>(_mm_malloc((count+1) * sizeof(ElemType), 64));
        }
        ~MergeSortAndReduce() {
            _mm_free(buffer);
            _mm_free(prefix);
        }

    protected:
        // during each iteration, we only need the lower half (2^iter) of each segment.
        inline void scan(SortElemType const * data, size_t const & count, 
            size_t const & gap = std::numeric_limits<size_t>::max()) {
            memset(prefix, 0, (count + 1) * sizeof(ElemType));
            ElemType *curr = prefix + 1;

            if (gap >= count) {
                // continuous.  all are summed.
                for (size_t i = 0; i < count; ++i) {
                    curr[i] = prefix[i] + data[i];
                    // curr[i].count = i + 1;
                }
            } else if (gap == 1) {  // 1 prefix element accessed per iter..
                // every other.  so same as copy every other.
                for (size_t i = 0; i < count; i += 2) {
                    curr[i] = data[i];
                    // curr[i].count = 1;
                }
            } else if (gap == 2) {  // 3 prefix elements accessed per iter.
                size_t max = count & (std::numeric_limits<size_t>::max() << 1);
                size_t i = 0;
                for (; i < max; i += 4) {
                    curr[i] = prefix[i] + data[i];
                    // curr[i].count = 1;
                    curr[i+1] = prefix[i+1] + data[i+1];
                    // curr[i+1].count = 2;
                }
                // last bit.  since we mod by gap==2, there should be only 1 element possibly left.
                if (i < count) {
                    curr[i] = prefix[i] + data[i];
                    // curr[i].count = 1;
                }
            } else if ((gap & (gap-1)) == 0) { // at gap of 4, 5 elements of prefix are accessed per iter. use double loop
                size_t max = count & (~(gap - 1));
                size_t i = 0, j;
                for (; i < max; i += gap) {
                    for (j = 0; j < gap; ++j, ++i) {
                        curr[i] = prefix[i] + data[i];
                        // curr[i].count = j + 1;
                    }
                }
                for (j = 0; i < count; ++j, ++i) {  // since max is masked from gap-1, if i < count, left over is < gap.
                    curr[i] = prefix[i] + data[i];
                    // curr[i].count = j + 1;
                }
            } else {
                fprintf(stderr, "ERROR: NON-POW2 Gap detected for scan.");
            }
        }

        inline size_t merge_ascending_concordant(SortElemType const * in,
			size_t const & start, size_t const & middle, size_t const & end,
			SortElemType * out, bool print = false) {

            // prefix is exclusive, with size count+1.
            // start, middle, and end are indices for in, and starts with 0.
            // st0 is the element just before st1 that is not the same as yj
            size_t k = start;
			size_t st1 = start, st2 = middle;
            ElemType pre; // empty.  // prefix sum of all left element that are strictly smaller.
            ElemType curr; // empty.


            size_t swap_count = 0;
            // size_t gap = end - start;
            // size_t block = k / gap;

            // fprintf(stderr, "range: %lu, %lu, %lu\n", start, middle, end);

            SortElemType last = in[st1];   // initialize to first element.
            // this will leave pre as empty for the first element:  if in[st1] > in[st2], last < in[st2] is not going to be true.  else: no action due to last.
            //      then last will be updated.  important part is pre is not updated to curr for first call, and will remain empty.  
            // if (st2 == end)
            //     SortElemType last = (in[st1] <= in[st2]) ? in[st1] : in[st2];

            // size_t L = 0;
			for (; (st1 < middle) && (st2 < end); ++k) {
                if (in[st2] < in[st1]) { // Right smaller, update d
                    // if we have consecutive Right side selection after a L/R split equal stretch,
                    // pre may not be updated to include the first L that is equal to the first R.
                    // subsequent R selections, even if larger than first R, still use the same pre.
                    // pre needs to be updated.
                    if (last < in[st2]) {  // update pre when right side is out of an equal stretch.
                        pre = curr;                 // merge_less is < if ties are excluded, <= if ties on L are included.
                    }

					out[k] = in[st2];

                    out[k].merge_update(pre);  // uses exscan
                    swap_count += middle - st1;

                    // if (print) fprintf(stderr, "ASCD, %lu, %lu, %lu, R, %lu, %f, %ld, %ld\n", gap, block, k, st2, out[k].key, static_cast<long>(pre.count), static_cast<long>(out[k].concCount));
					++st2;
                } else { // Left smaller or equal
                    if (last < in[st1]) {  // sets current partial sum when L is smaller.
                        pre = curr;  // if strictly smaller, then move the prefix sum pointer up.  exscan for next step
                    }  // but this does not account for when right side exists the equal stretch.

					out[k] = in[st1];  // select left.

                    // this computes the current prefix.
                    curr += in[st1];   // update current prefix sum
                    // curr.count = ++L;  // increment the left count, including self, and overwrite the current count = L+in[st1].count
                    
                    // if (print) fprintf(stderr, "ASCD, %lu, %lu, %lu, L, %lu, %f, -1, %ld\n", gap, block, k, st1, out[k].key, static_cast<long>(out[k].concCount));
    				++st1;
                }
                last = out[k];                        

			}
            
            // ==== clean up at the end.  only one of 2 will proceed.  pre will not be updated.
            // left side unfinished.
            for (; st1 < middle; ++st1, ++k) {
                out[k] = in[st1];
                // none in LEFT, so don't need to update curr for merge_update.

                    // if (print) fprintf(stderr, "ASCD, %lu, %lu, %lu, LE, %lu, %f, -1, %ld\n", gap, block, k, st1, out[k].key, static_cast<long>(out[k].concCount));
            }
            // right side unfinished.  use the strictly-smaller_element prefix sum.
            for (; st2 < end; ++st2, ++k) {
                // none in right.  in[st2] is either equal or larger.  if equal, pre would be sum up to last unequal point.  if larger, pre needs to be updated to current.
                // adjust pre if last element is a repeat.
                if (last < in[st2])
                    pre = curr;

                out[k] = in[st2];
                out[k].merge_update(pre);
                // if (print) fprintf(stderr, "ASCD, %lu, %lu, %lu, RE, %lu, %f, %ld, %ld\n", gap, block, k, st2, out[k].key, static_cast<long>(pre.count), static_cast<long>(out[k].concCount));
            }

            return swap_count;
        }

        inline size_t merge_ascending_discordant(SortElemType const * in,
			size_t const & start, size_t const & middle, size_t const & end,
			SortElemType * out, bool print = false) {

            // suffix is exclusive, not counting equal part, and only the higher part of left half.
            // start, middle, and end are indices for in, and starts with 0.
            // st0 is the element just before st1 that is not the same as yj
            size_t k = start;
			size_t st1 = start, st2 = middle;
            size_t swap_count = 0;

            // prefix of entire left side.
            ElemType suffix; // prefix sum of all left element that are strictly smaller.
            for (; st1 < middle; ++st1) {
                suffix += in[st1]; 
            }
            st1 = start;

			for (; (st1 < middle) && (st2 < end); ++k) {
                if (in[st2] < in[st1]) { // Right smaller, update d

					out[k] = in[st2];
                    out[k].merge_update(suffix);    // we update with suffix, and exclude any ties on the left side.  ties on right side all receive the same suffix.
                    swap_count += (middle - st1);
                    // if (print) fprintf(stderr, "ASCD, %lu, %lu, %lu, R, %lu, %f, %ld, %ld\n", gap, block, k, st2, out[k].key, static_cast<long>(pre.count), static_cast<long>(out[k].concCount));
					++st2;
                } else { // Left smaller or equal
					out[k] = in[st1];  // select left.

                    // reduce the suffix by current.
                    suffix -= in[st1];   // update current suffix sum, including the ties.
                                        
                    // if (print) fprintf(stderr, "ASCD, %lu, %lu, %lu, L, %lu, %f, -1, %ld\n", gap, block, k, st1, out[k].key, static_cast<long>(out[k].concCount));
    				++st1;
                }
			}
            
            // ==== clean up at the end.  only one of 2 will proceed.  pre will not be updated.
            // left side unfinished.
            for (; st1 < middle; ++st1, ++k) {
                out[k] = in[st1];

                // none on the RIGht that requires merge_update.
                    // if (print) fprintf(stderr, "ASCD, %lu, %lu, %lu, LE, %lu, %f, -1, %ld\n", gap, block, k, st1, out[k].key, static_cast<long>(out[k].concCount));
            }
            // right side unfinished. 
            for (; st2 < end; ++st2, ++k) {
                out[k] = in[st2];
                // none on the LEFT  so RIGHT merge_update would be with 0.
                // if (print) fprintf(stderr, "ASCD, %lu, %lu, %lu, RE, %lu, %f, %ld, %ld\n", gap, block, k, st2, out[k].key, static_cast<long>(pre.count), static_cast<long>(out[k].concCount));
                // if (print) fprintf(stderr, "ASCD, %lu, RE, %lu, %f, %ld, %ld\n", k, st2, out[k].key, static_cast<long>(suffix.count), (middle - start));
            }

            return swap_count;
        }

        // descending, as in original data.
        inline size_t merge_descending_concordant(SortElemType const * in,
			size_t const & start, size_t const & middle, size_t const & end,
			SortElemType * out, bool print = false) {

            // prefix is exclusive, with size count+1.
            // start, middle, and end are indices for in, and starts with 0.
        
            size_t k = start;
			size_t st1 = start, st2 = middle;
            ElemType pre = prefix[middle];  // assume exclusive scan.  this then has sum of all on left, start.. middle-1
            ElemType sum;

            size_t R2L = 0;
            // size_t gap = end - start;
            // size_t block = k / gap;

			for (; (st1 < middle) && (st2 < end); ++k) {
				if (in[st1] < in[st2]) {
					out[k] = in[st2];
                    sum = pre - prefix[st1];
                    out[k].merge_update(sum);
                    // if (print) fprintf(stderr, "DESC, %lu, %lu, %lu, R, %lu, %f, %ld, %ld\n", gap, block, k, st2, out[k].key, static_cast<long>(sum.count), static_cast<long>(out[k].concCount));
					++st2;
                    ++R2L;
				} else {
					out[k] = in[st1];
                    // if (print) fprintf(stderr, "DESC, %lu, %lu, %lu, L, %lu, %f, -1, %ld\n", gap, block, k, st1, out[k].key, static_cast<long>(out[k].concCount));
					++st1;
                }
			}

			// size_t st=end, e=end;
			if (st2 == end) {
				// left side unfinished.
                for (; st1 < middle; ++st1, ++k) {
                    out[k] = in[st1];
                    // if (print) fprintf(stderr, "DESC, %lu, %lu, %lu, LE, %lu, %f, -1, %ld\n", gap, block, k, st1, out[k].key, static_cast<long>(out[k].concCount));
                }
				// st = st1;
				// e = middle;
			} else if (st1 == middle) {
				// right side unfinished.
                for (; st2 < end; ++st2, ++k) {
                    out[k] = in[st2];
                    // if (print) fprintf(stderr, "DESC, %lu, %lu, %lu, RE, %lu, %f, -1, %ld\n", gap, block, k, st2, out[k].key, static_cast<long>(out[k].concCount));
                }
				// st = st2;
				// e = end;
			}
			// for (; st < e; ++st, ++k) {
			// 	out[k] = in[st];
			// }
            return R2L;
        }


    public:
        inline void clear(SortElemType* data, size_t const & count) {
            memset(data, 0, count * sizeof(SortElemType));
        }
        inline void initialize(SortElemType* data, size_t const & count) {
            for (size_t i = 0; i < count; ++i) {
                data[i].init();
            }
        }
        
        // sorted_block_size is for later - when we integrate insertion sort for small size.
		void sort(SortElemType* data, size_t const & count, bool print=false, size_t sorted_block_size = 1) {
            
			// if an insertion sort step, do it here.
			this->initialize(data, count);
            this->clear(buffer, count);

			SortElemType* temp = data;
			SortElemType* temp2 = buffer;

			size_t gap;
			size_t start, middle, end;

            // if (print) fprintf(stderr, "variant, gap, block, outpos, half, inpos, key, update, conCount\n");


            if (variant == DC_MERGESORT_DESC_CONCORDANT) {
                for (size_t i = sorted_block_size; i < count; i *= 2) {
                    gap = i *2;

                    this->scan(temp, count);

                    for (size_t j = 0; j < count; j += gap) {
                        start = j;
                        middle = std::min(j+i, count);
                        end = std::min(j+gap, count);
                                            
                        this->merge_descending_concordant(
                            temp,
                            start, middle, end,
                            temp2, print);
                    }
                    std::swap(temp, temp2);  // switch the 2.  up-to-date result is now in temp again.
                }
                
            } else if (variant == DC_MERGESORT_ASCEND_CONCORDANT) {
                for (size_t i = sorted_block_size; i < count; i *= 2) {
                    gap = i * 2;

                    for (size_t j = 0; j < count; j += gap) {
                        start = j;
                        middle = std::min(j+i, count);
                        end = std::min(j+gap, count);
                                            
                        this->merge_ascending_concordant(
                            temp,
                            start, middle, end,
                            temp2, print);
                    }
                    std::swap(temp, temp2);  // switch the 2.  up-to-date result is now in temp again.
                }

            } else if (variant == DC_MERGESORT_ASCEND_DISCORDANT) {
                for (size_t i = sorted_block_size; i < count; i *= 2) {
                    gap = i * 2;

                    for (size_t j = 0; j < count; j += gap) {
                        start = j;
                        middle = std::min(j+i, count);
                        end = std::min(j+gap, count);
                                            
                        this->merge_ascending_discordant(
                            temp,
                            start, middle, end,
                            temp2, print);
                    }
                    std::swap(temp, temp2);  // switch the 2.  up-to-date result is now in temp again.
                }

            }



			if (temp2 == data) {  // result in temp but data is temp2.
				// copy
				memcpy(data, temp, sizeof(SortElemType) * count);

			} // else data already has results (in temp)
		}
};


