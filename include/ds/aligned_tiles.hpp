/* TODO:
 * [ ] unique():  remove redundant tiles.
 * [ ] reflect() = transpose + merge, but internally resize memory for speed.
 */

#pragma once

#include "utils/memory.hpp"
#include "utils/partition.hpp"
#include "ds/aligned_matrix.hpp"
#include <vector>
#include <utility>  // pair
#include <limits>
#include <algorithm>  // min/max

#include <cstring> // memset, memcpy

#include "utils/report.hpp"
#include "utils/benchmark.hpp"

#ifdef USE_MPI
#include <mpi.h>
#include "utils/mpi_types.hpp"
#endif
 
namespace splash { namespace ds { 

// sequential set of tiles, each is r_size x c_size
template <typename T, typename PARTITION>
class aligned_tiles;

// does not allow random access of tiles.
template <typename T, typename S>
class aligned_tiles<T, splash::utils::partition2D<S>> {

    public:
        using data_type = T;
        using offset_type = S;
        using id_type = typename splash::utils::partition2D<S>::id_type;
        using size_type = size_t;
    protected:
        size_t _align;  // alignment
        mutable T * _data;  // consists of count tiles, each r_size * c_size
        mutable std::vector<splash::utils::partition2D<S>> parts;
        mutable std::vector<size_type> offsets;  // size is 1 larger than number of parts, last entry contains total count

        mutable size_t bytes;

        struct sort_elem {
            S r_offset;
            S c_offset;
            size_t part_id;
            size_t ptr_offset;

            sort_elem() = default;
            sort_elem(S const & r, S const & c, size_t const & id, size_t const & off) :
                r_offset(r), c_offset(c), part_id(id), ptr_offset(off) {}
            sort_elem(sort_elem const & other) = default;
            sort_elem(sort_elem && other) = default;
            sort_elem& operator=(sort_elem const & other) = default;
            sort_elem& operator=(sort_elem && other) = default;
        };

        // only used by internal functions.
        aligned_tiles(size_type const & _count, size_type const & elements, size_t const & align = 0) :
            _align(align == 0 ? splash::utils::get_cacheline_size() : align), 
            parts(_count), offsets(_count + 1),
            bytes(elements * sizeof(T)) {

            _data = reinterpret_cast<T*>(splash::utils::aalloc(bytes, _align));
            memset(_data, 0, bytes);

            offsets[_count] = elements;
                // PRINT_RT("INTERNAL _data: %p, %lu\n", _data, sizeof(_data));
                // FLUSH();
        }

    public:
        aligned_tiles() : _align(splash::utils::get_cacheline_size()),
            _data(nullptr), bytes(0) {
                // PRINT_RT("DEFAULT _data: %p, %lu\n", _data, sizeof(_data));
                // FLUSH();
            }

        aligned_tiles(splash::utils::partition2D<S> const * start, size_type const & count, size_t const & align = 0) :
            _align(align == 0 ? splash::utils::get_cacheline_size() : align),
            parts(start, start + count) {

            size_type elements = 0;
            offsets.reserve(count + 1);
            for (auto part : parts) {
                offsets.emplace_back(elements);
                elements += part.r.size * part.c.size;
            }
            offsets.emplace_back(elements);
            bytes = elements * sizeof(T);

            _data = reinterpret_cast<T*>(splash::utils::aalloc(bytes, _align));
            memset(_data, 0, bytes);
                // PRINT_RT("POPULATE _data: %p, %lu\n", _data, sizeof(_data));
                // FLUSH();
        }
        ~aligned_tiles() {
                // PRINT_RT("DELETE _data: %p, %lu\n", _data, sizeof(_data));
                // FLUSH();
            if (_data) splash::utils::afree(_data);
        }

        aligned_tiles(aligned_tiles const & other) : _align(other._align), parts(other.parts), offsets(other.offsets) {
            _data = reinterpret_cast<T*>(splash::utils::aalloc(other.allocated(), _align));
            memcpy(_data, other._data, other.allocated());
            bytes = other.allocated();
                // PRINT_RT("COPY _data: %p, %lu\n", _data, sizeof(_data));
                // FLUSH();
        }
        aligned_tiles & operator=(aligned_tiles const & other) {
            if (allocated() != other.allocated()) {
                if (_data) splash::utils::afree(_data);
                _data = reinterpret_cast<T*>(splash::utils::aalloc(other.allocated(), _align));
            }
            if (_data != other._data) memcpy(_data, other._data, other.allocated() );
            bytes = other.bytes;

            _align = other._align;
            parts.assign(other.parts.begin(), other.parts.end());
            offsets.assign(other.offsets.begin(), other.offsets.end());
                // PRINT_RT("COPY= _data: %p, %lu\n", _data, sizeof(_data));
                // FLUSH();
            return *this;
        }
        // move constructor.  take ownership.
        aligned_tiles(aligned_tiles && other) : 
            _align(other._align), _data(other._data),
            parts(std::move(other.parts)), offsets(std::move(other.offsets)),
            bytes(other.bytes) {
                other._data = nullptr;
                other.bytes = 0;
                // PRINT_RT("MOVE _data: %p, %lu\n", _data, sizeof(_data));
                // FLUSH();
        }
        aligned_tiles & operator=(aligned_tiles && other) {
            _align = other._align;

            if (_data) splash::utils::afree(_data);
            _data = other._data;
            other._data = nullptr;

            parts = std::move(other.parts);
            offsets = std::move(other.offsets);
            
            bytes = other.bytes; 
            other.bytes = 0;
                // PRINT_RT("MOVE= _data: %p, %lu\n", _data, sizeof(_data));
                // FLUSH();
            return *this;
        }

        size_type size() const { return parts.size(); }
        size_type allocated() const { return bytes; }
        splash::utils::partition2D<size_type> const & part(size_t const & id) const { return parts[id]; };
        splash::utils::partition2D<size_type>& part(size_t const & id) { return parts[id]; };
        T const * data(size_t const & id) const { return _data + offsets[id]; }
        T* data(size_t const & id) { return _data + offsets[id]; }
        void clear_data() { memset(_data, 0, bytes); };

    protected:
        bool is_sorted_by_offsets() const  {
            return std::is_sorted(parts.begin(), parts.end(), 
                [](splash::utils::partition2D<size_type> const & a, 
                    splash::utils::partition2D<size_type> const & b){
                return (a.r.offset == b.r.offset) ? (a.c.offset < b.c.offset) : (a.r.offset < b.r.offset);
            });
        }

    public:

        aligned_tiles sort_by_offsets() const {
            if (is_sorted_by_offsets()) return *this;

            auto stime = getSysTime();
            // ====== first sort.
            std::vector<sort_elem> ids;
            ids.reserve(parts.size());

            for (size_t i = 0; i < parts.size(); ++i) {
                ids.emplace_back(parts[i].r.offset, parts[i].c.offset, i, offsets[i] );
            }
            auto etime = getSysTime();
            PRINT_RT("sort: set up in %f sec\n", get_duration_s(stime, etime));

            stime = getSysTime();

            // sort the partitions
            std::stable_sort(ids.begin(), ids.end(), [](sort_elem const & a, sort_elem const & b){
                return (a.r_offset == b.r_offset) ? (a.c_offset < b.c_offset) : (a.r_offset < b.r_offset);
            });
            etime = getSysTime();
            PRINT_RT("sort: sorted_ids in %f sec\n", get_duration_s(stime, etime));

            stime = getSysTime();
            
            // ======= now copy with reorder
            // PRINT_RT("aligned_tiles SORT ");
            aligned_tiles output(ids.size(), offsets.back(), _align);
            // PRINT_RT("aligned_tiles SORT DONE\n");
            // FLUSH();
            etime = getSysTime();
            PRINT_RT("sort: size_output in %f sec\n", get_duration_s(stime, etime));

            stime = getSysTime();
            
            size_type offset = 0;
            size_type s = 0;
            splash::utils::partition2D<size_type> part;
            for (size_t i = 0; i < ids.size(); ++i) {
                part = parts[ids[i].part_id];
                output.parts[i] = part;
                output.offsets[i] = offset;
                s = part.r.size * part.c.size;
                memcpy(output._data + offset, _data + ids[i].ptr_offset, s * sizeof(T));
                offset += s;
            }
            etime = getSysTime();
            PRINT_RT("sort: shuffled in %f sec\n", get_duration_s(stime, etime));

            return output;
        }
        
    protected:
        inline size_type transpose_tile(T const* orig, 
            size_type const & orig_rows, 
            size_type const & orig_cols,
            T * transposed) const {
            T const * ptr = orig;
            T * nptr = transposed;
            for (size_type c = 0; c < orig_cols; ++c) {
                ptr = orig + c;
                for (size_type r = 0; r < orig_rows; ++r, 
                    ++nptr, ptr += orig_cols) {
                    *nptr = *ptr;
                }
            }

            return orig_rows * orig_cols;
        }

    public:

        // transpose.  offsets don't change.  id and id_cols do not change.  r and c are swapped.
        aligned_tiles transpose() const {
            // PRINT_RT("aligned_tiles TRANSPOSE ");
            aligned_tiles output(parts.data(), parts.size(), _align);
            for (size_type i = 0; i < output.parts.size(); ++i) {
                output.parts[i].r = parts[i].c;
                output.parts[i].c = parts[i].r;
                
                transpose_tile(_data + offsets[i], 
                    parts[i].r.size, parts[i].c.size, 
                    output._data + output.offsets[i]);
            }

            return output;
        }

        aligned_tiles merge(aligned_tiles const & other) const {
            // PRINT_RT("aligned_tiles OPERATOR+ ");

            aligned_tiles output(parts.size() + other.parts.size(), offsets.back() + other.offsets.back(), _align);
            // PRINT_RT("left, parts: %ld, elements: %ld\n", size(), offsets.back());
            // PRINT_RT("right, parts: %ld, elements: %ld\n", other.size(), other.offsets.back());
            size_t i = 0;
            for (; i < parts.size(); ++i) {
                output.parts[i] = parts[i];
                output.offsets[i] = offsets[i];
            }
            memcpy(output._data, _data, allocated() );

            size_type offset = offsets.back();
            for (size_t j = 0; j < other.parts.size(); ++j, ++i) {
                output.parts[i] = other.parts[j];
                output.offsets[i] = offset + other.offsets[j];
            }
            memcpy(output._data + offsets.back(), other._data, other.allocated());
            // PRINT_RT("addition, parts: %ld, elements: %ld\n", output.size(), output.offsets.back());

            return output;
        }

        // transpose.  offsets don't change.  id and id_cols do not change.  r and c are swapped.
        aligned_tiles reflect_diagonally() const {
            // first count non-diagonal.
            size_t non_diag = 0;
            size_t non_diag_elements = 0;
            size_t i = 0;
            for (i = 0; i < parts.size(); ++i)  {
                if (parts[i].r.offset != parts[i].c.offset) {
                    ++non_diag;
                    non_diag_elements += parts[i].r.size * parts[i].c.size;
                }
            }

            // allocate
            aligned_tiles output(parts.size() + non_diag, offsets.back() + non_diag_elements, _align);
            
            // copy over the old.
            memcpy(output.parts.data(), parts.data(), sizeof(splash::utils::partition2D<S>) * parts.size());
            memcpy(output.offsets.data(), offsets.data(), sizeof(size_t) * parts.size());
            memcpy(output._data, _data, allocated() );

            // copy the rest and transpose.
            size_t j = 0;
            size_t off = offsets.back();
            for (j = 0; j < parts.size(); ++j)  {
                if (parts[j].r.offset != parts[j].c.offset) {
                    output.parts[i].r = parts[j].c;
                    output.parts[i].c = parts[j].r;
                    output.offsets[i] = off;
                    
                    transpose_tile(_data + offsets[j], 
                        parts[j].r.size, parts[j].c.size, 
                        output._data + off);

                    off += parts[j].r.size * parts[j].c.size;
                    ++i;
                }
            }

            return output;
        }


        splash::utils::partition2D<S> get_bounds() {
            if (parts.size() > 0) {
                S rmin = std::numeric_limits<S>::max();
                S cmin = std::numeric_limits<S>::max();
                S rmax = std::numeric_limits<S>::lowest();
                S cmax = std::numeric_limits<S>::lowest();
                
                for (size_t i = 0; i < parts.size(); ++i) {
                    rmin = std::min(rmin, parts[i].r.offset);
                    cmin = std::min(cmin, parts[i].c.offset);
                    rmax = std::max(rmax, parts[i].r.offset + parts[i].r.size);
                    cmax = std::max(cmax, parts[i].c.offset + parts[i].c.size);
                }
                // PRINT_RT("rmin %lu rmax %lu, cmin %lu cmax %lu\n", rmin, rmax, cmin, cmax);
                return splash::utils::partition2D<S>(
                    splash::utils::partition<S>(rmin, rmax-rmin, 0),
                    splash::utils::partition<S>(cmin, cmax-cmin, 0),
                    0, 1);
            } else { 
                return splash::utils::partition2D<S>(); 
            }
        }

        // fill dense matrix
        void copy_to(aligned_matrix<T> & matrix, size_t const & row_offset, size_t const & col_offset, size_t const & i) const {
            // every part.
            auto part = parts[i];
            // PRINT_RT("COPY_TO tile %lu: ", i);
            // print("COPY TO: ", i);
            T const * src = _data + offsets[i];  // src block of data
            if ((part.r.offset < row_offset) || (part.r.offset + part.r.size > row_offset + matrix.rows())) {
                PRINT_RT("row offset: %lu size %lu ", row_offset, matrix.rows());
                part.r.print("row: ");
            }
            if ((part.c.offset < col_offset) || (part.c.offset + part.c.size > col_offset + matrix.columns())) {
                PRINT_RT("col offset: %lu size %lu ", col_offset, matrix.columns());
                part.c.print("col: ");
            }
            T * dest; 
            size_t row = part.r.offset - row_offset;
            size_t col = part.c.offset - col_offset;

            for (size_type r = 0; r < part.r.size; ++r,
                src += part.c.size, ++row) {
                dest = matrix.data(row, col);  // dest block of data.
                // every row.
                memcpy(dest, src, part.c.size * sizeof(T));
            }               
        }

        void copy_to(aligned_matrix<T> & matrix, size_t const & row_offset, size_t const & col_offset) const {
            for (size_t i = 0; i < parts.size(); ++i) {
                this->copy_to(matrix, row_offset, col_offset, i);
            }
        }

        template <typename TT = T, typename std::enable_if<std::is_arithmetic<TT>::value, int>::type = 1>
        inline void print(const char* prefix, size_t const & id) const {
            TT const *  ptr = _data + offsets[id];;

            parts[id].print(prefix);
            PRINT_RT("%s data offsets = %lu\n", prefix, offsets[id]);
            for (size_t r = 0; r < parts[id].r.size; ++r) {
                PRINT_RT("%s ", prefix);
                for (size_t c = 0; c < parts[id].c.size; ++c, ++ptr) {
                    PRINT("%.17lf, ", *ptr);
                }
                PRINT("\n");
            }
        }
        template <typename TT = T, typename std::enable_if<!std::is_arithmetic<TT>::value, int>::type = 1>
        inline void print(const char* prefix, size_t const & id) const {
            TT const *  ptr = _data + offsets[id];;

            parts[id].print(prefix);
            PRINT_RT("%s data offsets = %lu\n", prefix, offsets[id]);
            for (size_t r = 0; r < parts[id].r.size; ++r) {
                PRINT_RT("%s ", prefix);
                for (size_t c = 0; c < parts[id].c.size; ++c, ++ptr) {
                    ptr->print();
                    PRINT(", ");
                }
                PRINT("\n");
            }
        }

        void print(const char * prefix) const {
            for (size_t i = 0; i < parts.size(); ++i)  {
                this->print<T>(prefix, i);
            }
        }

        // partition by row, for MPI.  based on offsets only.
#ifdef USE_MPI
        aligned_tiles gather(int const & target_rank, MPI_Comm comm = MPI_COMM_WORLD ) const {

            int procs;
            int rank;
            MPI_Comm_size(comm, &procs);
            MPI_Comm_rank(comm, &rank);

            if (procs == 1) return *this;

            // get count of partitions
            int * part_counts = nullptr;
            int * elem_counts = nullptr;
            int nparts = parts.size();
            int elems = offsets.back();
            if (rank == target_rank) {
                part_counts = new int[procs] {};
                elem_counts = new int[procs] {};
            }
            splash::utils::mpi::datatype<int> int_dt;
            MPI_Gather(&nparts, 1, int_dt.value, part_counts, 1, int_dt.value, target_rank, comm);
            MPI_Gather(&elems, 1, int_dt.value, elem_counts, 1, int_dt.value, target_rank, comm);
            int *part_offsets = nullptr;
            int *elem_offsets = nullptr;
            if (rank == target_rank) {
                part_offsets = new int[procs + 1];
                elem_offsets = new int[procs + 1];
                part_offsets[0] = 0;
                elem_offsets[0] = 0;
                for (int i = 0; i < procs; ++i) {
                    part_offsets[i + 1] = part_offsets[i] + part_counts[i];
                    elem_offsets[i + 1] = elem_offsets[i] + elem_counts[i];
                }
            }

            // allocate output
            // PRINT_RT("aligned_tiles GATHER ");
            aligned_tiles output;
            if (rank == target_rank) {  
                // PRINT_RT("aligned_tiles GATHER assign");
                output = std::move(aligned_tiles(part_offsets[procs], elem_offsets[procs], _align));
            }
            // -------- move parts by bytes
            splash::utils::mpi::datatype<splash::utils::partition2D<S>> part2d_dt;
            MPI_Gatherv(parts.data(), nparts, part2d_dt.value,
                output.parts.data(), part_counts, part_offsets, part2d_dt.value, 
                target_rank, comm);
            if (rank == target_rank) {
                delete [] part_counts;
                delete [] part_offsets;
            }
            // -------- move data by bytes
            splash::utils::mpi::datatype<T> t_dt;
            MPI_Gatherv(_data, elems, t_dt.value,
                output._data, elem_counts, elem_offsets, t_dt.value,
                target_rank, comm);
            if (rank == target_rank) {
                delete [] elem_counts;
                delete [] elem_offsets;
            }
            
            // --------- update offsets
            if (rank == target_rank) {
                S offset = 0;
                for (int i = 0; i < output.parts.size(); ++i) {
                    output.offsets[i] = offset;
                    offset += output.parts[i].r.size * output.parts[i].c.size;
                }
            }
            return output;
        }
#else   
        aligned_tiles gather(int const & target_rank) const {
            return *this;
        }
#endif


#ifdef USE_MPI
        aligned_tiles row_partition(splash::utils::partition<S> const & row_partition, 
            MPI_Comm comm = MPI_COMM_WORLD) const {
            
            int procs;
            int rank;
            MPI_Comm_size(comm, &procs);
            MPI_Comm_rank(comm, &rank);

            auto stime = getSysTime();

            // -------- sort first.  we'd need things in order anyway.
            // PRINT_RT("aligned_tiles ROW_PARTITION sort ");
            aligned_tiles sorted = sort_by_offsets();

            auto etime = getSysTime();
            PRINT_RT("partition: sorted in %f sec\n", get_duration_s(stime, etime));

            if (procs == 1)  return sorted;

            stime = getSysTime();

            // -------- aggregate the offsets
            int *upper_bounds = new int[procs]{};
            upper_bounds[rank] = row_partition.offset + row_partition.size;
            splash::utils::mpi::datatype<int> int_dt;
            MPI_Allgather(MPI_IN_PLACE, 1, int_dt.value, upper_bounds, 1, int_dt.value, comm);

            // PRINT_RT("UPPER_BOUNDS [");
            // for (int i = 0; i < procs; ++i) {
            //     PRINT_RT("%d ", upper_bounds[i]);
            // }
            // PRINT_RT("]\n");

            
            // PRINT_RT("aligned_tiles ROW_PARTITION sort DONE\n");
            // FLUSH();
            
            // -------- get counts.
            int * part_counts = new int[procs]{};
            int * part_offsets = new int[procs]{};
            int * elem_counts = new int[procs]{};
            int * elem_offsets = new int[procs]{};
            int part_id = 0;
            size_type upper_bound;
            auto start = sorted.parts.begin();
            auto end = sorted.parts.end();
            // count the number of parts per processor 
            for (int i = 0; i < procs; ++i) {
                upper_bound = upper_bounds[i];
                // linear search for next row >= upper_bound
                end = std::find_if_not(start, sorted.parts.end(), 
                    [&upper_bound](splash::utils::partition2D<size_type> const & a){
                        return a.r.offset < upper_bound;
                });
                // get partition counts
                part_counts[i] = std::distance(start, end);
                start = end;

                part_offsets[i] = part_id;
                elem_offsets[i] = sorted.offsets[part_id];
                // increment part_id
                part_id += part_counts[i];

                // offsets is 1 larger than parts, with last entry being the total element count.
                // should not get segv here.
                elem_counts[i] = sorted.offsets[part_id] - elem_offsets[i];
            } 
            // PRINT_RT("PART count [");
            // for (int i = 0; i < procs; ++i) {
            //     PRINT_RT("%d ", part_counts[i]);
            // }
            // PRINT_RT("]\n");
            // PRINT_RT("PART offset [");
            // for (int i = 0; i < procs; ++i) {
            //     PRINT_RT("%d ", part_offsets[i]);
            // }
            // PRINT_RT("]\n");
            // PRINT_RT("ELEM count [");
            // for (int i = 0; i < procs; ++i) {
            //     PRINT_RT("%d ", elem_counts[i]);
            // }
            // PRINT_RT("]\n");
            // PRINT_RT("ELEM offset [");
            // for (int i = 0; i < procs; ++i) {
            //     PRINT_RT("%d ", elem_offsets[i]);
            // }
            // PRINT_RT("]\n");

            delete [] upper_bounds;

            etime = getSysTime();
            PRINT_RT("partition: count parts in %f sec\n", get_duration_s(stime, etime));

            stime = getSysTime();

            // ------ move counts
            int * recv_parts = new int[procs]{};
            int * recv_elems = new int[procs]{};
            MPI_Alltoall(part_counts, 1, int_dt.value, recv_parts, 1, int_dt.value, comm);
            MPI_Alltoall(elem_counts, 1, int_dt.value, recv_elems, 1, int_dt.value, comm);

            int * recv_part_offsets = new int[procs + 1]{};
            int * recv_elem_offsets = new int[procs + 1]{};
            recv_part_offsets[0] = 0;
            recv_elem_offsets[0] = 0;
            for (int i = 0; i < procs; ++i) {
                recv_part_offsets[i + 1] = recv_part_offsets[i] + recv_parts[i];
                recv_elem_offsets[i + 1] = recv_elem_offsets[i] + recv_elems[i];
            }
            // PRINT_RT("RECV PART count [");
            // for (int i = 0; i < procs; ++i) {
            //     PRINT_RT("%d ", recv_parts[i]);
            // }
            // PRINT_RT("]\n");
            // PRINT_RT("RECV PART offset [");
            // for (int i = 0; i <= procs; ++i) {
            //     PRINT_RT("%d ", recv_part_offsets[i]);
            // }
            // PRINT_RT("]\n");
            // PRINT_RT("RECV ELEM count [");
            // for (int i = 0; i < procs; ++i) {
            //     PRINT_RT("%d ", recv_elems[i]);
            // }
            // PRINT_RT("]\n");
            // PRINT_RT("RECV ELEM offset [");
            // for (int i = 0; i <= procs; ++i) {
            //     PRINT_RT("%d ", recv_elem_offsets[i]);
            // }
            // PRINT_RT("]\n");
            etime = getSysTime();
            PRINT_RT("partition: a2a counts in %f sec\n", get_duration_s(stime, etime));

            stime = getSysTime();

            // --------- allocate
            // PRINT_RT("aligned_tiles ROW_PARTITION out ");
            aligned_tiles output(recv_part_offsets[procs], recv_elem_offsets[procs], _align);
            // PRINT_RT("aligned_tiles ROW_PARTITION out DONE\n");
            etime = getSysTime();
            PRINT_RT("partition: alloc out in %f sec\n", get_duration_s(stime, etime));

            stime = getSysTime();

            // -------- move parts by bytes
            splash::utils::mpi::datatype<splash::utils::partition2D<S>> part2d_dt;
            MPI_Alltoallv(sorted.parts.data(), part_counts, part_offsets,  part2d_dt.value,
                output.parts.data(), recv_parts, recv_part_offsets, part2d_dt.value, comm);

            delete [] part_counts;
            delete [] part_offsets;
            delete [] recv_parts;
            delete [] recv_part_offsets;

            etime = getSysTime();
            PRINT_RT("partition: move parts in %f sec\n", get_duration_s(stime, etime));

            stime = getSysTime();

            
            // -------- move data by bytes
            splash::utils::mpi::datatype<T> t_dt;
            MPI_Alltoallv(sorted._data, elem_counts, elem_offsets, t_dt.value,
                output._data, recv_elems, recv_elem_offsets, t_dt.value, comm);

            delete [] elem_counts;
            delete [] elem_offsets;
            delete [] recv_elems;
            delete [] recv_elem_offsets;
            
            etime = getSysTime();
            PRINT_RT("partition: move elem in %f sec\n", get_duration_s(stime, etime));

            stime = getSysTime();
            // -------- reconstruct offsets
            S offset = 0;
            for (size_t i = 0; i < output.parts.size(); ++i) {
                output.offsets[i] = offset;
                offset += output.parts[i].r.size * output.parts[i].c.size;
            }

            etime = getSysTime();
            PRINT_RT("partition: reconstruct offset in %f sec\n", get_duration_s(stime, etime));

            return output;
        }
#else   
        aligned_tiles row_partition(splash::utils::partition<S> const & row_partition) const {
            return sort_by_offsets();
        }
#endif
};


}}