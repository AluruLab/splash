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
        mutable std::vector<splash::utils::partition2D<S>> _parts;
        mutable std::vector<size_type> offsets;  // size is 1 larger than number of _parts, last entry contains total count

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
            _parts(_count), offsets(_count + 1),
            bytes(elements * sizeof(T)) {

            _data = reinterpret_cast<T*>(splash::utils::aalloc(bytes, _align));
            memset(_data, 0, bytes);

            offsets[_count] = elements;
                // FMT_PRINT_RT("INTERNAL _data: {:p}, {}\n", _data, sizeof(_data));
                // FMT_FLUSH();
        }

    public:
        aligned_tiles() : _align(splash::utils::get_cacheline_size()),
            _data(nullptr), bytes(0) {
                // FMT_PRINT_RT("DEFAULT _data: {:p}, {}\n", _data, sizeof(_data));
                // FMT_FLUSH();
            }

        aligned_tiles(splash::utils::partition2D<S> const * start, size_type const & count, size_t const & align = 0) :
            _align(align == 0 ? splash::utils::get_cacheline_size() : align),
            _parts(start, start + count) {

            size_type elements = 0;
            offsets.reserve(count + 1);
            for (auto part : _parts) {
                offsets.emplace_back(elements);
                elements += part.r.size * part.c.size;
            }
            offsets.emplace_back(elements);
            bytes = elements * sizeof(T);

            _data = reinterpret_cast<T*>(splash::utils::aalloc(bytes, _align));
            memset(_data, 0, bytes);
                // FMT_PRINT_RT("POPULATE _data: {:p}, {}\n", _data, sizeof(_data));
                // FMT_FLUSH();
        }
        ~aligned_tiles() {
                // FMT_PRINT_RT("DELETE _data: {:p}, {}\n", _data, sizeof(_data));
                // FMT_FLUSH();
            if (_data) splash::utils::afree(_data);
        }

        aligned_tiles(aligned_tiles const & other) : _align(other._align), _parts(other._parts), offsets(other.offsets) {
            _data = reinterpret_cast<T*>(splash::utils::aalloc(other.allocated(), _align));
            memcpy(_data, other._data, other.allocated());
            bytes = other.allocated();
                // FMT_PRINT_RT("COPY _data: {:p}, {}\n", _data, sizeof(_data));
                // FMT_FLUSH();
        }
        aligned_tiles & operator=(aligned_tiles const & other) {
            if (allocated() != other.allocated()) {
                if (_data) splash::utils::afree(_data);
                _data = reinterpret_cast<T*>(splash::utils::aalloc(other.allocated(), _align));
            }
            if (_data != other._data) memcpy(_data, other._data, other.allocated() );
            bytes = other.bytes;

            _align = other._align;
            _parts.assign(other._parts.begin(), other._parts.end());
            offsets.assign(other.offsets.begin(), other.offsets.end());
                // FMT_PRINT_RT("COPY= _data: {:p}, {}\n", _data, sizeof(_data));
                // FMT_FLUSH();
            return *this;
        }
        // move constructor.  take ownership.
        aligned_tiles(aligned_tiles && other) : 
            _align(other._align), _data(other._data),
            _parts(std::move(other._parts)), offsets(std::move(other.offsets)),
            bytes(other.bytes) {
                other._data = nullptr;
                other.bytes = 0;
                // FMT_PRINT_RT("MOVE _data: {:p}, {}\n", _data, sizeof(_data));
                // FMT_FLUSH();
        }
        aligned_tiles & operator=(aligned_tiles && other) {
            _align = other._align;

            if (_data) splash::utils::afree(_data);
            _data = other._data;
            other._data = nullptr;

            _parts = std::move(other._parts);
            offsets = std::move(other.offsets);
            
            bytes = other.bytes; 
            other.bytes = 0;
                // FMT_PRINT_RT("MOVE= _data: {:p}, {}\n", _data, sizeof(_data));
                // FMT_FLUSH();
            return *this;
        }

        size_type size() const { return _parts.size(); }
        size_type elements() const { return offsets.back(); }
        size_type allocated() const { return bytes; }
        splash::utils::partition2D<size_type> const & part(size_t const & id) const { return _parts[id]; };
        splash::utils::partition2D<size_type>& part(size_t const & id) { return _parts[id]; };
        splash::utils::partition2D<size_type> const * parts(size_t const & id = 0) const { return _parts.data() + id; };
        splash::utils::partition2D<size_type>* parts(size_t const & id = 0) { return _parts.data() + id; };

        size_type const & offset(size_t const & id) const { return offsets[id]; };
        size_type& offset(size_t const & id) { return offsets[id]; };
        T const * data(size_t const & id = 0) const { return _data + offsets[id]; }
        T* data(size_t const & id = 0) { return _data + offsets[id]; }
        void clear_data() { memset(_data, 0, bytes); };

    protected:
        bool is_sorted_by_offsets() const  {
            return std::is_sorted(_parts.begin(), _parts.end(), 
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
            ids.reserve(_parts.size());

            for (size_t i = 0; i < _parts.size(); ++i) {
                ids.emplace_back(_parts[i].r.offset, _parts[i].c.offset, i, offsets[i] );
            }
            auto etime = getSysTime();
            FMT_ROOT_PRINT("sort: set up in {} sec\n", get_duration_s(stime, etime));

            stime = getSysTime();

            // sort the partitions
            std::stable_sort(ids.begin(), ids.end(), [](sort_elem const & a, sort_elem const & b){
                return (a.r_offset == b.r_offset) ? (a.c_offset < b.c_offset) : (a.r_offset < b.r_offset);
            });
            etime = getSysTime();
            FMT_ROOT_PRINT("sort: sorted_ids in {} sec\n", get_duration_s(stime, etime));

            stime = getSysTime();
            
            // ======= now copy with reorder
            // FMT_ROOT_PRINT("aligned_tiles SORT ");
            aligned_tiles output(ids.size(), offsets.back(), _align);
            // FMT_ROOT_PRINT("aligned_tiles SORT DONE\n");
            // FMT_FLUSH();
            etime = getSysTime();
            FMT_ROOT_PRINT("sort: size_output in {} sec\n", get_duration_s(stime, etime));

            stime = getSysTime();
            
            size_type offset = 0;
            size_type s = 0;
            splash::utils::partition2D<size_type> part;
            for (size_t i = 0; i < ids.size(); ++i) {
                part = _parts[ids[i].part_id];
                output._parts[i] = part;
                output.offsets[i] = offset;
                s = part.r.size * part.c.size;
                memcpy(output._data + offset, _data + ids[i].ptr_offset, s * sizeof(T));
                offset += s;
            }
            etime = getSysTime();
            FMT_ROOT_PRINT("sort: shuffled in {} sec\n", get_duration_s(stime, etime));

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
            // FMT_PRINT_RT("aligned_tiles TRANSPOSE ");
            aligned_tiles output(_parts.data(), _parts.size(), _align);
            for (size_type i = 0; i < output._parts.size(); ++i) {
                output._parts[i].r = _parts[i].c;
                output._parts[i].c = _parts[i].r;
                
                transpose_tile(_data + offsets[i], 
                    _parts[i].r.size, _parts[i].c.size, 
                    output._data + output.offsets[i]);
            }

            return output;
        }

        aligned_tiles merge(aligned_tiles const & other) const {
            // FMT_PRINT_RT("aligned_tiles OPERATOR+ ");

            aligned_tiles output(_parts.size() + other._parts.size(), offsets.back() + other.offsets.back(), _align);
            // FMT_PRINT_RT("left, _parts: {}, elements: {}\n", size(), offsets.back());
            // FMT_PRINT_RT("right, _parts: {}, elements: {}\n", other.size(), other.offsets.back());
            size_t i = 0;
            for (; i < _parts.size(); ++i) {
                output._parts[i] = _parts[i];
                output.offsets[i] = offsets[i];
            }
            memcpy(output._data, _data, allocated() );

            size_type offset = offsets.back();
            for (size_t j = 0; j < other._parts.size(); ++j, ++i) {
                output._parts[i] = other._parts[j];
                output.offsets[i] = offset + other.offsets[j];
            }
            memcpy(output._data + offsets.back(), other._data, other.allocated());
            // FMT_PRINT_RT("addition, _parts: {}, elements: {}\n", output.size(), output.offsets.back());

            return output;
        }

        // transpose.  offsets don't change.  id and id_cols do not change.  r and c are swapped.
        aligned_tiles reflect_diagonally() const {
            // first count non-diagonal.
            size_t non_diag = 0;
            size_t non_diag_elements = 0;
            size_t i = 0;
            for (i = 0; i < _parts.size(); ++i)  {
                if (_parts[i].r.offset != _parts[i].c.offset) {
                    ++non_diag;
                    non_diag_elements += _parts[i].r.size * _parts[i].c.size;
                }
            }

            // allocate
            aligned_tiles output(_parts.size() + non_diag, offsets.back() + non_diag_elements, _align);
            
            // copy over the old.
            memcpy(output._parts.data(), _parts.data(), sizeof(splash::utils::partition2D<S>) * _parts.size());
            memcpy(output.offsets.data(), offsets.data(), sizeof(size_t) * _parts.size());
            memcpy(output._data, _data, allocated() );

            // copy the rest and transpose.
            size_t j = 0;
            size_t off = offsets.back();
            for (j = 0; j < _parts.size(); ++j)  {
                if (_parts[j].r.offset != _parts[j].c.offset) {
                    output._parts[i].r = _parts[j].c;
                    output._parts[i].c = _parts[j].r;
                    output.offsets[i] = off;
                    
                    transpose_tile(_data + offsets[j], 
                        _parts[j].r.size, _parts[j].c.size, 
                        output._data + off);

                    off += _parts[j].r.size * _parts[j].c.size;
                    ++i;
                }
            }

            return output;
        }


        splash::utils::partition2D<S> get_bounds() {
            if (_parts.size() > 0) {
                S rmin = std::numeric_limits<S>::max();
                S cmin = std::numeric_limits<S>::max();
                S rmax = std::numeric_limits<S>::lowest();
                S cmax = std::numeric_limits<S>::lowest();
                
                for (size_t i = 0; i < _parts.size(); ++i) {
                    rmin = std::min(rmin, _parts[i].r.offset);
                    cmin = std::min(cmin, _parts[i].c.offset);
                    rmax = std::max(rmax, _parts[i].r.offset + _parts[i].r.size);
                    cmax = std::max(cmax, _parts[i].c.offset + _parts[i].c.size);
                }
                // FMT_PRINT_RT("rmin {} rmax {}, cmin {} cmax {}\n", rmin, rmax, cmin, cmax);
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
            auto part = _parts[i];
            // FMT_PRINT_RT("COPY_TO tile {}: ", i);
            // print("COPY TO: ", i);
            T const * src = _data + offsets[i];  // src block of data
            if ((part.r.offset < row_offset) || (part.r.offset + part.r.size > row_offset + matrix.rows())) {
                FMT_PRINT_RT("row offset: {} size {} ", row_offset, matrix.rows());
                part.r.print("row: ");
            }
            if ((part.c.offset < col_offset) || (part.c.offset + part.c.size > col_offset + matrix.columns())) {
                FMT_PRINT_RT("col offset: {} size {} ", col_offset, matrix.columns());
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
            for (size_t i = 0; i < _parts.size(); ++i) {
                this->copy_to(matrix, row_offset, col_offset, i);
            }
        }

        template <typename TT = T, typename std::enable_if<std::is_arithmetic<TT>::value, int>::type = 1>
        inline void print(const char* prefix, size_t const & id) const {
            TT const *  ptr = _data + offsets[id];;

            _parts[id].print(prefix);
            FMT_PRINT_RT("{} data offsets = {}\n", prefix, offsets[id]);
            for (size_t r = 0; r < _parts[id].r.size; ++r) {
                FMT_PRINT_RT("{} ", prefix);
                for (size_t c = 0; c < _parts[id].c.size; ++c, ++ptr) {
                    FMT_PRINT("{}, ", *ptr);
                }
                FMT_PRINT("\n");
            }
        }
        template <typename TT = T, typename std::enable_if<!std::is_arithmetic<TT>::value, int>::type = 1>
        inline void print(const char* prefix, size_t const & id) const {
            TT const *  ptr = _data + offsets[id];;

            _parts[id].print(prefix);
            FMT_PRINT_RT("{} data offsets = {}\n", prefix, offsets[id]);
            for (size_t r = 0; r < _parts[id].r.size; ++r) {
                FMT_PRINT_RT("{} ", prefix);
                for (size_t c = 0; c < _parts[id].c.size; ++c, ++ptr) {
                    ptr->print();
                    FMT_PRINT(", ");
                }
                FMT_PRINT("\n");
            }
        }

        void print(const char * prefix) const {
            for (size_t i = 0; i < _parts.size(); ++i)  {
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
            int nparts = _parts.size();
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
            // FMT_PRINT_RT("aligned_tiles GATHER ");
            aligned_tiles output;
            if (rank == target_rank) {  
                // FMT_PRINT_RT("aligned_tiles GATHER assign");
                output = std::move(aligned_tiles(part_offsets[procs], elem_offsets[procs], _align));
            }
            // -------- move _parts by bytes
            splash::utils::mpi::datatype<splash::utils::partition2D<S>> part2d_dt;
            MPI_Gatherv(_parts.data(), nparts, part2d_dt.value,
                output._parts.data(), part_counts, part_offsets, part2d_dt.value, 
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
                for (int i = 0; i < output._parts.size(); ++i) {
                    output.offsets[i] = offset;
                    offset += output._parts[i].r.size * output._parts[i].c.size;
                }
            }
            return output;
        }
#else   
        aligned_tiles gather(int const & target_rank) const {
            return *this;
        }
#endif


        // partition by row, for MPI.  based on offsets only.
#ifdef USE_MPI
        aligned_tiles allgather(MPI_Comm comm = MPI_COMM_WORLD ) const {

            int procs;
            int rank;
            MPI_Comm_size(comm, &procs);
            MPI_Comm_rank(comm, &rank);

            if (procs == 1) return *this;

            // get counts and offsets of partitions
            int * part_counts = new int[procs] {};
            int * elem_counts = new int[procs] {};
            part_counts[rank] = _parts.size();
            elem_counts[rank] = offsets.back();
            splash::utils::mpi::datatype<int> int_dt;
            MPI_Allgather(MPI_IN_PLACE, 1, int_dt.value, part_counts, 1, int_dt.value, comm);
            MPI_Allgather(MPI_IN_PLACE, 1, int_dt.value, elem_counts, 1, int_dt.value, comm);

            int *part_offsets = new int[procs + 1];
            int *elem_offsets = new int[procs + 1];
            part_offsets[0] = 0;
            elem_offsets[0] = 0;
            for (int i = 0; i < procs; ++i) {
                part_offsets[i + 1] = part_offsets[i] + part_counts[i];
                elem_offsets[i + 1] = elem_offsets[i] + elem_counts[i];
            }

            // allocate output
            // FMT_PRINT_RT("aligned_tiles GATHER ");
            aligned_tiles output(part_offsets[procs], elem_offsets[procs], _align);

            // -------- move _parts by bytes
            splash::utils::mpi::datatype<splash::utils::partition2D<S>> part2d_dt;
            MPI_Allgatherv(_parts.data(), _parts.size(), part2d_dt.value,
                output._parts.data(), part_counts, part_offsets, part2d_dt.value, comm);
            delete [] part_counts;
            delete [] part_offsets;

            // -------- move data by bytes
            splash::utils::mpi::datatype<T> t_dt;
            MPI_Allgatherv(_data, offsets.back(), t_dt.value,
                output._data, elem_counts, elem_offsets, t_dt.value, comm);
            delete [] elem_counts;
            delete [] elem_offsets;
            
            // --------- update offsets
            S offset = 0;
            for (size_t i = 0; i < output._parts.size(); ++i) {
                output.offsets[i] = offset;
                offset += output._parts[i].r.size * output._parts[i].c.size;
            }
            return output;
        }
#else   
        aligned_tiles allgather() const {
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
            // FMT_PRINT_RT("aligned_tiles ROW_PARTITION sort ");
            aligned_tiles sorted = sort_by_offsets();

            auto etime = getSysTime();
            // FMT_ROOT_PRINT("partition: sorted in {} sec\n", get_duration_s(stime, etime));

            if (procs == 1)  return sorted;

            stime = getSysTime();

            // -------- aggregate the offsets
            int *upper_bounds = new int[procs]{};
            upper_bounds[rank] = row_partition.offset + row_partition.size;
            splash::utils::mpi::datatype<int> int_dt;
            MPI_Allgather(MPI_IN_PLACE, 1, int_dt.value, upper_bounds, 1, int_dt.value, comm);

            // FMT_PRINT_RT("UPPER_BOUNDS [");
            // for (int i = 0; i < procs; ++i) {
            //     FMT_PRINT_RT("{} ", upper_bounds[i]);
            // }
            // FMT_PRINT_RT("]\n");

            
            // FMT_PRINT_RT("aligned_tiles ROW_PARTITION sort DONE\n");
            // FMT_FLUSH();
            
            // -------- get counts.
            int * part_counts = new int[procs]{};
            int * part_offsets = new int[procs]{};
            int * elem_counts = new int[procs]{};
            int * elem_offsets = new int[procs]{};
            int part_id = 0;
            size_type upper_bound;
            auto start = sorted._parts.begin();
            auto end = sorted._parts.end();
            // count the number of _parts per processor 
            for (int i = 0; i < procs; ++i) {
                upper_bound = upper_bounds[i];
                // linear search for next row >= upper_bound
                end = std::find_if_not(start, sorted._parts.end(), 
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

                // offsets is 1 larger than _parts, with last entry being the total element count.
                // should not get segv here.
                elem_counts[i] = sorted.offsets[part_id] - elem_offsets[i];
            } 
            // FMT_PRINT_RT("PART count [");
            // for (int i = 0; i < procs; ++i) {
            //     FMT_PRINT_RT("{} ", part_counts[i]);
            // }
            // FMT_PRINT_RT("]\n");
            // FMT_PRINT_RT("PART offset [");
            // for (int i = 0; i < procs; ++i) {
            //     FMT_PRINT_RT("{} ", part_offsets[i]);
            // }
            // FMT_PRINT_RT("]\n");
            // FMT_PRINT_RT("ELEM count [");
            // for (int i = 0; i < procs; ++i) {
            //     FMT_PRINT_RT("{} ", elem_counts[i]);
            // }
            // FMT_PRINT_RT("]\n");
            // FMT_PRINT_RT("ELEM offset [");
            // for (int i = 0; i < procs; ++i) {
            //     FMT_PRINT_RT("{} ", elem_offsets[i]);
            // }
            // FMT_PRINT_RT("]\n");

            delete [] upper_bounds;

            etime = getSysTime();
            // FMT_ROOT_PRINT("partition: count _parts in {} sec\n", get_duration_s(stime, etime));

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
            // FMT_PRINT_RT("RECV PART count [");
            // for (int i = 0; i < procs; ++i) {
            //     FMT_PRINT_RT("{} ", recv_parts[i]);
            // }
            // FMT_PRINT_RT("]\n");
            // FMT_PRINT_RT("RECV PART offset [");
            // for (int i = 0; i <= procs; ++i) {
            //     FMT_PRINT_RT("{} ", recv_part_offsets[i]);
            // }
            // FMT_PRINT_RT("]\n");
            // FMT_PRINT_RT("RECV ELEM count [");
            // for (int i = 0; i < procs; ++i) {
            //     FMT_PRINT_RT("{} ", recv_elems[i]);
            // }
            // FMT_PRINT_RT("]\n");
            // FMT_PRINT_RT("RECV ELEM offset [");
            // for (int i = 0; i <= procs; ++i) {
            //     FMT_PRINT_RT("{} ", recv_elem_offsets[i]);
            // }
            // FMT_PRINT_RT("]\n");
            etime = getSysTime();
            // FMT_ROOT_PRINT("partition: a2a counts in {} sec\n", get_duration_s(stime, etime));

            stime = getSysTime();

            // --------- allocate
            // FMT_PRINT_RT("aligned_tiles ROW_PARTITION out ");
            aligned_tiles output(recv_part_offsets[procs], recv_elem_offsets[procs], _align);
            // FMT_PRINT_RT("aligned_tiles ROW_PARTITION out DONE\n");
            etime = getSysTime();
            // FMT_ROOT_PRINT("partition: alloc out in {} sec\n", get_duration_s(stime, etime));

            stime = getSysTime();

            // -------- move _parts by bytes
            splash::utils::mpi::datatype<splash::utils::partition2D<S>> part2d_dt;
            MPI_Alltoallv(sorted._parts.data(), part_counts, part_offsets,  part2d_dt.value,
                output._parts.data(), recv_parts, recv_part_offsets, part2d_dt.value, comm);

            delete [] part_counts;
            delete [] part_offsets;
            delete [] recv_parts;
            delete [] recv_part_offsets;

            etime = getSysTime();
            // FMT_ROOT_PRINT("partition: move _parts in {} sec\n", get_duration_s(stime, etime));

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
            // FMT_ROOT_PRINT("partition: move elem in {} sec\n", get_duration_s(stime, etime));

            stime = getSysTime();
            // -------- reconstruct offsets
            S offset = 0;
            for (size_t i = 0; i < output._parts.size(); ++i) {
                output.offsets[i] = offset;
                offset += output._parts[i].r.size * output._parts[i].c.size;
            }

            etime = getSysTime();
            // FMT_ROOT_PRINT("partition: reconstruct offset in {} sec\n", get_duration_s(stime, etime));

            return output;
        }
#else   
        aligned_tiles row_partition(splash::utils::partition<S> const & row_partition) const {
            return sort_by_offsets();
        }
#endif
};


}}