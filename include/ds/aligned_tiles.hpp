#pragma once

#include "utils/memory.hpp"
#include "utils/partition.hpp"
#include "ds/algined_matrix.hpp"
#include <vector>
#include <utility>  // pair
#include <limits>
#include <algorithm>  // min/max

#ifdef WITH_MPI
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

    protected:
        size_t _align;  // alignment
        T * _data;  // consists of count tiles, each r_size * c_size
        std::vector<splash::utils::partition2D<S>> parts;
        std::vector<size_t> offsets;  // size is 1 larger than number of parts, last entry contains total count

        struct sort_elem {
            S r_offset;
            S c_offset;
            size_t part_id;
            size_t ptr_offset;
        };

        // only used by internal functions.
        aligned_tiles(size_t const & _count, size_t const & elements, size_t const & align = 0) :
            _align(align == 0 ? splash::utils::get_cacheline_size() : align) {

            _data = reinterpret_cast<T*>(splash::utils::aligned_alloc(elements * sizeof(T), _align));
            memset(_data, 0, allocated());

            parts.allocate(_count);
            offsets.resize(_count + 1, 0);
            offsets[_count] = elements;
        }

    public:
        aligned_tiles(splash::utils::partition2D<S> const * start, size_t const & count, size_t const & align = 0) :
            _align(align == 0 ? splash::utils::get_cacheline_size() : align),
            parts(start, start + count) {

            size_t elements = 0;
            offsets.allocate(count + 1);
            for (auto part : parts) {
                offsets.push_back(elements);
                elements += part.r.size * part.c.size;
            }
            offsets.push_back(elements);

            _data = reinterpret_cast<T*>(splash::utils::aligned_alloc(elements * sizeof(T), _align));
            memset(_data, 0, allocated());
        }
        ~aligned_tiles() {
            if (_data) splash::utils::aligned_free(_data);
        }

        aligned_tile(aligned_tile const & other) : _align(other._align), parts(other.parts), offsets(other.offsets) {
            _data = reinterpret_cast<T*>(splash::utils::aligned_alloc(elements * sizeof(T), _align));
            memcpy(_data, other._data, other.allocated() * sizeof(T));
        }
        aligned_tile & operator=(aligned_tile const & other) {
            if ((_align != other._align) && (allocated() != other.allocated())) {
                splash::utils::aligned_free(_data);
                _data = reinterpret_cast<T*>(splash::utils::aligned_alloc(elements * sizeof(T), _align));
            }
            memcpy(_data, other._data, other.allocated() * sizeof(T));

            _align = other._align;
            parts.assign(other.parts.begin(), other.parts.end());
            offsets.assign(other.offsets.begin(), other.offsets.end());
            return *this;
        }
        // move constructor.  take ownership.
        aligned_tile(aligned_tile && other) : _align(other._align) {
            parts.swap(other.parts);
            offsets.swap(other.offsets);
            std::swap(_data, other._data);
        }
        aligned_tile & operator=(aligned_tile && other) {
            std::swap(_align, other._align);
            parts.swap(other.parts);
            offsets.swap(other.offsets);
            std::swap(_data, other._data);

        }



        size_t size() const { return parts.size(); }
        size_t allocated() const { return offsets.last(); }
        T* data(size_t const & id) { return _data + offsets[id]; }

    protected:
        bool is_sorted_by_offsets() {
            return std::is_sorted(parts.begin(), parts.end(), [](partition2D<size_t> const & a, partition2D<size_t> const & b){
                return (a.r.offset == b.r.offset) ? (a.c.offset < b.c.offset) : (a.r.offset < b.r.offset);
            });
        }

    public:

        aligned_tiles<T, S> sort_by_offsets() {
            if (is_sorted_by_offset()) return *this;


            // ====== first sort.
            std::vector<sort_elem> x;
            x.allocate(parts.size());

            for (size_t i = 0; i < parts.size(); ++i) {
                x.emplace_back(parts[i].r.offset, parts[i].c.offset, i, offsets[i] );
            }

            // sort the partitions
            std::stable_sort(x.begin(), x.end(), [](sort_elem const & a, sort_elem const & b){
                return (a.r_offset == b.r_offset) ? (a.c_offset < b.c_offset) : (a.r_offset < b.r_offset);
            });
            
            // ======= now copy with reorder
            aligned_tiles<T, S> output(x.size(), allocated(), _align);
            size_t offset = 0;
            size_t s = 0;
            partition2D<size_t> part;
            for (size_t i = 0; i < x.size(); ++i) {
                part = parts[x[i].part_id];
                output.parts.push_back(part);
                output.offsets[i] = offset;
                s = part.r.size * part.c.size;
                memcpy(output._data + offset, _data + x[i].ptr_offset, s * sizeof(T));
                offset += s;
            }

            return output;
        }
        
    protected:
        size_t transpose_tile(T const* orig, 
            size_t const & orig_rows, 
            size_t const & orig_cols,
            T * transposed) {
            T const * ptr;
            T * nptr;
            for (size_t c = 0; c < orig_cols; ++c) {
                
                for (size_t r = 0; r < orig_rows; ++r, ++nptr) {
                    *nptr = ptr[c * orig_rows + r];
                }
            }

            return orig_rows * orig_cols;
        }

    public:

        // transpose.  offsets don't change.  id and id_cols do not change.  r and c are swapped.
        aligned_tiles<T, S> transpose() {
            aligned_tiles<T, S> output(parts.data(), parts.size(), _align);
            size_t offset = 0;
            for (size_t i = 0; i < output.parts.size(); ++i) {
                output.parts[i].r = parts[i].c;
                output.parts[i].c = parts[i].r;
                
                offset += 
                    transpose_tile(_data + offset, parts[i].r.size, parts[i].c.size, output._data + offset);
            }

            return output;
        }

        aligned_tiles<T, S> operator+(aligned_tiles<T, S> const & other) {
            aligned_tiles<T, S> output(parts.size() + other.parts.size(), allocated() + other.allocated(), _align);

            size_t i = 0;
            for (; i < parts.size(); ++i) {
                output.parts[i] = parts[i];
                output.offsets[i] = offsets[i];
            }
            memcpy(output._data, _data, allocated() * sizeof(T));

            size_t offset = allocated();
            for (size_t j = 0; j < other.parts.size(); ++j, ++i) {
                output.parts[i] = other.parts[j];
                output.offsets[i] = offset + other.offsets[j];
            }
            memcpy(output._data + allocated(), other._data, other.allocated() * sizeof(T));

        }

        // fill dense matrix
        void copy_to(aligned_matrix<T, S> & matrix) {
            T const * src;
            T * dest;
            for (size_t i = 0; i < parts.size(); ++i) {
                // every part.
                auto part = parts[i];
                src = _data + offsets[i];
                dest = matrix.data(part.r.offset, part.c.offset);

                for (size_t r = 0; r < part.r.size; ++r,
                    src += part.c.size, dest += matrix.cols()) {
                    // every row.
                    memcpy(dest, src, part.c.size * sizeof(T));
                }   

            }
        }

        // partition by row, for MPI.  based on offsets only.
#ifdef WITH_MPI
        aligned_tiles<T, S> gather(int const & target_rank, 
            MPI_Comm comm = MPI_COMM_WORLD ) {

            int procs;
            int rank;
            MPI_Comm_size(comm, &procs);
            MPI_Comm_rank(comm, &rank);

            // get count of partitions
            int * part_counts = nullptr;
            int * elem_counts = nullptr;
            int parts = parts.size();
            int elems = parts.size();
            if (rank == target_rank) {
                part_counts = new int[procs] {};
                elem_counts = new int[procs] {};
            }
            splash::utils::mpi::datatype<int> int_dt;
            MPI_Gather(&parts, 1, int_dt.value, part_counts, 1, int_dt.value, target_rank, comm);
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
            aligned_tiles<T, S> output;
            if (rank == target_rank) {
                output = std::move(aligned_tile<T, S>(part_offsets[procs], elem_offsets[procs], _align));
            }
            // -------- move parts by bytes
            splash::utils::mpi::datatype<partition2D<S>> part2d_dt;
            MPI_Gatherv(parts.data(), parts, part2d_dt.value,
                output.parts.data(), part_counts, part_offsets, part2d_dt.value, 
                target_rank, comm);

            // -------- move data by bytes
            splash::utils::mpi::datatype<T> t_dt;
            MPI_Gatherv(_data, elems, t_dt.value,
                output._data, elem_counts, elem_offsets, t_dt.value,
                target_rank, comm);
            
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


        aligned_tiles<T, S> row_partition(splash::utils::partition<S> const & row_partition, 
            MPI_Comm comm = MPI_COMM_WORLD) {
            
            int procs;
            int rank;
            MPI_Comm_size(comm, &procs);
            MPI_Comm_rank(comm, &rank);
            
            // -------- aggregate the offsets
            int *upper_bounds = new int[procs]{};
            upper_bounds[rank] = row_partition.offset + row_partition.size;
            splash::utils::mpi::datatype<int> int_dt;
            MPI_Allgather(MPI_IN_PLACE, 1, int_dt.value, upper_bounds, 1, int_dt.value, comm);

            // -------- sort first.  we'd need things in order anyway.
            aligned_tiles<T, S> sorted = sort_by_offsets();
            
            // -------- get counts.  this should be faster than sorting first.
            int * part_counts = new int[procs]{};
            int * part_offsets = new int[procs]{};
            int * elem_counts = new int[procs]{};
            int * elem_offsets = new int[procs]{};
            int part_id = 0;
            size_t upper_bound;
            auto start = parts.begin();
            auto end = parts.end();
            // count the number of parts per processor 
            for (int i = 0;; i < procs; ++i) {
                upper_bound = upper_bounds[i];
                // linear search for next row >= upper_bound
                end = std::find_if_not(start, parts.end(), 
                    [&upper_bound](partition2D<size_t> const & a){
                        return a.r.offset < upper_bound;
                });
                // get partition counts
                part_counts[i] = std::distance(start, end);
                start = end;

                part_offsets[i] = part_id;
                elem_offsets[i] = offsets[part_id];
                // increment part_id
                part_id += part_counts[i];

                // offsets is 1 larger than parts, with last entry being the total element count.
                // should not get segv here.
                elem_counts[i] = offsets[part_id] - elem_offsets[i];
            } 



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

            // --------- allocate
            aligned_tiles<T, S> output(recv_part_offsets[procs], recv_elem_offsets[procs], _align);

            // -------- move parts by bytes
            splash::utils::mpi::datatype<partition2D<S>> part2d_dt;
            MPI_Alltoallv(parts.data(), part_counts, part_offsets,  part2d_dt.value,
                output.parts.data(), recv_parts, recv_part_offsets, part2d_dt.value, comm);

            // -------- move data by bytes
            splash::utils::mpi::datatype<T> t_dt;
            MPI_Alltoallv(_data, elem_counts, elem_offsets,  t_dt.value,
                output._data, recv_elems, recv_elem_offsets, t_dt.value, comm);
            
            // -------- reconstruct offsets
            S offset = 0;
            for (int i = 0; i < output.parts.size(); ++i) {
                output.offsets[i] = offset;
                offset += output.parts[i].r.size * output.parts[i].c.size;
            }
            return output;

#else   
        aligned_tiles<T, S> row_partition() {
            return *this;
#endif
        }
};


}}