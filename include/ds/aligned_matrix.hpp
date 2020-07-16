/*
 *  aligned_matrix.hpp
 *
 *  Created on: June 11, 2020
 *  Author: Tony Pan
 *  Affiliation: Institute for Data Engineering and Science
 *  			Georgia Institute of Technology, Atlanta, GA 30332
 */

/* TODO:
 * [ ] use a partition object as dimension descriptor
 */

#pragma once  // instead of #ifndef ...

#include "utils/memory.hpp"
#include "utils/partition.hpp"
#include <cstring> //memset, memcpy

#ifdef USE_MPI
#include "mpi.h"
#include "utils/mpi_types.hpp"
#endif

// forward declare
namespace splash { 
namespace ds { 
template<typename FloatType>
class aligned_matrix;
}

#ifdef USE_MPI
namespace utils { namespace mpi {

// define data type for 1 row.
template <typename T> 
struct datatype<splash::ds::aligned_matrix<T>, false> {
    datatype(size_t const & count, size_t const & row_bytes) {
        splash::utils::mpi::datatype<T> dt1;
        splash::utils::mpi::datatype<unsigned char> dt2;
        MPI_Datatype type[2] = {
            dt1.value,
            dt2.value
        };
        int blocklen[2];
        blocklen[0] = count;
        blocklen[1]  = row_bytes - count * sizeof(T);
        MPI_Aint disp[2];
        disp[0] = 0;
        disp[1] = count * sizeof(T);

        MPI_Type_create_struct(2, blocklen, disp, type, &value);
        MPI_Type_commit(&value);
    }
    ~datatype() {
        MPI_Type_free(&value);
    }
    MPI_Datatype value;
};

}}
#endif

namespace ds { 

// NOTE: this is a local data structure. 
//       It is not thread safe.
template<typename FloatType>
class aligned_matrix {
    public:
        using reference = FloatType &;
        using const_reference = FloatType const &;
        using pointer = FloatType *;
        using const_pointer = FloatType const *;
        using iterator = pointer;
        using const_iterator = const_pointer;

        using data_type = FloatType;
        using size_type = size_t;

    protected:
        unsigned char* _data; // internal pointer.
        size_type _rows;   // number of vectors in the data
        size_type _cols;   // vector length
        size_t _align;  // alignment
        size_type _bytes_per_row;   // assume row major
        bool manage;   // owning _data or not.

        inline pointer _get_row(size_type const & row = 0) {
            return reinterpret_cast<pointer>(_data + row * _bytes_per_row);
        }
        inline const_pointer _get_row(size_type const & row = 0) const {
            return reinterpret_cast<const_pointer>(_data + row * _bytes_per_row);
        }

    public:
        aligned_matrix() : 
            _data(nullptr), _rows(0), _cols(0), _align(splash::utils::get_cacheline_size()), _bytes_per_row(0), manage(true) {}
        
        // construct, optionally with allocated data.
        // alignment of 0 indicates: use system's cacheline size.
        aligned_matrix(size_type const & rows, size_type const & cols, size_t const & align = 0, 
                        void* data = nullptr, bool const & copy=true) :
            _rows(rows), _cols(cols), _align(align == 0 ? splash::utils::get_cacheline_size() : align),
            _bytes_per_row(splash::utils::get_aligned_size(cols * sizeof(FloatType), _align)),
            manage(copy)
        {
            if (manage) {
                _data = reinterpret_cast<unsigned char*>(splash::utils::aalloc_2D(_rows, _cols * sizeof(FloatType), _align));  // total size is multiple of alignment.
                memset(_data, 0, this->allocated());
            }
            if (data) {
                if (copy)
                   memcpy(_data, data, _rows * _bytes_per_row);
                else _data = reinterpret_cast<unsigned char*>(data);
            }
        }
        // copy constructor and assignment.  deep copy.
        aligned_matrix(aligned_matrix const & other) : 
            aligned_matrix(other._rows, other._cols, other._align, other._data, other.manage) {}
        aligned_matrix & operator=(aligned_matrix const & other) {
            if (!other.manage) 
                _data = other._data;
            else {
                // not same size.  free and reallocate.
                if (this->allocated() != other.allocated()) {
                    if (_data) splash::utils::afree(_data);
                    _data = reinterpret_cast<unsigned char*>(splash::utils::aalloc(other.allocated(), other._align));  // total size is multiple of alignment.
                }
                memcpy(_data, other._data, other.allocated());
            }
            _rows = other._rows;
            _cols = other._cols;
            _align = other._align;
            _bytes_per_row = other._bytes_per_row;
            manage = other.manage;
            return *this;
        }
        // move constructor.  take ownership.
        aligned_matrix(aligned_matrix && other) : 
            _data(other._data), _rows(other._rows), _cols(other._cols), _align(other._align),
            _bytes_per_row(other._bytes_per_row), manage(other.manage)
         {
            other._data = nullptr;
            other.manage = false;
            other._rows = 0;
            other._cols = 0;
            other._bytes_per_row = 0;
        }
        aligned_matrix & operator=(aligned_matrix && other) {
            if (_data && manage) {
                splash::utils::afree(_data);
            }
            _data = other._data;
            manage = other.manage;
            other._data = nullptr;
            other.manage = false;

            _rows = other._rows;
            other._rows = 0;
            
            _cols = other._cols;
            other._cols = 0;
            
            _align = other._align;
            _bytes_per_row = other._bytes_per_row;
            other._bytes_per_row = 0;
            
            return *this;
        }


        ~aligned_matrix() {
            if (_data && manage) {
                splash::utils::afree(_data);
            }
            _data = nullptr;
        }

        inline size_type size() const {  return _rows * _cols; }
        inline size_type allocated() const {
            return _rows * _bytes_per_row;
        }
        // resizes. copy data.
        inline void resize(size_type const  & rows, size_type const & cols)  {
            if ((rows == _rows)  && (cols == _cols)) return;
            // allocate new data.
            unsigned char* data = nullptr;
            size_type bytes_per_row = splash::utils::get_aligned_size(cols * sizeof(FloatType), _align);
            if ((rows > 0) && (cols > 0)) {
                data = reinterpret_cast<unsigned char*>(splash::utils::aalloc_2D(rows, bytes_per_row, _align));  // total size is multiple of alignment.
                memset(data, 0, rows * bytes_per_row);
            }

            // copy data if any.
            if (_data && data) {
                size_type rmin = std::min(rows, _rows);
                size_type bmin = std::min(bytes_per_row, _bytes_per_row);
                unsigned char* src = _data;
                unsigned char* dest = data;
                for (size_type r = 0; r < rmin; ++r, src+=_bytes_per_row, dest+= bytes_per_row) {
                    memcpy(dest, src, bmin);
                }
            }
            // swap.
            std::swap(_data, data);
            _rows = rows;
            _cols = cols;
            _bytes_per_row = bytes_per_row;

            // if not managing, then okay to replace the pointer..  else need to free old data.
            if (data && manage) {
                splash::utils::afree(data);
            }
            // if was not managing, now is.
            manage = true;
        }


        inline size_type rows() const {  return _rows; }
        inline size_type columns() const {  return _cols; }
        inline size_type column_bytes() const {
            return _bytes_per_row;
        }


        inline pointer data(size_type const & row = 0, size_type const & col = 0) noexcept { 
            return _get_row(row) + col; 
        }
        inline const_pointer data(size_type const & row = 0, size_type const & col = 0) const noexcept {
            return _get_row(row) + col; 
        }

        // data value accessor
        inline reference operator()(size_type const & row = 0, size_type const & col = 0) { 
            return _get_row(row)[col]; 
        }
        inline const_reference operator()(size_type const & row = 0, size_type const & col = 0) const { 
            return _get_row(row)[col]; 
        }

        inline reference at(size_type const & row = 0, size_type const & col = 0) { 
            return _get_row(row)[col]; 
        }
        inline const_reference at(size_type const & row = 0, size_type const & col = 0) const { 
            return _get_row(row)[col]; 
        }

        // inline explicit operator FloatType*() { return reinterpret_cast<pointer>(_data); }
        // inline explicit operator FloatType*() const { return reinterpret_cast<const_pointer>(_data); }


	    /*transpose input matrix*/
	    aligned_matrix<FloatType> local_transpose() {
            aligned_matrix<FloatType> output(_cols, _rows, _align);
        
            pointer in;
            /*transpose the matrix*/
            for(size_type row = 0; row < _rows; ++row){
                in = this->_get_row(row);
                for(size_type col = 0; col < _cols; ++col) {
                    output(col, row) = in[col];
                }
            }

        }
        // shallow copy?
        aligned_matrix<FloatType> deep_copy() {
            aligned_matrix<FloatType> output(_rows, _cols, _align);
            memcpy(output._data, _data, this->allocated());
        }

#ifdef USE_MPI
        bool check_aligned_matrix(MPI_Comm comm = MPI_COMM_WORLD) const {
            // first get the row and column counts on each rank
            size_t check[2];
            size_t result[2];
            check[0] = _align;
            check[1] = _cols;
            result[0] = 0;
            result[1] = 0;
            splash::utils::mpi::datatype<size_t> ul_dt;
            MPI_Allreduce(check, result, 2, ul_dt.value, MPI_MAX, comm);

            return (check[0] == result[0]) && (check[1] == result[1]);
        }
#endif
            
#ifdef USE_MPI
        aligned_matrix<FloatType> gather(int target_rank = 0, MPI_Comm comm = MPI_COMM_WORLD) const {
            int procs;
            int rank;
            MPI_Comm_size(comm, &procs);
            MPI_Comm_rank(comm, &rank);

            if (procs == 1) return *this;

            // validate that all columns are same.
            // validate that all alignments are same.
            if (! check_aligned_matrix()) {
                throw std::logic_error("column count or alignment do not match between MPI processes.\n");
            }

            // get total row count
            int rows = _rows;
            int *row_counts = nullptr;
            if (rank == target_rank) {
                row_counts = new int[procs];
            }
            splash::utils::mpi::datatype<int> i_dt;
            MPI_Gather(&rows, 1, i_dt.value, 
                row_counts, 1, i_dt.value, 
                target_rank, comm);
            
            int *row_offsets = nullptr;
            if (rank == target_rank) {
                row_offsets = new int[procs + 1];
                row_offsets[0] = 0;
                for (int i = 0; i < procs; ++i) {
                    row_offsets[i + 1] = row_offsets[i] + row_counts[i];
                }
            }
            
            // allocate output
            aligned_matrix<FloatType> output;
            if (rank == target_rank) {
                // allocate final
                output = std::move(aligned_matrix<FloatType>(row_offsets[procs] , _cols, _align));
            }

            // gatherv, row by row.
            splash::utils::mpi::datatype<aligned_matrix<FloatType>> row_dt(_cols, _bytes_per_row);
            MPI_Gatherv(_data, _rows, row_dt.value, 
                output.data(), row_counts, row_offsets, row_dt.value, 
                target_rank, comm);

            if (rank == target_rank) {
                delete [] row_counts;
                delete [] row_offsets;
            }      
            // return
            return output;
        }
#else
        aligned_matrix<FloatType> gather(int target_rank = 0) const {
            return *this;
        }
#endif

#ifdef USE_MPI
        aligned_matrix<FloatType> scatter(int src_rank = 0, MPI_Comm comm = MPI_COMM_WORLD) const {
            int procs;
            int rank;
            MPI_Comm_size(comm, &procs);
            MPI_Comm_rank(comm, &rank);

            if (procs == 1) return *this;

            // validate that all columns are same.
            // validate that all alignments are same.
            if (! check_aligned_matrix()) {
                throw std::logic_error("column count or alignment do not match between MPI processes.\n");
            }

            // partition by the number of MPI procs.

            splash::utils::partitioner1D<PARTITION_EQUAL> partitioner;
            int *counts = nullptr;
            int *offsets = nullptr;
            if (rank == src_rank) {
                std::vector<splash::utils::partition<size_type>> parts = partitioner.divide(_rows, procs);

                counts = new int[procs];
                offsets = new int[procs];
                for (int i = 0; i < procs; ++i) {
                    counts[i] = parts[i].size;
                    offsets[i] = parts[i].offset;
                }
            }
            
            splash::utils::partition<size_type> part = partitioner.get_partition(_rows, procs, rank);

            // allocate the data
            aligned_matrix<FloatType> result(part.size, _cols, _align);

            // scatter data.
            splash::utils::mpi::datatype<aligned_matrix<FloatType>> row_dt(_cols, _bytes_per_row);
            MPI_Scatterv(_data, counts, offsets, row_dt.value, 
                result.data(), part.size, row_dt.value, src_rank, comm);
            
            if (rank == src_rank) {
                delete [] counts;
                delete [] offsets;
            }
            return result;
        }
#else
        aligned_matrix<FloatType> scatter(int src_rank = 0) const {
            return *this;
        }
#endif


#ifdef USE_MPI
        aligned_matrix<FloatType> allgather(MPI_Comm comm = MPI_COMM_WORLD) const {
            int procs;
            int rank;
            MPI_Comm_size(comm, &procs);
            MPI_Comm_rank(comm, &rank);

            if (procs == 1) return *this;

            // validate that all columns are same.
            // validate that all alignments are same.
            if (! check_aligned_matrix()) {
                throw std::logic_error("column count or alignment do not match between MPI processes.\n");
            }

            // get total row count
            int rows = _rows;
            int *row_counts = new int[procs];
            splash::utils::mpi::datatype<int> i_dt;
            MPI_Allgather(&rows, 1, i_dt.value, 
                row_counts, 1, i_dt.value, 
                comm);
            
            int *row_offsets = new int[procs + 1];
            row_offsets[0] = 0;
            for (int i = 0; i < procs; ++i) {
                row_offsets[i+1] = row_offsets[i] + row_counts[i];
            }

            // allocate output
            aligned_matrix<FloatType> output(row_offsets[procs], _cols, _align);

            // gatherv.
            splash::utils::mpi::datatype<aligned_matrix<FloatType>> row_dt(_cols, _bytes_per_row);
            MPI_Allgatherv(_data, _rows, row_dt.value, 
                output.data(), row_counts, row_offsets, row_dt.value, 
                comm);

            delete [] row_counts;
            delete [] row_offsets;

            // return
            return output;
        }
#else
        aligned_matrix<FloatType> allgather() const {
            return *this;
        }
#endif

#ifdef USE_MPI
        void allgather_inplace(splash::utils::partition<size_type> const & part, MPI_Comm comm = MPI_COMM_WORLD) {
            int procs;
            int rank;
            MPI_Comm_size(comm, &procs);
            MPI_Comm_rank(comm, &rank);

            if (procs == 1) return;

            // validate that all columns are same.
            // validate that all alignments are same.
            if (! check_aligned_matrix()) {
                throw std::logic_error("column count or alignment do not match between MPI processes.\n");
            }


            // get counts and offsets
            int rows = part.size;
            int *row_counts = new int[procs];
            splash::utils::mpi::datatype<int> i_dt;
            MPI_Allgather(&rows, 1, i_dt.value, 
                row_counts, 1, i_dt.value, 
                comm);
            
            int *row_offsets = new int[procs + 1];
            row_offsets[0] = 0;
            for (int i = 0; i < procs; ++i) {
                row_offsets[i+1] = row_offsets[i] + row_counts[i];
            }

            // check for consistent row counts
            if (row_offsets[procs] != static_cast<int>(_rows)) {
                throw std::logic_error("row count does not match between MPI processes.\n");
            }

            // gatherv.
            splash::utils::mpi::datatype<aligned_matrix<FloatType>> row_dt(_cols, _bytes_per_row);
            MPI_Allgatherv(MPI_IN_PLACE, row_counts[rank], row_dt.value, 
                _data, row_counts, row_offsets, row_dt.value, 
                comm);

            delete [] row_counts;
            delete [] row_offsets;

            // return
        }
#else
        void allgather_inplace(splash::utils::partition<size_type> const & part) {
        }
#endif



        // ---- NOT YET NEEDED -------
#ifdef USE_MPI
        aligned_matrix<FloatType> shift(int rank_distance, MPI_Comm comm = MPI_COMM_WORLD) const {
            int procs;
            int rank;
            MPI_Comm_size(comm, &procs);
            MPI_Comm_rank(comm, &rank);

            if (procs == 1) return *this;

            // validate that all columns are same.
            // validate that all alignments are same.
            if (! check_aligned_matrix()) {
                throw std::logic_error("column count or alignment do not match between MPI processes.\n");
            }

            // sendrecv rows and bytes info.
            size_type rows = 0;
            int dest_rank = (rank + rank_distance) % procs;
            int src_rank = (rank + procs - rank_distance) % procs;
             
            // move size
            splash::utils::mpi::datatype<size_type> s_dt;
            MPI_Sendrecv(&_rows, 1, s_dt.value, dest_rank, 1,
                &rows, 1, s_dt.value, src_rank, 1, 
                comm, MPI_STATUS_IGNORE);

            // allocate
            aligned_matrix<FloatType> output(rows, _cols, _align);

            // move data.
            splash::utils::mpi::datatype<aligned_matrix<FloatType>> row_dt(_cols, _bytes_per_row);
            MPI_Sendrecv(_data, _rows, row_dt.value, dest_rank, 1,
                output.data(), rows, row_dt.value, src_rank, 1,
                comm, MPI_STATUS_IGNORE);

            return output;
        }
#else
        aligned_matrix<FloatType> shift(int rank_distance) const {
            return *this;
        }
#endif

        // aligned_matrix<FloatType> transpose(MPI_Comm comm = MPI_COMM_WORLD) {}

        template <typename TT = FloatType, typename std::enable_if<std::is_arithmetic<TT>::value, int>::type = 1>
        void print(const char* prefix) const {
            const_pointer d;

            for (size_type row = 0; row < _rows; ++row){
                d = this->_get_row(row);
                PRINT_RT("%s: ", prefix);
                for (size_type col = 0; col < _cols; ++col) {
                    PRINT("%f,", d[col]);
                }
                PRINT("\n");
            }
        }
        template <typename TT = FloatType, typename std::enable_if<(!std::is_arithmetic<TT>::value), int>::type = 1>
        void print(const char* prefix) const {
            const_pointer d;
            for (size_type row = 0; row < _rows; ++row){
                d = this->_get_row(row);
                PRINT_RT("%s:", prefix);
                for (size_type col = 0; col < _cols; ++col) {
                    d[col].print("");
                    PRINT(", ");
                }
                PRINT("\n");
            }
        }

};



}
}
