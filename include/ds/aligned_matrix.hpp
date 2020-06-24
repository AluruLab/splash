/*
 *  aligned_matrix.hpp
 *
 *  Created on: June 11, 2020
 *  Author: Tony Pan
 *  Affiliation: Institute for Data Engineering and Science
 *  			Georgia Institute of Technology, Atlanta, GA 30332
 */

#pragma once  // instead of #ifndef ...

#include "utils/memory.hpp"
#include "utils/partition.hpp"

#ifdef WITH_MPI
#include "mpi.h"
#include "utils/mpi_types.hpp"
#endif

// forward declare
namespace splash { namespace ds { 
template<typename FloatType>
class aligned_matrix;
}}

#ifdef WITH_MPI
namespace splash { namespace utils { namespace mpi {

// define data type for 1 row.
template <typename T> 
struct datatype<splash::ds::aligned_matrix<T>, false> {
    datatype(size_t const & count, size_t const & row_bytes) {
        splash::utils::mpi::datatype<T> dt1;
        splash::utils::mpi::datatype<unsigned char> dt2;
        MPI_Datatype type[2] = {
            dt1.value,
            dt2.value
        }
        int blocklen[2] = {count, row_bytes - count * sizeof(T)};
        MPI_Aint disp[2] = {
            0,
            count * sizeof(T)
        }
        MPI_Type_create_struct(2, blocklen, disp, type, &value);
        MPI_Type_commit(&value);
    }
    ~datatype() {
        MPI_Type_free(&value);
    }
    MPI_Datatype value;
};

}}}
#endif

namespace splash { namespace ds { 

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

    protected:
        void* _data; // internal pointer.
        size_t _rows;   // number of vectors in the data
        size_t _cols;   // vector length
        size_t _align;  // alignment
        size_t _bytes_per_row;   // assume row major
        bool manage;   // owning _data or not.

        inline pointer _get_row(size_t const & row = 0) {
            return reinterpret_cast<pointer>(_data + row * _bytes_per_row);
        }
        inline const_pointer _get_row(size_t const & row = 0) const {
            return reinterpret_cast<const_pointer>(_data + row * _bytes_per_row);
        }

    public:
        aligned_matrix() : 
            _data(nullptr), _rows(0), _cols(0), _align(splash::utils::get_cacheline_size()), _bytes_per_row(0), manage(true) {}
        
        // construct, optionally with allocated data.
        // alignment of 0 indicates: use system's cacheline size.
        aligned_matrix(size_t const & rows, size_t const & cols, size_t const & align = 0, 
                        void* data = nullptr, bool const & copy=true) :
            _rows(rows), _cols(cols), _align(align == 0 ? splash::utils::get_cacheline_size() : align),
            _bytes_per_row(splash::utils::get_aligned_size(cols * sizeof(FloatType), _align)),
            manage(copy)
        {
            if (manage) {
                _data = splash::utils::aligned_alloc_2D(_rows, _cols * sizeof(FloatType), _align);  // total size is multiple of alignment.
                memset(_data, 0, allocated());
            }
            if (data)
                if (copy)
                   memcpy(_data, data, _rows * _bytes_per_row);
                else _data = data;
        }
        // copy constructor and assignment.  deep copy.
        aligned_matrix(aligned_matrix const & other) : 
            aligned_matrix(other._rows, other._cols, other._align, other._data, other.manage) {}
        aligned_matrix & operator=(aligned_matrix const & other) {
            if (!other.manage) 
                _data = other._data;
            else {
                // not same size.  free and reallocate.
                if (allocated() != other.allocated()) {
                    if (_data) splash::utils::aligned_free(_data);
                    _data = splash::utils::aligned_alloc(other.allocated(), other._align);  // total size is multiple of alignment.
                    
                memcpy(_data, other._data, other.allocated());
            }
            _rows = other._rows;
            _cols = other._cols;
            _align = other._align;
            _bytes_per_row = other._bytes_per_row;
            manage = other.manage;
        }
        // move constructor.  take ownership.
        aligned_matrix(aligned_matrix && other) : aligned_matrix() {
            std::swap(_data, other._data);
            std::swap(_rows, other._rows);
            std::swap(_cols, other._cols);
            std::swap(_align, other._align);
            std::swap(_bytes_per_row, other._bytes_per_row);
            std::swap(manage, other.manage);
        }
        aligned_matrix & operator=(aligned_matrix && other) {
            std::swap(_data, other._data);
            std::swap(_rows, other._rows);
            std::swap(_cols, other._cols);
            std::swap(_align, other._align);
            std::swap(_bytes_per_row, other._bytes_per_row);
            std::swap(manage, other.manage);
        }


        ~aligned_matrix() {
            if (_data && manage) {
                splash::utils::aligned_free(_data);
            }
            _data = nullptr;
        }

        inline size_t size() const {  return _rows * _cols; }
        inline size_t allocated() const {
            return _rows * _bytes_per_row;
        }

        inline size_t rows() const {  return _rows; }
        inline size_t columns() const {  return _cols; }
        inline size_t column_bytes() const {
            return _bytes_per_row;
        }


        inline pointer data(size_t const & row = 0, size_t const & col = 0) noexcept { 
            return _get_row(row) + col; 
        }
        inline const_pointer data(size_t const & row = 0, size_t const & col = 0) const noexcept {
            return _get_row(row) + col; 
        }

        // data value accessor
        inline reference operator()(size_t const & row = 0, size_t const & col = 0) { 
            return _get_row(row)[col]; 
        }
        inline const_reference operator()(size_t const & row = 0, size_t const & col = 0) const { 
            return _get_row(row)[col]; 
        }

        inline reference at(size_t const & row = 0, size_t const & col = 0) { 
            return _get_row(row)[col]; 
        }
        inline const_reference at(size_t const & row = 0, size_t const & col = 0) const { 
            return _get_row(row)[col]; 
        }

        // inline explicit operator FloatType*() { return reinterpret_cast<pointer>(_data); }
        // inline explicit operator FloatType*() const { return reinterpret_cast<const_pointer>(_data); }


	    /*transpose input matrix*/
	    aligned_matrix<FloatType> local_transpose() {
            aligned_matrix<FloatType> output(_cols, _rows, _align);
        
            pointer in;
            /*transpose the matrix*/
            for(size_t row = 0; row < _rows; ++row){
                in = this->_get_row(row);
                for(size_t col = 0; col < _cols; ++col) {
                    output(col, row) = in[col];
                }
            }

        }
        // shallow copy?
        aligned_matrix<FloatType> deep_copy() {
            aligned_matrix<FloatType> output(_rows, _cols, _align);
            memcpy(output._data, _data, allocated());
        }

#ifdef WITH_MPI
        bool check_aligned_matrix(MPI_Comm comm = MPI_COMM_WORLD) {
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
            
#ifdef WITH_MPI
        aligned_matrix<FloatType> gather(int target_rank = 0, MPI_Comm comm = MPI_COMM_WORLD) {
            // validate that all columns are same.
            // validate that all alignments are same.
            if (! check_aligned_matrix()) {
                throw std::logic_error("column count or alignment do not match between MPI processes.\n");
            }

            int procs;
            int rank;
            MPI_Comm_size(comm, &procs);
            MPI_Comm_rank(comm, &rank);

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
#else
        aligned_matrix<FloatType> gather(int target_rank = 0) {
            return *this;
#endif
        }

#ifdef WITH_MPI
        aligned_matrix<FloatType> scatter(int src_rank = 0, MPI_Comm comm = MPI_COMM_WORLD) {
            // validate that all columns are same.
            // validate that all alignments are same.
            if (! check_aligned_matrix()) {
                throw std::logic_error("column count or alignment do not match between MPI processes.\n");
            }

            // partition by the number of MPI procs.
            int procs;
            int rank;
            MPI_Comm_size(comm, &procs);
            MPI_Comm_rank(comm, &rank);

            splash::utils::partitioner1D<size_t> partitioner;
            int *counts = nullptr;
            int *offsets = nullptr;
            if (rank == src_rank) {
                std::vector<splash::utils::partition<size_t>> parts = partitioner.divide(_rows, procs);

                counts = new int[procs];
                offsets = new int[procs];
                for (int i = 0; i < procs; ++i) {
                    counts[i] = parts[i].size;
                    offsets[i] = parts[i].offset;
                }
            }
            
            splash::utils::partition<size_t> part = partitioner.get_partition(_rows, procs, rank);

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
            return results;
#else
        aligned_matrix<FloatType> scatter(int src_rank = 0) {
            return *this;
#endif
        }


#ifdef WITH_MPI
        aligned_matrix<FloatType> allgather(MPI_Comm comm = MPI_COMM_WORLD) {
            // validate that all columns are same.
            // validate that all alignments are same.
            if (! check_aligned_matrix()) {
                throw std::logic_error("column count or alignment do not match between MPI processes.\n");
            }

            int procs;
            int rank;
            MPI_Comm_size(comm, &procs);
            MPI_Comm_rank(comm, &rank);

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
#else
        aligned_matrix<FloatType> allgather() {
            return *this;
#endif
        }

#ifdef WITH_MPI
        void allgather_inplace(splash::utils::partition<size_t> const & part, MPI_Comm comm = MPI_COMM_WORLD) {
            // validate that all columns are same.
            // validate that all alignments are same.
            if (! check_aligned_matrix()) {
                throw std::logic_error("column count or alignment do not match between MPI processes.\n");
            }

            int procs;
            int rank;
            MPI_Comm_size(comm, &procs);
            MPI_Comm_rank(comm, &rank);

            // get counts and offsets
            int rows = part.size;
            int *row_counts = new int[procs];
            splash::utils::mpi::datatype<int> i_dt;
            MPI_Allgather(&rows, 1, i_dt.value, 
                row_counts, 1, i_dt.value, 
                comm);
            
            int *row_offsets = new int[procs + 1];
            byte_offsets[0] = 0;
            for (int i = 0; i < procs; ++i) {
                byte_offsets[i+1] = byte_offsets[i] + byte_counts[i];
            }

            // check for consistent row counts
            if (row_offsets[procs] != _rows) {
                throw std::logic_error("row count does not match between MPI processes.\n");
            }

            // gatherv.
            splash::utils::mpi::datatype<aligned_matrix<FloatType>> row_dt(_cols, _bytes_per_row);
            MPI_Allgatherv(MPI_IN_PLACE, row_counts[rank], row_dt.value, 
                _data, row_counts, row_offsets, row_dt.value, 
                comm);

            delete [] byte_counts;
            delete [] byte_offsets;

            // return
            return output;
#else
        void allgather_inplace(splash::utils::partition<size_t> const & part) {
            return *this;
#endif
        }



        // ---- NOT YET NEEDED -------
#ifdef WITH_MPI
        aligned_matrix<FloatType> shift(int rank_distance, MPI_Comm comm = MPI_COMM_WORLD) {
            // validate that all columns are same.
            // validate that all alignments are same.
            if (! check_aligned_matrix()) {
                throw std::logic_error("column count or alignment do not match between MPI processes.\n");
            }

            int procs;
            int rank;
            MPI_Comm_size(comm, &procs);
            MPI_Comm_rank(comm, &rank);

            // sendrecv rows and bytes info.
            size_t rows = 0;
            int dest_rank = (rank + rank_distance) % procs;
            int src_rank = (rank + proc - rank_distance) % procs;
             
            // move size
            splash::utils::mpi::datatype<size_t> s_dt;
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
#else
        aligned_matrix<FloatType> shift(int rank_distance) {
            return *this;
#endif
        }

        // aligned_matrix<FloatType> transpose(MPI_Comm comm = MPI_COMM_WORLD) {}

        void print() {
            pointer d;
            for (size_t row = 0; row < _rows; ++row){
                d = this->_get_row(row);
                for (size_t col = 0; col < _cols; ++col) {
                    printf("%f,", d[col]);
                }
                printf("\n");
            }
        }

};



}}
