#pragma once

#include <cstring>
#include <vector>
#include <algorithm>
#include <type_traits>

#ifdef USE_MPI
#include <mpi.h>
#include "utils/mpi_types.hpp"
#endif

namespace splash { namespace utils {

// this file contains 1D and 2D partitioners.
//      start with full partitioners (full 1D, and full 2D rectilinear, fixed size blocks or equipartitioned.)
//      the full partitioner produces a linearized list of partitions
//      when partitioning, the parameter can be specified using a partition object as well.
//      
// subsequent filters can be used to rearrange the linear layout and exclude certain parts.

#define PARTITION_EQUAL 1
#define PARTITION_FIXED 2

#define PARTITION_TILE_DIM 8

template <typename ST>
struct partition {
    static_assert( ::std::is_integral<ST>::value, "Partition: supports only integral template parameter" );

    using id_type = typename std::conditional<
        (sizeof(ST) > 2),
        typename std::conditional<
            (sizeof(ST) == 8), int64_t, int32_t>::type,
        typename std::conditional<
            (sizeof(ST) == 2), int16_t, int8_t>::type
        >::type;

    ST offset;
    ST size;
    id_type id;

    partition() = default;
    partition(ST const & _off, ST const & _size, id_type const & _id) :
        offset(_off), size(_size), id(_id) {}
    partition(partition const & other) = default;
    partition(partition && other) = default;
    partition& operator=(partition const & other) = default;
    partition& operator=(partition && other) = default;

    void print(const char * prefix) {
        FMT_PRINT_RT("{} Partition: offset: {}, size: {}, id: {}\n", prefix, offset, size, id);
    }  


};

#ifdef USE_MPI
namespace mpi {

template <typename ST>
struct datatype<partition<ST>, false> {
    datatype() {
        splash::utils::mpi::datatype<ST> dt1;
        splash::utils::mpi::datatype<typename partition<ST>::id_type> dt2;
        MPI_Datatype type[3] = {
            dt1.value,
            dt1.value,
            dt2.value
        };
        int blocklen[3] = {1, 1, 1};
        partition<ST> test {};
        MPI_Aint disp[3]; 
        disp[0] = reinterpret_cast<unsigned char *>(&test.offset) - reinterpret_cast<unsigned char *>(&test);
        disp[1] = reinterpret_cast<unsigned char *>(&test.size) - reinterpret_cast<unsigned char *>(&test);
        disp[2] = reinterpret_cast<unsigned char *>(&test.id) - reinterpret_cast<unsigned char *>(&test);
        MPI_Type_create_struct(3, blocklen, disp, type, &value);
        MPI_Type_commit(&value);
    }
    ~datatype() {
        MPI_Type_free(&value);
    }
    MPI_Datatype value;
};

}
#endif


template <int Strategy>
class partitioner1D;

template <>
class partitioner1D<PARTITION_EQUAL> {

    protected:

        template <typename ST>
        inline ST get_offset(ST const & block, ST const & remainder, int const & id) const {
            return block * id + (static_cast<ST>(id) < remainder ? id : remainder);
        }
        template <typename ST>
        inline ST get_size(ST const & block, ST const & remainder, int const & id) const {
            return block + (static_cast<ST>(id) < remainder ? 1 : 0);
        }

    public:
        // ------------ block partition a 1D range.
        template <typename ST>
        inline std::vector<partition<ST>> divide(ST const & total, ST const & parts) const {
            return divide(partition<ST>(0, total, 0), parts);
        }

        // partitions the given partition.  advance the offset, but does minimal with the id and parts.
        template <typename ST>
        inline std::vector<partition<ST>> divide(partition<ST> const & src, ST const & parts) const {
            ST block = src.size / parts;
            ST remainder = src.size - block * parts;

            std::vector<partition<ST>> partitions;
            partitions.reserve(parts);

            ST i = 0;
            ST offset = src.offset;
            block += 1;
            for (; i < remainder; ++i) {
                partitions.emplace_back(offset, block, i);
                offset += block;
            }
            block -= 1;
            for (; i < parts; ++i) {
                partitions.emplace_back(offset, block, i);
                offset += block;
            }

            return partitions;
        }

        // block partition a 1D range and return the target partition
        template <typename ST>
        inline partition<ST> get_partition(ST const & total, int const & parts, int const & id) const {
            return get_partition(partition<ST>(0, total, 0), parts, id);
        }

        template <typename ST>
        inline partition<ST> get_partition(partition<ST> const & src, int const & parts, int const & id) const {
            ST block = src.size / parts;
            ST remainder = src.size - block * parts;

            return partition<ST>(src.offset + get_offset(block, remainder, id), 
                get_size(block, remainder, id),
                id);
        }

};

template <>
class partitioner1D<PARTITION_FIXED> {

    public:

        // ---------------- fixed size partition a 1D range.
        template <typename ST>
        inline std::vector<partition<ST>> divide(ST const & total, ST const & block) const {
            return divide(partition<ST>(0, total, 0), block);
        }
        template <typename ST>
        inline std::vector<partition<ST>> divide(partition<ST> const & src, ST const & block) const {
            typename partition<ST>::id_type parts = (src.size + block - 1) / block;

            std::vector<partition<ST>> partitions;
            partitions.reserve(parts);

            typename partition<ST>::id_type i = 0;
            ST offset = src.offset;
            for (; i < parts; ++i) {
                partitions.emplace_back(offset, block, i);
                offset += block;
            }
            partitions[parts - 1].size = src.offset + src.size - partitions[parts - 1].offset;

            return partitions;
        }

        // block partition a 1D range and return the target partition
        template <typename ST>
        inline partition<ST> get_partition(ST const & total, ST const & block, int const & id) const {
            return get_partition(partition<ST>(0, total, 0), block, id);
        }
        template <typename ST>
        inline partition<ST> get_partition(partition<ST> const & src, ST const & block, int const & id) const {
            typename partition<ST>::id_type parts = (src.size + block - 1) / block;  // number of parts
            ST offset = block * id;   // offset from src.offset

            return partition<ST>( offset + src.offset, 
                std::min(block, src.size - offset),
                id);
        }

};

// the id is a 
template <typename ST>
struct partition2D {
    static_assert( ::std::is_integral<ST>::value, "Partition: supports only integral template parameter" );

    using part1D_type = partition<ST>;
    using id_type = typename partition<ST>::id_type;

    part1D_type r;
    part1D_type c;
    id_type id;         // linear id
    id_type id_cols;  // id row size.

    partition2D() = default;
    partition2D(part1D_type const & _r, part1D_type const & _c, id_type const & _id, id_type const & _id_cols) :
        r(_r), c(_c), id(_id), id_cols(_id_cols) {}
    partition2D(partition2D const & other) = default;
    partition2D(partition2D && other) = default;
    partition2D& operator=(partition2D const & other) = default;
    partition2D& operator=(partition2D && other) = default;


    void print(const char * prefix) {
        char pre[1024];
        strcpy(pre, prefix);
        strcpy(pre + strlen(prefix), " ROW ");
        r.print(pre);
        strcpy(pre + strlen(prefix), " COL ");
        c.print(pre);
        FMT_PRINT_RT("{} Partition2D id: {}, id row width {}\n", prefix, id, id_cols);
    }  
};

#ifdef USE_MPI
namespace mpi {

template <typename ST>
struct datatype<partition2D<ST>, false> {
    datatype() {
        splash::utils::mpi::datatype<typename partition2D<ST>::part1D_type> dt1;
        splash::utils::mpi::datatype<typename partition2D<ST>::id_type> dt2;
        MPI_Datatype type[4] = {
            dt1.value,
            dt1.value,
            dt2.value,
            dt2.value
        };
        int blocklen[4] = {1, 1, 1, 1};
        partition2D<ST> test {};
        MPI_Aint disp[4];
        disp[0] = reinterpret_cast<unsigned char *>(&test.r) - reinterpret_cast<unsigned char *>(&test);
        disp[1] = reinterpret_cast<unsigned char *>(&test.c) - reinterpret_cast<unsigned char *>(&test);
        disp[2] = reinterpret_cast<unsigned char *>(&test.id) - reinterpret_cast<unsigned char *>(&test);
        disp[3] = reinterpret_cast<unsigned char *>(&test.id_cols) - reinterpret_cast<unsigned char *>(&test);
        MPI_Type_create_struct(4, blocklen, disp, type, &value);
        MPI_Type_commit(&value);
    }
    ~datatype() {
        MPI_Type_free(&value);
    }
    MPI_Datatype value;
};

}
#endif



template <int Strategy>
class partitioner2D;

template <>
class partitioner2D<PARTITION_EQUAL> {

    protected:
        partitioner1D<PARTITION_EQUAL> r_partitioner;
        partitioner1D<PARTITION_EQUAL> c_partitioner;


    public:
        // -------------- block partition a 2D range, rectilinear.
        template <typename ST>
        inline std::vector<partition2D<ST>> divide(ST const & rows, ST const & cols,
            int const & row_parts, int const & col_parts) const {
            return divide(partition2D<ST>(
                        partition<ST>(0, rows, 0),
                        partition<ST>(0, cols, 0),
                        0,
                        1), row_parts, col_parts);
        }

        template <typename ST>
        inline std::vector<partition2D<ST>> divide(partition2D<ST> const & src,
            int const & row_parts, int const & col_parts) const {

            // first partition horizontal and vertical
            std::vector<partition<ST>> r_partitions = r_partitioner.divide(src.r, row_parts);
            std::vector<partition<ST>> c_partitions = c_partitioner.divide(src.c, col_parts);

            int parts = row_parts * col_parts;
            std::vector<partition2D<ST>> partitions;
            partitions.reserve(parts);

            // iterator and make a combined.
            int id = 0;
            for (int i = 0; i < row_parts; ++i) {
                for (int j = 0; j < col_parts; ++j) {
                    partitions.emplace_back(
                        r_partitions[i],
                        c_partitions[j],
                        id++,
                        col_parts);
                }
            }
            return partitions;
        }

        // ----------------- block partition a 2D range rectilinear, and return the target partition
        template <typename ST>
        inline partition2D<ST> get_partition(ST const & rows, ST const & cols,
            int const & row_parts, int const & col_parts,
            int const & id) const {
            return get_partition(partition2D<ST>(
                        partition<ST>(0, rows, 0),
                        partition<ST>(0, cols, 0),
                        0,
                        1), row_parts, col_parts, id);
            }

        template <typename ST>
        inline partition2D<ST> get_partition(partition2D<ST> const & src,
            int const & row_parts, int const & col_parts,
            int const & id) const {

            int r_id = id / col_parts;
            int c_id = id - r_id * col_parts;

            // first partition horizontal and vertical
            partition<ST> r_partition = r_partitioner.get_partition(src.r, row_parts, r_id);
            partition<ST> c_partition = c_partitioner.get_partition(src.c, col_parts, c_id);

            return partition2D<ST>(
                        r_partition,
                        c_partition,
                        id,
                        col_parts);
        }

};

template <>
class partitioner2D<PARTITION_FIXED> {
    protected:
        partitioner1D<PARTITION_FIXED> r_partitioner;
        partitioner1D<PARTITION_FIXED> c_partitioner;

    public:

        // ---------------- fixed size partition a 2D range, rectilinear..
        template <typename ST>
        inline std::vector<partition2D<ST>> divide(ST const & rows, ST const & cols,
            ST const & row_block, ST const & col_block) const {
            return divide(partition2D<ST>(
                        partition<ST>(0, rows, 0),
                        partition<ST>(0, cols, 0),
                        0,
                        1), row_block, col_block);

            }
        template <typename ST>
        inline std::vector<partition2D<ST>> divide(partition2D<ST> const & src,
            ST const & row_block, ST const & col_block) const {

            // first partition horizontal and vertical
            std::vector<partition<ST>> r_partitions = r_partitioner.divide(src.r, row_block);
            std::vector<partition<ST>> c_partitions = c_partitioner.divide(src.c, col_block);

            size_t row_parts = r_partitions.size();
            size_t col_parts = c_partitions.size();
            std::vector<partition2D<ST>> partitions;
            partitions.reserve(row_parts * col_parts);

            // iterator and make a combined.
            int id = 0;
            for (size_t i = 0; i < row_parts; ++i) {
                for (size_t j = 0; j < col_parts; ++j) {
                    partitions.emplace_back(
                        r_partitions[i],
                        c_partitions[j],
                        id++,
                        col_parts);
                }
            }

            return partitions;
        }

        // block partition a 2D range rectilinear, and return the target partition
        template <typename ST>
        inline partition2D<ST> get_partition(ST const & rows, ST const & cols,
            ST const & row_block, ST const & col_block,
            int const & id) const {

            return get_partition(partition2D<ST>(
                        partition<ST>(0, rows, 0),
                        partition<ST>(0, cols, 0),
                        0,
                        1), row_block, col_block, id);
        }
                    

        template <typename ST>
        inline partition2D<ST> get_partition(partition2D<ST> const & src,
            ST const & row_block, ST const & col_block,
            int const & id) const {

            int col_parts = (src.c.size + col_block - 1) / col_block;

            int r_id = id / col_parts;
            int c_id = id - r_id * col_parts;

            partition<ST> r_partition = r_partitioner.get_partition(src.r, row_block, r_id);
            partition<ST> c_partition = c_partitioner.get_partition(src.c, col_block, c_id);

            return partition2D<ST>(
                        r_partition,
                        c_partition,
                        id,
                        col_parts);
        }

};


// ------------- additional partition filters.  


class upper_triangle_filter {
    protected:
        template <typename T>
        inline T get_linear_id(T const & r, T const & c, T const & columns) const {
            if ((c - off_diag) < r) return -1;
            // get rows above.  a full rectangle (r * columns) - lower triangle in rxr
            // r-1 rows, * (r-1 + 1)/2
            T out = ((r + 1) * r) / 2 + r * (columns - off_diag - r);
            // get curr row's col count.
            out += (c - off_diag - r);

            return out;
        }

        int64_t off_diag;

    public:
        upper_triangle_filter(int64_t const & dist_from_diag = 0) : off_diag(dist_from_diag) {}

        template <typename ST>
        inline std::vector<partition2D<ST>> filter(std::vector<partition2D<ST>> const & parts) const {
            if (parts.size() == 0) return parts;
            
            std::vector<partition2D<ST>> selected;
            selected.reserve(parts.size());

            // keep the original r and c coord, as well as orig cols.  change id to be sequential.
            typename partition2D<ST>::id_type id = 0;
            for (auto part : parts) {
                id = get_linear_id(part.r.id, part.c.id, part.id_cols);
                if (id >= 0) {
                    selected.push_back(part);
                    selected.back().id = id;
                }
            }
            return selected;
        }

        // if one that will be filtered out, then id is -1.
        template <typename ST>
        partition2D<ST> filter(partition2D<ST> const & part) const {
            partition2D<ST> output;
            typename partition2D<ST>::id_type id = get_linear_id(part.r.id, part.c.id, part.id_cols);
            if (id >= 0) {
                output = part;
                output.id = id;
            } else {
                output.id = -1;
            }
            return output;
        }
};


// IMPORTANT ASSUMPTION: WIDTH >= HEIGHT
class banded_diagonal_filter {
    // columns is even or odd, need columns/2 +1 per row.
    // with even number of columns, a little less than full band.
    protected:
        // linear id is by row.
        template <typename T>
        inline T get_linear_id(T const & r, T const & c, T const & w, T const & bw) const {
            // do some checks.  ASSUMPTION: height < width
            if (r >= w) FMT_PRINT_RT("ERROR: row id exceed width (==height): {} >= {}\n", r, w);
            if (c >= w) FMT_PRINT_RT("ERROR: col id exceed width: {} >= {}\n", c, w);

            // shifted by -r.  then shift by w to make all positive, then % w to restrict to [0, w)
            // this converts column id to relative to the off-diagonal (0 if main diagonal)
            T c_id = (c + w + w - (r + off_diag)) % w;  // assuming off_diag < w, this is positive.  2 w, 1 each for r and off-diag.
            
            T diag_id = c_id * w + r;  // diagonal first id.  ASSUMES height == w.
            T mx = w * (w + 1) / 2;  // again, assume square.  1/2 of all off-diag, + diag.
            T id = r * bw + c_id;
            id = (diag_id < mx) ? id : -1;

            // FMT_PRINT_RT("r, c->cid = {}, {}->{}, id = {}\n", r, c, c_id, id);
            return id;
        }

        int64_t band_width;
        int64_t off_diag;

    public:
        banded_diagonal_filter(int64_t const & _band_width = std::numeric_limits<int64_t>::max(), 
            int64_t const & _dist_from_diag = 0) : band_width(_band_width), off_diag(_dist_from_diag) {}

        template <typename ST>
        inline std::vector<partition2D<ST>> filter(std::vector<partition2D<ST>> const & parts) const {
            if (parts.size() == 0) return parts;
            
            std::vector<partition2D<ST>> selected;
            selected.reserve(parts.size());

            // keep the original r and c coord, as well as orig cols.  change id to be sequential.
            using id_type = typename partition2D<ST>::id_type;
            id_type id = 0;
            // if even, need 1 + cols/2 in order to cover full matrix
            id_type w = parts.front().id_cols;

            id_type bw = (w / 2 + 1);
            bw = std::min(static_cast<id_type>(this->band_width), bw);
            FMT_ROOT_PRINT("Banded Diagnonal Filter for tiles: width = {},  bandwidth = {}\n", w, bw);

            if (this->off_diag >= w) FMT_PRINT_RT("ERROR: diagonal offset exceed width: {} >= {}\n", this->off_diag, w);
            
            for (auto part : parts) {
                id = get_linear_id(part.r.id, part.c.id, w, bw);
                if (id >= 0) {
                    selected.push_back(part);
                    selected.back().id = id;
                    // selected.back().print("[TEST]");
                }
            }
            return selected;
        }

        // if one that will be filtered out, then id is -1.
        template <typename ST>
        partition2D<ST> filter(partition2D<ST> const & part) const {
            partition2D<ST> output;
            using id_type = typename partition2D<ST>::id_type;
            id_type w = part.front().id_cols;

            if (this->off_diag >= w) FMT_PRINT_RT("ERROR: diagonal offset exceed width: {} >= {}\n", this->off_diag, w);

            id_type bw = (w / 2 + 1);
            bw = std::min(static_cast<id_type>(this->band_width), bw);

            auto id = get_linear_id(part.r.id, part.c.id, w, bw);
            if (id >= 0) {
                output = part;
            }
            output.id = id;
            return output;
        }
};




}}