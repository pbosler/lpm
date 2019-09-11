#ifndef LPM_OCTREE_UTIL_HPP
#define LPM_OCTREE_UTIL_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmUtilities.hpp"
#include "LpmKokkosUtil.hpp"
#include "Kokkos_Core.hpp"
#include <cstdint>

namespace Lpm {
namespace Octree {

typedef uint_fast32_t key_type;
typedef uint_fast32_t id_type;
typedef uint_fast64_t code_type;
typedef ko::View<Real[6]> box_type;
typedef ko::View<Real*[3]> points_view;

/// 2 raised to a nonnegative integer power
template <typename T, typename IntT2=int> KOKKOS_INLINE_FUNCTION
T pintpow2(const IntT2& k) {
    T result = 1;
    for (int i=1; i<=k; ++i) {
        result *= 2;
    }
    return result;
}

/**
    key = x1y1z1x2y2z2...xdydzd
    
    where bits xi, yi, zi, correspond to left (0) and right(1) of the midpoint of the octree
    node i's parent in the x,y,z direction
*/
template <typename CPtType> KOKKOS_INLINE_FUNCTION
key_type compute_key(const CPtType& pos, const int& level_depth) {
    /// assume root box is [-1,1]^3
    Real cx, cy, cz; // crds of box centroid
    Real half_len = 1.0; // half-length of box edges
    cx = 0;
    cy = 0;
    cz = 0;
    uint_fast32_t key = 0;
    for (int i=0; i<level_depth; ++i) {
        const int pl = 3*(level_depth-1);
        const bool xr = (pos(0) > cx);
        const bool yr = (pos(1) > cy);
        const bool zr = (pos(2) > cz);
        if (xr) {
            key += pintpow2<key_type>(pl-1);
            cx += half_len;
        }
        else {
            cx -= half_len;
        }
        if (yr) {
            key += pintpow2<key_type>(pl-2);
            cy += half_len;
        }
        else {
            cy -= half_len;
        }
        if (zr) {
            key += pintpow2<key_type>(pl-3);
            cz += half_len;
        }
        else {
            cz -= half_len;
        }
        half_len *= 0.5;
    }
    return key;
}

KOKKOS_INLINE_FUNCTION
uint64_t encode(const uint64_t key, const uint32_t id) {
    return ((key<<32) + id);
}

KOKKOS_INLINE_FUNCTION
uint32_t decode_id(const uint64_t& code) {
    return uint32_t(code);
}

struct BBox {
    Real bds[6];
    
    KOKKOS_INLINE_FUNCTION
    BBox() {init();}
    
    KOKKOS_INLINE_FUNCTION
    BBox(const BBox& rhs) {
        for (int i=0; i<6; ++i) {
            bds[i] = rhs.bds[i];
        }
    }
    
    KOKKOS_INLINE_FUNCTION
    void init() {
        for (int i=0; i<6; ++i) {
            bds[i] = (i%2 == 0 ? ko::reduction_identity<Real>::min() : ko::reduction_identity<Real>::max());
        }
    }
    
    KOKKOS_INLINE_FUNCTION
    BBox(const Real* b) {
        for (int i=0; i<6; ++i) {
            bds[i] = b[i];
        }
    }
    
    KOKKOS_INLINE_FUNCTION
    BBox(const ko::View<Real[6]> bv) {
        for (int i=0; i<6; ++i) {
            bds[i] = bv(i);
        }
    }
    
    KOKKOS_INLINE_FUNCTION
    BBox(const ko::View<Real[6],Host> bv) {
        for (int i=0; i<6; ++i) {
            bds[i] = bv(i);
        }
    }
};

template <typename Space> 
struct BBoxReducer {
    public:
    typedef BBoxReducer reducer;
    typedef BBox value_type;
    typedef ko::View<value_type[1],Space,ko::MemoryUnmanaged> result_view_type;
    
    private:
        value_type& value;
        bool references_scalar_v;
    
    public:
    KOKKOS_INLINE_FUNCTION
    BBoxReducer(value_type& val) : value(val), references_scalar_v(true) {}
            
    KOKKOS_INLINE_FUNCTION
    void join(value_type& dest, const value_type& src) const {
        for (int i=0; i<6; i+=2) {
            if (src.bds[i] < dest.bds[i]) dest.bds[i] = src.bds[i];
        }
        for (int i=1; i<6; i+=2) {
            if (src.bds[i] > dest.bds[i]) dest.bds[i] = src.bds[i];
        }
    }
    
    KOKKOS_INLINE_FUNCTION
    void join(volatile value_type& dest, const volatile value_type& src) const {
        for (int i=0; i<6; i+=2) {
            if (src.bds[i] < dest.bds[i]) dest.bds[i] = src.bds[i];
        }
        for (int i=1; i<6; i+=2) {
            if (src.bds[i] > dest.bds[i]) dest.bds[i] = src.bds[i];
        }
    }
    
    KOKKOS_INLINE_FUNCTION
    void init(value_type& val) const {
        val.init();
    }
    
    KOKKOS_INLINE_FUNCTION
    value_type& reference() const {return value;}
    
    KOKKOS_INLINE_FUNCTION
    result_view_type view() const {
       return result_view_type(&value);
    }
    
    KOKKOS_INLINE_FUNCTION
    bool references_scalar() const {return references_scalar_v;}
};

template <typename Space=DevExe> KOKKOS_FUNCTION
BBox get_bbox(const ko::View<Real*[3],Space>& pts) {
    BBox result;
    ko::parallel_reduce(pts.extent(0), KOKKOS_LAMBDA (const Index& i, BBox& b) {
        if (pts(i,0) < b.bds[0]) b.bds[0] = pts(i,0);
        if (pts(i,0) > b.bds[1]) b.bds[1] = pts(i,0);
        if (pts(i,1) < b.bds[2]) b.bds[2] = pts(i,1);
        if (pts(i,1) > b.bds[3]) b.bds[3] = pts(i,1);
        if (pts(i,2) < b.bds[4]) b.bds[4] = pts(i,2);
        if (pts(i,2) > b.bds[5]) b.bds[5] = pts(i,2);
    }, BBoxReducer<Space>(result));
    return result;
}

}
}
#endif
