#ifndef LPM_OCTREE_UTIL_HPP
#define LPM_OCTREE_UTIL_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmUtilities.hpp"
#include "LpmKokkosUtil.hpp"
#include "LpmBox3d.hpp"
#include "Kokkos_Core.hpp"
#include <cstdint>
#include <limits>
#include <iostream>
#include <cassert>
#include <cmath>

namespace Lpm {
namespace Octree {

typedef uint_fast32_t key_type;
typedef uint32_t id_type;
typedef uint_fast64_t code_type;

#ifdef LPM_HAVE_CUDA
#define MAX_OCTREE_DEPTH 10
#else
static constexpr Int MAX_OCTREE_DEPTH = 10;
#endif

/// 2 raised to a nonnegative integer power
template <typename T=key_type, typename IntT2=int> 
KOKKOS_INLINE_FUNCTION
T pintpow2(IntT2 k) {
    assert(k>= 0);
    T result = 1;
    while (k>0) {
        result *= 2;
        --k;
    }
    return result;
}

template <typename T=key_type, typename IntT2=int>
KOKKOS_INLINE_FUNCTION
T pintpow8(IntT2 k) {
    assert(k>=0);
    T result = 1;
    while (k>0) {
        result *= 8;
        --k;
    }
    return result;
}

/**
    key = x1y1z1x2y2z2...xdydzd
    
    where bits xi, yi, zi, correspond to left (0) and right(1) of the centroid of octree
    node i in the x,y,z direction
*/
template <typename CPtType> KOKKOS_INLINE_FUNCTION
key_type compute_key_for_point(const CPtType& pos, const int& depth, BBox bb) {
	assert(depth>0 && depth<=MAX_OCTREE_DEPTH);
    Real cx, cy, cz; // crds of box centroid
    if (std::abs(boxAspectRatio(bb) - 1) > ZERO_TOL) {
        box2cube(bb);
    }
    boxCentroid(cx, cy, cz, bb);
    Real half_len = 0.5*(bb.xmax-bb.xmin); // half-length of box edges
    key_type key = 0;
    const Int nbits = 3*depth; // key length in bits
    for (int i=1; i<=depth; ++i) {
        half_len *= 0.5;
    	const int b = nbits - 3*i; // position of z-bit for level
    	const bool rightz = pos(2) >= cz;
    	const bool righty = pos(1) >= cy;
    	const bool rightx = pos(0) >= cx;
    	key += (rightz ? pintpow2(b) : 0);
    	cz  += (rightz ? half_len : -half_len);
    	key += (righty ? pintpow2(b+1) : 0);
    	cy  += (righty ? half_len : -half_len);
    	key += (rightx ? pintpow2(b+2) : 0);
    	cx  += (rightx ? half_len : -half_len);
	}	
    return key;
}

KOKKOS_INLINE_FUNCTION
key_type parent_key(const key_type& k, const int& lev, const int& max_depth=MAX_OCTREE_DEPTH) {
    assert(max_depth >0 && max_depth <= MAX_OCTREE_DEPTH);
    assert(lev > 0 && lev <= max_depth);
	const key_type nbits = 3*max_depth;
	const key_type pzb = nbits-3*(lev-1);// position of parent's z bit
	key_type mask = 0;
	for (int i=nbits; i>=pzb; --i) // turn on all bits at or higher than pzb
		mask += pintpow2(i); 
	return key_type(k & mask);
}

KOKKOS_INLINE_FUNCTION
key_type local_key(const key_type& k, const int& lev, const Int& max_depth=MAX_OCTREE_DEPTH) {
    assert(max_depth >0 && max_depth <= MAX_OCTREE_DEPTH);
    assert(lev > 0 && lev <= max_depth);
    const key_type nbits = 3*max_depth;
    const key_type pzb = nbits - 3*(lev-1); // position of parent's z bit
    key_type mask = 0;
    for (int i=pzb-3; i<pzb; ++i) // turn on 3 bits lower than pzb
        mask += pintpow2(i);
    return key_type((k & mask)>>(pzb-3)); // shift so result is in [0,7]
}

KOKKOS_INLINE_FUNCTION
BBox box_from_key(const key_type& k, const BBox& rbox, const Int& lev, const Int& max_depth) {
    assert(max_depth >0 && max_depth <= MAX_OCTREE_DEPTH);
    assert(lev > 0 && lev <= max_depth);
    Real cx, cy, cz;
    boxCentroid(cx, cy, cz, rbox);
    Real half_len = 0.5*(rbox.xmax - rbox.xmin);
    for (Int i=1; i<=lev; ++i) {
        half_len *= 0.5;
        const key_type lkey = local_key(k, i, max_depth);
        cz += ((lkey&1) > 0 ? half_len : -half_len);
        cy += ((lkey&2) > 0 ? half_len : -half_len);
        cx += ((lkey&4) > 0 ? half_len : -half_len);
    }
    return BBox(cx-half_len, cx+half_len, cy-half_len, cy+half_len, cz-half_len, cz+half_len);
}


KOKKOS_INLINE_FUNCTION
key_type node_key(const key_type& pk, const key_type& lk, const int& lev, const int& max_depth) {
    const key_type pzb = 3*max_depth - 3*(lev-1);
    const key_type sloc = (lk << (pzb-3));
    return pk + sloc;
}

KOKKOS_INLINE_FUNCTION
code_type encode(const key_type key, const id_type id) {
    assert(id < std::numeric_limits<id_type>::max());
    assert(key < std::numeric_limits<key_type>::max());
    code_type result(key);
    return ((result<<32) + id);
}

KOKKOS_INLINE_FUNCTION
id_type decode_id(const code_type& code) {
    return id_type(code);
}

KOKKOS_INLINE_FUNCTION
key_type decode_key(const code_type& code) {
    return key_type((code>>32));
}

template <typename CVT> KOKKOS_INLINE_FUNCTION
Index binarySearchCodes(const key_type& key, const CVT& sorted_codes, const bool& get_first) {
	Index low = 0;
	Index high = sorted_codes.extent(0)-1;
	Index result = NULL_IND;
	while (low <= high) {
		Index mid = (low + high) / 2;
		const key_type mid_key = decode_key(sorted_codes(mid));
		if (key == mid_key) {
			result = mid;
			if (get_first) {
				high = mid-1;
			}
			else {
				low = mid+1;
			}
		}
		else if (key < mid_key) {
			high = mid-1;
		}
		else {
			low = mid + 1;
		}
	}
	return result;
}

template <typename CVT> KOKKOS_INLINE_FUNCTION
Index binarySearchKeys(const key_type& key, const CVT& sorted_keys, const bool& get_first) {
    Index low = 0;
    Index high = sorted_keys.extent(0)-1;
    Index result = NULL_IND;
    while (low <= high) {
        Index mid = (low+high)/2;
        const key_type mid_key = sorted_keys(mid);
        if (key == mid_key) {
            result = mid;
            if (get_first) {
                high = mid-1;
            }
            else {
                low = mid+1;
            }
        }
        else if (key < mid_key) {
            high = mid-1;
        }
        else {
            low = mid+1;
        }
    }
    return result;
}


}
}
#endif
