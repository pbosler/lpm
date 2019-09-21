#ifndef LPM_OCTREE_UTIL_HPP
#define LPM_OCTREE_UTIL_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmUtilities.hpp"
#include "LpmKokkosUtil.hpp"
#include "LpmBox3d.hpp"
#include "Kokkos_Core.hpp"
#include <cstdint>
#include <iostream>
#include <cassert>

namespace Lpm {
namespace Octree {

typedef uint_fast32_t key_type;
typedef uint32_t id_type;
typedef uint_fast64_t code_type;

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
key_type compute_key(const CPtType& pos, const int& level_depth, const int& max_depth) {
	assert(max_depth>=0 && max_depth<=10);
    /// assume root box is [-1,1]^3
    Real cx, cy, cz; // crds of box centroid
    Real half_len = 1.0; // half-length of box edges
    cx = 0;
    cy = 0;
    cz = 0;
    key_type key = 0;
    const Int nbits = 3*max_depth; // key length in bits
    for (int i=1; i<=level_depth; ++i) {
    	const int b = nbits - 3*i;
    	if (pos(2) > cz) {
    		key += std::pow(2,b);
    		cz += half_len;
    	}
    	else {
    		cz -= half_len;
    	}
    	if (pos(1) > cy) {
    		key += std::pow(2,b+1);
    		cy += half_len;
    	}
    	else {
    		cy -= half_len;
    	}
    	if (pos(0) > cx) {
    		key += std::pow(2,b+2);
    		cx += half_len;
    	}
    	else {
    		cx -= half_len;
    	}
    	half_len *= 0.5;
	}	
    return key;
}

KOKKOS_INLINE_FUNCTION
key_type parent_key(const key_type& k, const int& lev, const int& max_depth) {
	const key_type nbits = 3*max_depth;
	const key_type pb = nbits-3*(lev-1);// position of parent's z bit
	key_type mask = 0;
	for (int i=nbits; i>=pb; --i)
		mask += std::pow(2,i); 
	return key_type(k & mask);
}

KOKKOS_INLINE_FUNCTION
code_type encode(const key_type key, const id_type id) {
    code_type result(key);
    return ((result<<32) + id);
}

KOKKOS_INLINE_FUNCTION
id_type decode_id(const code_type& code) {
    return id_type(code);
}

KOKKOS_INLINE_FUNCTION
key_type decode_key(const code_type& code) {
    //code_type result((code-decode_id(code)));
//     code_type result(code);
    return key_type((code>>32));
}

struct PermuteKernel {
    ko::View<Real*[3]> outpts;
    ko::View<Real*[3]> inpts;
    ko::View<uint_fast64_t*> codes;
    
    PermuteKernel(ko::View<Real*[3]>& op, const ko::View<Real*[3]>& ip, const ko::View<uint_fast64_t*>& c) :
        outpts(op), inpts(ip), codes(c) {}
        
    KOKKOS_INLINE_FUNCTION
    void operator() (const id_type& i) const {
        const id_type old_id = decode_id(codes(i));
        for (int j=0; j<3; ++j) {
            outpts(i,j) = inpts(old_id,j);
        }
    }
};

struct MarkDuplicates {
    ko::View<Index*> flags;
    ko::View<code_type*> codes;
    
    MarkDuplicates(ko::View<Index*> f, const ko::View<code_type*>& c) : flags(f), codes(c) {}
    
    struct MarkTag {};
    struct ScanTag {};
    
    KOKKOS_INLINE_FUNCTION
    void operator () (const MarkTag&, const Index& i) const {
        if (i > 0) {
            flags(i) = (decode_key(codes(i)) != decode_key(codes(i-1)));
        }
        else {
            flags(i) = 1;
        }
    }
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const ScanTag&, const Index& i, Index& ct, const bool& final_pass) const {
        const Index old_val = flags(i);
        ct += old_val;
        if (final_pass) {
            flags(i) = ct;
        }
    }
};

template <typename CVT> KOKKOS_INLINE_FUNCTION
Index binarySearch(const key_type& key, const CVT& sorted_codes, const bool& get_first) {
	Index low = 0;
	Index high = sorted_codes.extent(0);
	Index result = -1;
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

struct CopyIfKernel {
    ko::View<Index*> flags;
    ko::View<code_type*> codes_in;
    ko::View<key_type*> keys_out;
    ko::View<Index*[2]> inds_out;
    
    CopyIfKernel(ko::View<key_type*>& oc, ko::View<Index*[2]> io,
    	const ko::View<Index*>& f, const ko::View<code_type*>& ic) : 
        flags(f), codes_in(ic), keys_out(oc), inds_out(io) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator () (const Index& i) const {
        bool newval = true;
        if (i > 0) newval = (flags(i) > flags(i-1));
        if (newval) {
        	const Index node_ind = flags(i)-1;
        	const key_type newkey = decode_key(codes_in(i));
            keys_out(node_ind) = newkey;
            const Index first = binarySearch(newkey, codes_in, true);
            const Index last = binarySearch(newkey, codes_in, false);
            inds_out(node_ind,0) = first;
            inds_out(node_ind,1) = last - first + 1;
        }
    }
};

}
}
#endif
