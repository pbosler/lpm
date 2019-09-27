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
#include <cmath>

namespace Lpm {
namespace Octree {

typedef uint_fast32_t key_type;
typedef uint32_t id_type;
typedef uint_fast64_t code_type;

static constexpr Int MAX_OCTREE_DEPTH=10;

/// 2 raised to a nonnegative integer power
template <typename T=key_type, typename IntT2=int> 
KOKKOS_INLINE_FUNCTION
T pintpow2(IntT2 k) {
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
    T result = 1;
    while (k>0) {
        result *= 8;
        --k;
    }
    return result;
}

/**
    key = x1y1z1x2y2z2...xdydzd
    
    where bits xi, yi, zi, correspond to left (0) and right(1) of the midpoint of the octree
    node i's parent in the x,y,z direction
*/
template <typename CPtType> KOKKOS_INLINE_FUNCTION
key_type compute_key(const CPtType& pos, const int& level_depth, const int& max_depth, BBox& bb) {
	assert(max_depth>=0 && max_depth<=10);
    Real cx, cy, cz; // crds of box centroid
    if (std::abs(boxAspectRatio(bb) - 1) > ZERO_TOL) {
        box2cube(bb);
    }
    boxCentroid(cx, cy, cz, bb);
    Real half_len = 0.5*(bb.xmax-bb.xmin); // half-length of box edges
    key_type key = 0;
    const Int nbits = 3*max_depth; // key length in bits
    for (int i=1; i<=level_depth; ++i) {
    	const int b = nbits - 3*i;
    	if (pos(2) > cz) {
            key += pintpow2(b);
    		cz += half_len;
    	}
    	else {
    		cz -= half_len;
    	}
    	if (pos(1) > cy) {
            key += pintpow2(b+1);
    		cy += half_len;
    	}
    	else {
    		cy -= half_len;
    	}
    	if (pos(0) > cx) {
            key += pintpow2(b+2);
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
key_type parent_key(const key_type& k, const int& lev, const int& max_depth=MAX_OCTREE_DEPTH) {
	const key_type nbits = 3*max_depth;
	const key_type pzb = nbits-3*(lev-1);// position of parent's z bit
	key_type mask = 0;
	for (int i=nbits; i>=pzb; --i) // turn on all bits at or higher than pzb
		mask += pintpow2(i); 
	return key_type(k & mask);
}

KOKKOS_INLINE_FUNCTION
key_type local_key(const key_type& k, const int& lev, const Int& max_depth=MAX_OCTREE_DEPTH) {
    key_type nbits = 3*max_depth;
    const key_type pzb = nbits - 3*(lev-1); // position of parent's z bit
    key_type mask = 0;
    for (int i=pzb-3; i<pzb; ++i) // turn on 3 bits lower than pzb
        mask += pintpow2(i);
    return key_type((k & mask)>>(pzb-3));
}

KOKKOS_INLINE_FUNCTION
key_type node_key(const key_type& pk, const key_type& lk, const int& lev, const int& max_depth) {
    const key_type pzb = 3*max_depth - 3*(lev-1);
    const key_type sloc = (lk << (pzb-3));
    return pk + sloc;
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
    return key_type((code>>32));
}

struct EncodeFunctor {
    // output
    ko::View<code_type*> codes;
    // input
    ko::View<Real*[3]> pts;
    ko::View<BBox> box;
    Int level;
    Int max_depth;
    
    EncodeFunctor(ko::View<code_type*>& co, const ko::View<Real*[3]>& p, const ko::View<BBox>& b,
        const Int& lev, const Int& md) :
        codes(co), pts(p), box(b), level(lev), max_depth(md) {}
        
    KOKKOS_INLINE_FUNCTION
    void operator() (const Index& i) const {
        auto pos = ko::subview(pts, i, ko::ALL());
        const key_type key = compute_key(pos, level, max_depth, box());
        codes(i) = encode(key, i);
    }
};

struct PermuteFunctor {
    // output
    ko::View<Real*[3]> outpts;
    ko::View<Index*> orig_inds;
    // input
    ko::View<Real*[3]> inpts;
    ko::View<code_type*> codes;
    
    PermuteFunctor(ko::View<Real*[3]>& op, ko::View<Index*>& oi, const ko::View<Real*[3]>& ip, 
        const ko::View<code_type*>& c) :  outpts(op), orig_inds(oi), inpts(ip), codes(c) {}
        
    KOKKOS_INLINE_FUNCTION
    void operator() (const id_type& i) const {
        const id_type old_id = decode_id(codes(i));
        orig_inds(i) = old_id;
        for (int j=0; j<3; ++j) {
            outpts(i,j) = inpts(old_id,j);
        }
    }
};

struct UnpermuteFunctor {
    // output 
    ko::View<Real*[3]> outpts;
    // input
    ko::View<Real*[3]> inpts;
    ko::View<Index*> old_id;
    
    UnpermuteFunctor(ko::View<Real*[3]>& op, const ko::View<Real*[3]>& ip, const ko::View<Index*>& oi) :
        outpts(op), inpts(ip), old_id(oi) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const Index& i) const {
        for (int j=0; j<3; ++j) {
            outpts(old_id(i),j) = inpts(i,j);
        }
    }
};

struct MarkDuplicates {
    // output
    ko::View<Index*> flags;
    // input
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
Index binarySearchCodes(const key_type& key, const CVT& sorted_codes, const bool& get_first) {
	Index low = 0;
	Index high = sorted_codes.extent(0)-1;
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

template <typename CVT> KOKKOS_INLINE_FUNCTION
Index binarySearchKeys(const key_type& key, const CVT& sorted_keys) {
    Index low = 0;
    Index high = sorted_keys.extent(0)-1;
    Index result = -1;
    while (low <= high) {
        Index mid = (low+high)/2;
        const key_type mid_key = sorted_keys(mid);
        if (key == mid_key) {
            result = mid;
            high = mid-1;
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

struct UniqueNodeFunctor {
    // output
    ko::View<key_type*> keys_out;
    ko::View<Index*[2]> inds_out;
    
    // input
    ko::View<Index*> flags;
    ko::View<code_type*> codes_in;
    
    UniqueNodeFunctor(ko::View<key_type*>& oc, ko::View<Index*[2]> io,
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
            const Index first = binarySearchCodes(newkey, codes_in, true);
            const Index last = binarySearchCodes(newkey, codes_in, false);
            inds_out(node_ind,0) = first;
            inds_out(node_ind,1) = last - first + 1;
        }
    }
};

struct NodeAddressFunctor {
    // output
	ko::View<Index*> node_nums;
	ko::View<Index*> node_address;
	
	// input
	ko::View<key_type*> keys_in;
	Int lev;
	Int max_depth;
	
	NodeAddressFunctor(ko::View<Index*>& nn, ko::View<Index*> na, 
		const ko::View<key_type*>& kk, const Int& ll, const Int& md) :
		node_nums(nn), node_address(na), keys_in(kk), lev(ll), max_depth(md) {}
	
	struct MarkTag {};
	struct ScanTag {};
	
	KOKKOS_INLINE_FUNCTION
	void operator () (const MarkTag&, const Index& i) const {
		if (i>0) {
			const key_type pt_i = parent_key(keys_in(i), lev, max_depth);
			const key_type pt_im1 = parent_key(keys_in(i-1), lev, max_depth);
			if (pt_i == pt_im1) {
				node_nums(i) = 0;
			}
			else {
				node_nums(i) = 8;
			}
		}
	}
	
	KOKKOS_INLINE_FUNCTION
	void operator() (const ScanTag&, const Index& i, Index& ct, const bool& final_pass) const {
		const Index inc = node_nums(i);
		ct += inc;
		if (final_pass) {
			node_address(i) = ct;
		}
		
	}
};

struct NodeSetupFunctor {
    ko::View<key_type*> keys_out;
    ko::View<Index*> node_address;
    ko::View<key_type*> keys_in;
    Int level;
    Int max_depth;
    
    NodeSetupFunctor(ko::View<key_type*>& ko, const ko::View<Index*>& na, const ko::View<key_type*>& ki,
         const Int& lev, const Int& md) :
        keys_out(ko), node_address(na), keys_in(ki), level(lev), max_depth(md) {}

    KOKKOS_INLINE_FUNCTION
    void operator() (const member_type& mbr) const {
        const Index i=mbr.league_rank();
        bool new_parent = true;
        if (i>0) {
            new_parent = node_address(i) > node_address(i-1);
        }
        if (new_parent) {
            const key_type pkey = parent_key(keys_in(i), level, max_depth);
            //const key_type first_kid = 8*i; // bug here
            ko::parallel_for(ko::TeamThreadRange(mbr, 8), KOKKOS_LAMBDA (const Int& j) {
                keys_out(node_address(i)+j) = node_key(pkey, j, level, max_depth);
            });
        }
    }
};

struct NodeFillFunctor {
    // output
    ko::View<Index*> pt_idx;
    ko::View<Index*> pt_ct;
    ko::View<Index*> pt_in_node;
    
    // input
    ko::View<key_type*> keys_in;  // only non-empty keys
    ko::View<Index*> node_address;
    ko::View<Index*[2]> pt_inds;    
    Int level;
    Int max_depth;
    
    NodeFillFunctor(ko::View<Index*>& ps, ko::View<Index*>& pc, ko::View<key_type*>& pn,
        const ko::View<key_type*>& ki, const ko::View<Index*>& na, const ko::View<Index*[2]>& pi, 
        const Int& ll, const Int& md=MAX_OCTREE_DEPTH) : pt_idx(ps), pt_ct(pc), pt_in_node(pn),
        keys_in(ki), node_address(na), pt_inds(pi), level(ll), max_depth(md) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator() (const member_type& mbr) const {
    	const Index i = mbr.league_rank();
    	const key_type address = node_address(i) + local_key(keys_in(i), level, max_depth);
    	pt_idx(address) = pt_inds(i,0);
    	pt_ct(address) = pt_inds(i,1);
    	ko::parallel_for(ko::TeamThreadRange(mbr, pt_inds(i,0), pt_inds(i,0)+pt_inds(i,1)),
    		KOKKOS_LAMBDA (const Index& j) {
    			pt_in_node(j) = address;
    		});
    }
};


}
}
#endif
