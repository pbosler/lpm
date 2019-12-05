#ifndef LPM_OCTREE_KERNELS_HPP
#define LPM_OCTREE_KERNELS_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmUtilities.hpp"
#include "LpmKokkosUtil.hpp"
#include "LpmBox3d.hpp"
#include "LpmOctreeUtil.hpp"
#include "LpmOctreeLUT.hpp"

#include "Kokkos_Core.hpp"

#include <cstdint>
#include <iostream>
#include <cassert>
#include <cmath>

namespace Lpm {
namespace Octree {

/**
    Compute shuffled xyz key for a point, concatenate point id with key.

    Loop over: Points
*/
struct EncodeFunctor {
    // output
    ko::View<code_type*> codes;
    // input
    ko::View<Real*[3]> pts;
    ko::View<BBox> box;
    Int depth;

    EncodeFunctor(ko::View<code_type*>& co, const ko::View<Real*[3]>& p, const ko::View<BBox>& b, const Int& md) :
        codes(co), pts(p), box(b), depth(md) {}

    KOKKOS_INLINE_FUNCTION
    void operator() (const Index& i) const {
        // each thread i gets a point
        auto pos = ko::subview(pts, i, ko::ALL());
        const key_type key = compute_key_for_point(pos, depth, box());
        codes(i) = encode(key, i);
    }
};

/**
    Using sorted codes (input), move points into sorted order.

    Loop over: point codes
*/
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
        // Each thread gets a point code
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


/**
    Flag, Inclusive Scan Functor.

    Step 1: Flag
        Loop over sorted codes.  Set flag = 1 if new node key is found, 0 otherwise.
    Step 2: Scan
        Scan flags (inclusive)
        After scan, flag(npts-1)+1 = number of unique nodes.

*/
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

/**
    Collect data about each unique node, to be used later to construct the nodes in NodeArray.

    input = output of MarkDuplicates kernel's 2 steps.

    output = array containing unique node keys
        node_ind = flag(i) if flag(i) is a new node.  Otherwise, the thread is idle.
    for each node key, 2 indices:
        inds_out(node_ind, 0) = index of first point (in sorted_pts) contained by node
        inds_out(node_ind, 1) = count of points contained by node
*/
struct UniqueNodeFunctor {
    // output
    ko::View<key_type*> keys_out;
    ko::View<Index*[2]> inds_out;

    // input
    ko::View<Index*> flags;
    ko::View<code_type*> codes_in;

    UniqueNodeFunctor(ko::View<key_type*>& oc, ko::View<Index*[2]>& io,
    	const ko::View<Index*>& f, const ko::View<code_type*>& ic) :
        flags(f), codes_in(ic), keys_out(oc), inds_out(io) {}

    KOKKOS_INLINE_FUNCTION
    void operator () (const Index& i) const {
        // Each thread gets an index into flags
        bool newnode = true;
        if (i > 0) newnode = (flags(i) > flags(i-1));
        if (newnode) {
            // thread finds a new node
        	const Index node_ind = flags(i)-1;
        	const key_type newkey = decode_key(codes_in(i));
            keys_out(node_ind) = newkey;
            const Index first = binarySearchCodes(newkey, codes_in, true);
            const Index last = binarySearchCodes(newkey, codes_in, false);
            inds_out(node_ind,0) = first;
            inds_out(node_ind,1) = last - first + 1;
        }
        // else thread is idle
    }
};


/**
    For later nearest-neighbor searches, we want to add the siblings of every unique node, even if they're empty,
    so that each parent will have a full set of 8 child nodes.

    2-step Flag, Scan

    Loop over: unique nodes

    Step 1: Flag
    For thread i, where i>0:
        If thread i and thread i-1 have the same parent, flag node_num = 0
        If thread i and thread i-1 have different parents, flag node_num = 8
    Step 3: Scan (inclusive)

*/
struct NodeSiblingCounter {
    // output
	ko::View<Index*> nsiblings;

	// input
	ko::View<key_type*> keys_in;
	Int lev;
	Int max_depth;

	NodeSiblingCounter(ko::View<Index*> na, const ko::View<key_type*>& kk, const Int& ll, const Int& md) :
        nsiblings(na), keys_in(kk), lev(ll), max_depth(md) {}

	struct MarkTag {};
	struct ScanTag {};

	KOKKOS_INLINE_FUNCTION
	void operator () (const MarkTag&, const Index& i) const {
		if (i>0) {
			const key_type pt_i = parent_key(keys_in(i), lev, max_depth);
			const key_type pt_im1 = parent_key(keys_in(i-1), lev, max_depth);
            nsiblings(i) = (pt_i == pt_im1 ? 0 : 8);
		}
		else {
		    nsiblings(i) = 8;
		}
	}

	KOKKOS_INLINE_FUNCTION
	void operator() (const ScanTag&, const Index& i, Index& ct, const bool& final_pass) const {
		const Index inc = nsiblings(i);
        ct += inc;
		if (final_pass) {
			nsiblings(i) = ct;
		}
	}
};

/**
    Prepare NodeArrayD construction.

    Loop over: unique nodes

    For each unique parent, construct full set of 8 children; some of them may be empty (contain no points)
*/
struct NodeArrayDFunctor {
    // output
    ko::View<key_type*> node_keys;
    ko::View<Index*[2]> node_pt_inds;
    ko::View<Index*> node_parents;
    ko::View<Index*> pt_in_node;
    // input
    ko::View<Index*> nsiblings;
    ko::View<key_type*> ukeys;
    ko::View<Index*[2]> uinds;
    Int max_depth;

    NodeArrayDFunctor(ko::View<key_type*>& nk, ko::View<Index*[2]>& np, ko::View<Index*>& nprts,
        ko::View<Index*>& pinn, const ko::View<Index*>& ns, const ko::View<key_type*>& uk,
        const ko::View<Index*[2]>& ui, const Int& d) : node_keys(nk), node_pt_inds(np),
        node_parents(nprts), pt_in_node(pinn), nsiblings(ns), ukeys(uk), uinds(ui), max_depth(d) {}

    KOKKOS_INLINE_FUNCTION
    void operator () (const Index& i) const {
        bool new_parent = true;
        if (i>0) new_parent = (nsiblings(i) > nsiblings(i-1));
        if (new_parent) {
            const Index kid0_address = nsiblings(i)-8;
            const key_type pkey = parent_key(ukeys(i), max_depth, max_depth);
            for (int j=0; j<8; ++j) {
                const Index node_ind = kid0_address + j;
                const key_type new_key = pkey + j;
                node_keys(node_ind) = new_key;
                node_parents(node_ind) = NULL_IND;
                const Index found_key = binarySearchKeys(new_key, ukeys, true);
                if (found_key != NULL_IND) {
                    const Index points_start_ind = uinds(found_key,0);
                    const Index points_count = uinds(found_key,1);
                    node_pt_inds(node_ind, 0) = points_start_ind;
                    node_pt_inds(node_ind, 1) = points_count;
                    for (Index k=points_start_ind; k<points_start_ind+points_count; ++k) {
                        pt_in_node(k) = node_ind;
                    }
                }
                else {
                    node_pt_inds(node_ind,0) = NULL_IND;
                    node_pt_inds(node_ind,1) = 0;
                }
            }
        }
    }
};

/**
    Collect unique parents from lower level

    Loop over: lower level nodes, but only work on every 8th one

    nparents = nkeys_from_lower / 8;
*/
struct ParentNodeFunctor {
    // output
    ko::View<key_type*> keys_out;
    ko::View<Index*[2]> inds_out;
    // input
    ko::View<key_type*> keys_from_lower;
    ko::View<Index*[2]> inds_from_lower;
    Int level;
    Int lower_level;
    Int max_depth;

    ParentNodeFunctor(ko::View<key_type*>& ko, ko::View<Index*[2]>& io, const ko::View<key_type*>& kl,
        const ko::View<Index*[2]>& il, const Int& lev, const Int& md) :
        keys_out(ko), inds_out(io), keys_from_lower(kl), inds_from_lower(il),
        level(lev), lower_level(lev+1), max_depth(md) {}

    KOKKOS_INLINE_FUNCTION
    void operator() (const Index& i) const {
        // i in [0, nparents-1]
        const Index my_first_kid = 8*i; // address of first kid in lower level arrays
        const key_type my_key = parent_key(keys_from_lower(my_first_kid), lower_level, max_depth);
        keys_out(i) = my_key;
        inds_out(i,1) = 0;
        for (int j=0; j<8; ++j) {
            inds_out(i,1) += inds_from_lower(my_first_kid+j,1);
        }
        assert(inds_out(i,1)>0);
        for (int j=0; j<8; ++j) {
            if (inds_from_lower(my_first_kid+j,0) != NULL_IND) {
                inds_out(i,0) = inds_from_lower(my_first_kid+j,0);
                break;
            }
        }
    }
};

struct NodeArrayInternalFunctor {
    // output
    ko::View<key_type*> node_keys;
    ko::View<Index*[2]> node_pt_inds;
    ko::View<Index*> node_parents;
    ko::View<Index*[8]> node_kids;
    ko::View<Index*> parents_from_lower;
    // input
    ko::View<Index*> nsiblings;
    ko::View<key_type*> ukeys;
    ko::View<Index*[2]> uinds;
    ko::View<key_type*> keys_from_lower;
    Int level;
    Int max_depth;

    NodeArrayInternalFunctor(ko::View<key_type*>& nkeys, ko::View<Index*[2]>& npi, ko::View<Index*>& npts,
        ko::View<Index*[8]>& nkids, ko::View<Index*>& plow, const ko::View<Index*>& nsibs,
        const ko::View<key_type*>& uk, const ko::View<Index*[2]>& ui, const ko::View<key_type*>& klow,
        const Int& lev, const Int& max) : node_keys(nkeys), node_pt_inds(npi), node_parents(npts),
        node_kids(nkids), parents_from_lower(plow), nsiblings(nsibs), ukeys(uk), uinds(ui),
        keys_from_lower(klow), level(lev), max_depth(max) {}

    KOKKOS_INLINE_FUNCTION
    void operator() (const Index& i) const {
        bool new_parent = true;  // true if nodes at this level have different parents
        if (i>0) new_parent = (nsiblings(i) > nsiblings(i-1));
        if (new_parent) {
            const key_type pkey = parent_key(ukeys(i), level, max_depth); // key of common parent at next level up
            const Index kid0_address = nsiblings(i)-8;  // index of new parent's first child at this level
            for (int j=0; j<8; ++j) {
                const Index node_ind = kid0_address + j; // index of new node at this level
                const key_type new_key = node_key(pkey, j, level, max_depth); // key of new node at this level
                node_keys(node_ind) = new_key;
                node_parents(node_ind) = NULL_IND;
                const Index found_key = binarySearchKeys(new_key, ukeys, true);
                if (found_key != NULL_IND) {  // this sibling is nonempty
                    node_pt_inds(node_ind,0) = uinds(found_key,0);
                    node_pt_inds(node_ind,1) = uinds(found_key,1);
                    const Index kid0_lower = binarySearchKeys(new_key, keys_from_lower, true); // index of first child at lower level
                    for (int k=0; k<8; ++k) {
                        node_kids(node_ind,k) = kid0_lower + k;
                        parents_from_lower(kid0_lower+k) = node_ind;
                    }
                }
                else { // this sibling is empty
                    node_pt_inds(node_ind,0) = NULL_IND;
                    node_pt_inds(node_ind,1) = 0;
                    for (int k=0; k<8; ++k) {
                        node_kids(node_ind, k) = NULL_IND;
                    }
                }

            }
        }
    }
};

/** Listing 2 from Data Parallel Octree paper */
struct NeighborhoodFunctor {
    // output
    ko::View<Index*[27]> neighbors;
    // input
    ko::View<key_type*> keys;
    ko::View<Index*[8]> kids;
    ko::View<Index*> parents;
    ko::View<ParentLUT> ptable;
    ko::View<ChildLUT> ctable;
    Int level;
    Int max_depth;

    NeighborhoodFunctor(ko::View<Index*[27]>& n, const ko::View<key_type*>& k, const ko::View<Index*[8]>& c,
        const ko::View<Index*>& p, const Int& l, const Int& m) :
        neighbors(n), keys(k), kids(c), parents(p),level(l), max_depth(m),
        ptable("ParentLUT"), ctable("ChildLUT") {
        	assert(l>0);
        	assert(l<=m);}

	KOKKOS_INLINE_FUNCTION
	void operator() (const Index& t) const {
		const Index p = parents(t);
		const key_type i = local_key(keys(t), level, max_depth);
		for (int j=0; j<27; ++j) {
			const Index plut = table_val(i,j, ptable);
			const Index h = neighbors(p,plut);
			neighbors(t,j) = (h != NULL_IND ? kids(h, table_val(i,j,ctable)) : NULL_IND);
		}
	}
};

struct VertexSetupFunctor {
	// output
	ko::View<Index*[8]> owner;
	ko::View<Int*[8]> flags;
	ko::View<Int*> nverts_at_node;
	ko::View<Index*> address;
	// input
	ko::View<key_type*> keys;
	ko::View<Index*[27]> neighbors;
	Index level_offset;
	// local
	ko::View<NeighborsAtVertexLUT> nvtable;

	struct OwnerTag {};
	struct ReduceTag {};
	struct ScanTag {};

	VertexSetupFunctor(ko::View<Index*[8]>& o, ko::View<Int*[8]>& f, ko::View<Int*>& nvan, ko::View<Index*>& a,
		const ko::View<key_type*>& k, const ko::View<Index*[27]>& nn, const Index& lo) : owner(o), flags(f), nverts_at_node(nvan),
			address(a), keys(k), neighbors(nn), level_offset(lo), nvtable("NeighborsAtVertexLUT") {}

	KOKKOS_INLINE_FUNCTION
	void operator() (const OwnerTag&, const Index& t) const {
		Int nv = 0;
		for (int i=0; i<8; ++i){ // loop over node t's vertices (future: this loop can be flattened)
			key_type owner_key = keys(t);
			owner(t,i) = t;
			for (int j=0; j<8; ++j) { // loop over nodes at vertex
				const Index nbr_ind = neighbors(t, table_val(i,j,nvtable));
				if (nbr_ind != NULL_IND) {
					const key_type nbr_key = keys(nbr_ind);
					if (nbr_key < owner_key) {
						owner_key = nbr_key;
						owner(t,i) = nbr_ind;
					}
				}
			}
			if (owner(t,i) == t) {
				flags(t,i) = 1;
				++nv;
			}
			else {
				flags(t,i) = 0;
			}
		}
		nverts_at_node(t) = nv;
	}

	KOKKOS_INLINE_FUNCTION
	void operator() (const ReduceTag&, const Index& t, Index& ct) const {
		ct += nverts_at_node(t);
	}


	KOKKOS_INLINE_FUNCTION
	void operator() (const ScanTag&, const Index& t, Int& ct, const bool& final_pass) const {
		const Int old_val = nverts_at_node(t);
		if (final_pass) {
			address(t) = ct + level_offset;
		}
		ct += old_val;
	}
};

struct VertexFunctor {
	// output
	ko::View<Index*[8]> owner;
	ko::View<Int*[8]> flags;
	ko::View<Int*> nverts_at_node;
	ko::View<Index*> vertex_address;
	ko::View<Index*[8]> vertex_nodes;
	ko::View<Index*[8]> node_vertices;
	// input
	ko::View<key_type*> keys;
	ko::View<Index*[27]> neighbors;
	// local
	ko::View<NeighborsAtVertexLUT> nvtable;


	struct BuildTag {};
	struct ConnectTag {};

	VertexFunctor(ko::View<Index*[8]>& o, ko::View<Int*[8]>& f, ko::View<Int*>& nnv, ko::View<Index*>& va, ko::View<Index*[8]>& vn,
		ko::View<Index*[8]>& nv, const ko::View<Index*[27]>& nn) :
		owner(o), flags(f), nverts_at_node(nnv), vertex_address(va), vertex_nodes(vn), node_vertices(nv),
		neighbors(nn), nvtable("NeighborsAtVertexLUT") {}



	KOKKOS_INLINE_FUNCTION
	void operator() (const BuildTag&, const Index& t) const {
		const Index start_ind = vertex_address(t);
		Index loc_ind = 0;
		for (int i=0; i<8; ++i) {
			if (flags(t,i)) { // node t owns its ith vertex
				node_vertices(t,i) = start_ind + loc_ind;
				// build the vertex
				for (int j=0; j<8; ++j) {
					vertex_nodes(start_ind+loc_ind,j) = neighbors(t, table_val(i,j, nvtable));
				}
				loc_ind++;
			}
		}
	}

	KOKKOS_INLINE_FUNCTION
	void operator() (const ConnectTag&, const Index& t) const {
		for (int i=0; i<8; ++i) {
			if (!flags(t,i)) {
				const Index vert_owner = owner(t,i);
				Index nbr_ind = NULL_IND;
				Int nbr_ct = 0;
				for (int j=0; j<8; ++j) {
					if (neighbors(t,table_val(i,j, nvtable)) == vert_owner) {
						nbr_ind = table_val(i,j,nvtable);
						++nbr_ct;
					}
				}
				assert(nbr_ct == 1);
				node_vertices(t,i) = node_vertices(neighbors(t,nbr_ind), 7-i);
			}
		}
	}
};

}}
#endif
