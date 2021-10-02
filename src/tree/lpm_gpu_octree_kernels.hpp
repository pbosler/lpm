#ifndef LPM_GPU_OCTREE_KERNELS_HPP
#define LPM_GPU_OCTREE_KERNELS_HPP

#include "LpmConfig.h"
#include "lpm_kokkos_defs.hpp"
#include "tree/lpm_tree_defs.hpp"
#include "tree/lpm_tree_common.hpp"
#include "tree/lpm_gpu_octree_functions.hpp"
#include "tree/lpm_gpu_octree_lookup_tables.hpp"
#include "tree/lpm_box3d.hpp"
#include "lpm_assert.hpp"
#include "lpm_logger.hpp"

#include <bitset>

namespace Lpm {
namespace octree {

/**

  Reference:

   [Z11] K. Zhou, et. al., 2011. Data-parallel octrees for surface reconstruction,
    IEEE Trans. Vis. Comput. Graphics 17(5): 669--681. DOI: 10.1109/TVCG.2010.75
*/


/** @brief compute shuffled xyz key for each point at this octree level,
  pack key and pt idx into a sort code.

  Step 2 of Listing 1 from [Z11].

  use with range policy over number of points

  Output:
    codes: sort codes
  Input:
    pts: xyz coordinates of points
    level: octree level
*/
struct EncodeFunctor {
  // output
  Kokkos::View<code_type*> codes;
  // input
  Kokkos::View<Real*[3]> pts;
  Int level;

  /** @brief constructor.

    @param [in/out] c view to store output sort codes
    @param [in] p points
    @param [in] l level
  */
  EncodeFunctor(Kokkos::View<code_type*> c, const Kokkos::View<Real*[3]> p,
    const Int l) :
    codes(c),
    pts(p),
    level(l) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const id_type i) const {
    const auto pos = Kokkos::subview(pts, i, Kokkos::ALL);
    const auto key = compute_key_for_point(pos, level);
    codes(i) = encode(key, i);
  }
};


/** @brief reorder points according to their sort codes.

  Finishes Step 3 of Listing 1 from [Z11].

  Use with range policy over number of points.

  Output:
    sorted_pts : pts sorted by their octree key
    orig_inds : map of original idx to sorted idx
  Input:
    unsorted_pts : input points
    sort_codes : sorted sort codes (output from EncodeFunctor and sorted)
*/
struct SortPointsFunctor {
  // output
  Kokkos::View<Real*[3]> sorted_pts;
  Kokkos::View<Index*> orig_inds;
  // input
  Kokkos::View<Real*[3]> unsorted_pts;
  Kokkos::View<code_type*> sort_codes;

  SortPointsFunctor(Kokkos::View<Real*[3]> sp,
    Kokkos::View<Index*> inds,
    const Kokkos::View<Real*[3]> up,
    const Kokkos::View<code_type*> c) : sorted_pts(sp),
                                        orig_inds(inds),
                                        unsorted_pts(up),
                                        sort_codes(c) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const id_type i) const {
    const auto old_id = decode_id(sort_codes(i));
    orig_inds(i) = old_id;
    for (auto j=0; j<3; ++j) {
      sorted_pts(i,j) = unsorted_pts(old_id, j);
    }
  }
};


/** @brief reorder sorted points into their original index order

  Inverse of SortPointsFunctor.

  Use with range policy over number of points.

  Output:
    unsorted_pts : pts stored according to their original idx
  Input:
    sorted_pts: pts ordered by sort code (output from SortPointsFunctor)
    orig_id: map from sorted idx to unsorted idx (output from SortPointsFunctor)
*/
struct UnsortPointsFunctor {
  // output
  Kokkos::View<Real*[3]> unsorted_pts;
  // input
  Kokkos::View<Real*[3]> sorted_pts;
  Kokkos::View<Index*> orig_id;

  UnsortPointsFunctor(Kokkos::View<Real*[3]> up,
    const Kokkos::View<Real*[3]> sp,
    const Kokkos::View<Index*> inds) :
    unsorted_pts(up),
    sorted_pts(sp),
    orig_id(inds) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const id_type i) const {
    for (auto j=0; j<3; ++j) {
      unsorted_pts(orig_id(i), j) = sorted_pts(i,j);
    }
  }
};


/** @brief flag unique nodes with a value of 1, duplicate nodes with 0

  use with RangePolicy over number of points

  for step 4 of listing 1, first phase of stream compaction op.
*/
struct MarkUniqueFunctor {
  // output
  Kokkos::View<id_type*> unique_flags;
  // input
  Kokkos::View<code_type*> sort_codes;

  MarkUniqueFunctor(Kokkos::View<id_type*> f, const Kokkos::View<code_type*> c) :
    unique_flags(f),
    sort_codes(c) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index i, Index& ct) const {
    if (i == 0) {
      unique_flags(i) = 1;
    }
    else {
      const auto prev_key = decode_key(sort_codes(i-1));
      const auto key = decode_key(sort_codes(i));
      unique_flags(i) = (key != prev_key);
    }
    ct += unique_flags(i);
  }
};

/** @brief collect unique node info: keys, pt start idx, pt count

  use with range policy over number of points

  for step 4 of listing 1, third (last) phase of stream compaction op.
*/
struct UniqueNodeFunctor {
  // output
  Kokkos::View<key_type*> unique_keys;
  Kokkos::View<Index*> node_pt_idx_start;
  Kokkos::View<id_type*> node_pt_idx_count;
  // input
  Kokkos::View<id_type*> flags_after_scan;
  Kokkos::View<code_type*> sort_codes;

  UniqueNodeFunctor(Kokkos::View<key_type*> k,
                    Kokkos::View<Index*> idx0,
                    Kokkos::View<id_type*> idxn,
                    const Kokkos::View<id_type*> f,
                    const Kokkos::View<code_type*> c) :
    unique_keys(k),
    node_pt_idx_start(idx0),
    node_pt_idx_count(idxn),
    flags_after_scan(f),
    sort_codes(c) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index i) const {
    bool new_node = true;
    if (i>0) {
      new_node = (flags_after_scan(i) > flags_after_scan(i-1));
    }
    if (new_node) {
      const auto n_idx = flags_after_scan(i) - 1;
      const auto n_key = decode_key(sort_codes(i));
      const auto first = binary_search_first(n_key, sort_codes);
      const auto last = binary_search_last(n_key, sort_codes);
      unique_keys(n_idx) = n_key;
      node_pt_idx_start(n_idx) = first;
      node_pt_idx_count(n_idx) = last - first + 1;
    }
  }
};

/** @brief Flag nodes with a new parent with integer value 8, nodes
  with duplicate parent with 0.  Preparation for scan op to count the number
  of leaves.

  Step 5 of Listing 1 from [Z11].

  Use with range policy over the number of unique nodes.
*/
struct MarkUniqueParentFunctor {
  // output
  Kokkos::View<id_type*> node_address;
  // input
  Kokkos::View<key_type*> unique_keys;
  Int level;
  Int max_depth;

  MarkUniqueParentFunctor(Kokkos::View<id_type*> na,
                          const Kokkos::View<key_type*> k,
                          const Int l,
                          const Int md):
    node_address(na),
    unique_keys(k),
    level(l),
    max_depth(md) {}

  KOKKOS_INLINE_FUNCTION
  void operator () (const Index i) const {
    if (i>0) {
      const auto prev_parent = parent_key(unique_keys(i-1), level, max_depth);
      const auto parent = parent_key(unique_keys(i), level, max_depth);
      node_address(i) = (parent == prev_parent ? 0 : 8);
    }
    else {
      node_address(i) = 8;
    }
  }
};


/** @brief create the octree leaf node array

  Use with Kokkos::TeamPolicy with league size = number of unique nodes.

  Step 6 of Listing 1 from [Z11].

  For each unique parent, create a full set of 8 children
*/
struct NodeArrayDFunctor{
  // output
  Kokkos::View<key_type*> node_keys;
  Kokkos::View<Index*> node_pt_idx_start;
  Kokkos::View<id_type*> node_pt_idx_count;
  Kokkos::View<id_type*> node_parents;
  Kokkos::View<id_type*> point_in_node;
  // input
  Kokkos::View<id_type*> node_address;
  Kokkos::View<key_type*> unique_keys;
  Kokkos::View<Index*> unique_idx_start;
  Kokkos::View<id_type*> unique_idx_count;
  Int max_depth;

  NodeArrayDFunctor(Kokkos::View<key_type*> n_k,
                    Kokkos::View<Index*> n_idx0,
                    Kokkos::View<id_type*> n_idxn,
                    Kokkos::View<id_type*> n_p,
                    Kokkos::View<id_type*> p_n,
              const Kokkos::View<id_type*> n_a,
              const Kokkos::View<key_type*> u_k,
              const Kokkos::View<Index*> u_idx0,
              const Kokkos::View<id_type*> u_idxn,
              const Int md) :
    node_keys(n_k),
    node_pt_idx_start(n_idx0),
    node_pt_idx_count(n_idxn),
    node_parents(n_p),
    point_in_node(p_n),
    node_address(n_a),
    unique_keys(u_k),
    unique_idx_start(u_idx0),
    unique_idx_count(u_idxn),
    max_depth(md) {}


  KOKKOS_INLINE_FUNCTION
  void operator() (const member_type team) const {
    bool new_parent = true;
    const auto i = team.league_rank();
    if (i>0) {
      new_parent = (node_address(i) > node_address(i-1));
    }
    if (new_parent) {
      const auto kid0_address = node_address(i)-8;
      const auto pkey = parent_key(unique_keys(i), max_depth, max_depth);

      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 8),
        [=] (const id_type k) {
          const id_type node_idx = kid0_address + k;
          const key_type node_key = pkey + k;
          node_keys(node_idx) = node_key;
          node_parents(node_idx) = NULL_IDX;
          const auto ukey_idx = binary_search_first(node_key, unique_keys);
          const bool found_key = (ukey_idx != NULL_IDX);
          if (found_key) {
            node_pt_idx_start(node_idx) = unique_idx_start(ukey_idx);
            node_pt_idx_count(node_idx) = unique_idx_count(ukey_idx);
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, unique_idx_count(ukey_idx)),
              [=] (const key_type j) {
                point_in_node(unique_idx_start(ukey_idx) + j) = node_idx;
              } );
          }
          else {
            node_pt_idx_start(node_idx) = NULL_IDX;
            node_pt_idx_count(node_idx) = 0;
          }
       });
    }
  }
};

/** @brief Fill-in parent node data for lower level and accumulate child data
  to parent.

  Loop over lower level nodes, but only every 8th one

*/
struct ParentsFunctor {
  // output
  Kokkos::View<key_type*> parent_keys;
  Kokkos::View<Index*> parent_idx_start;
  Kokkos::View<id_type*> parent_idx_count;
  // input
  Kokkos::View<key_type*> keys_lower;
  Kokkos::View<Index*> idx_start_lower;
  Kokkos::View<id_type*> idx_count_lower;
  Int level;
  Int max_depth;

  ParentsFunctor(Kokkos::View<key_type*> pkeys,
                 Kokkos::View<Index*> pidx0,
                 Kokkos::View<id_type*> pidxn,
           const Kokkos::View<key_type*> lkeys,
           const Kokkos::View<Index*> lidx0,
           const Kokkos::View<id_type*> lidxn,
           const Int l,
           const Int md) :
    parent_keys(pkeys),
    parent_idx_start(pidx0),
    parent_idx_count(pidxn),
    keys_lower(lkeys),
    idx_start_lower(lidx0),
    idx_count_lower(lidxn),
    level(l),
    max_depth(md) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const id_type i) const {
    const auto kid0_address = 8*i;
    const auto mkey = parent_key(keys_lower(kid0_address), level+1, max_depth);
    parent_keys(i) = mkey;
    for (auto k=0; k<8; ++k) {
      if (idx_start_lower(kid0_address + k) != NULL_IDX) {
        parent_idx_start(i) = idx_start_lower(kid0_address+k);
        break;
      }
    }
    parent_idx_count(i) = 0;
    for (auto k=0; k<8; ++k) {
      parent_idx_count(i) += idx_count_lower(kid0_address+k);
    }
    LPM_KERNEL_ASSERT(parent_idx_count(i) > 0);
  }

//   KOKKOS_INLINE_FUNCTION
//   void operator() (const member_type team) const {
//     // each team handles 4 parents
//
//     // allocate scratch pad
//     const size_t keys_size = 4*sizeof(key_type);
//     const size_t kids0_size = 4*sizeof(id_type);
//     key_type* shared_keys = (key_type*)team.team_shmem().get_shmem(keys_size);
//     id_type* shared_kids = (id_type*)team.team_shmem().get_shmem(kids0_size);
//
//   }
};

struct NodeArrayInternalFunctor {
  // output
  Kokkos::View<key_type*> node_keys;
  Kokkos::View<Index*> node_idx_start;
  Kokkos::View<id_type*> node_idx_count;
  Kokkos::View<id_type*> node_parents;
  Kokkos::View<id_type*[8]> node_kids;
  Kokkos::View<id_type*> parents_from_lower;
  // input
  Kokkos::View<id_type*> node_address;
  Kokkos::View<key_type*> parent_keys;
  Kokkos::View<Index*> parent_idx0;
  Kokkos::View<id_type*> parent_idxn;
  Kokkos::View<key_type*> keys_from_lower;
  Int level;
  Int max_depth;

  NodeArrayInternalFunctor(Kokkos::View<key_type*> keys,
                           Kokkos::View<Index*> idx0,
                           Kokkos::View<id_type*> idxn,
                           Kokkos::View<id_type*> np,
                           Kokkos::View<id_type*[8]> nk,
                           Kokkos::View<id_type*> lp,
                     const Kokkos::View<id_type*> na,
                     const Kokkos::View<key_type*> pk,
                     const Kokkos::View<Index*> pidx0,
                     const Kokkos::View<id_type*> pidxn,
                     const Kokkos::View<key_type*> lk,
                     const Int l,
                     const Int md) :
    node_keys(keys),
    node_idx_start(idx0),
    node_idx_count(idxn),
    node_parents(np),
    node_kids(nk),
    parents_from_lower(lp),
    node_address(na),
    parent_keys(pk),
    parent_idx0(pidx0),
    parent_idxn(pidxn),
    keys_from_lower(lk),
    level(l),
    max_depth(md) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const id_type i) const {
    bool new_grandparent = true;
    if (i>0) new_grandparent = (node_address(i) > node_address(i-1));
    if (new_grandparent) {
      const auto grandparent_key = parent_key(parent_keys(i), level, max_depth);
      const auto parent0_address = node_address(i)-8;
      for (auto p=0; p<8; ++p) {
        const auto pidx = parent0_address + p;
        const auto pkey = node_key(grandparent_key, p, level, max_depth);
        node_keys(pidx) = pkey;
        node_parents(pidx) = NULL_IDX;
        const auto key_idx = binary_search_first(pkey, parent_keys);
        const bool key_found = (key_idx != NULL_IDX);
        if (key_found) {
          node_idx_start(pidx) = parent_idx0(key_idx);
          node_idx_count(pidx) = parent_idxn(key_idx);
          const auto kid0_idx = binary_search_first(pkey, keys_from_lower);
          LPM_KERNEL_ASSERT(kid0_idx != NULL_IDX);
          for (auto k=0; k<8; ++k) {
            node_kids(pidx,k) = kid0_idx + k;
            parents_from_lower(kid0_idx+k) = pidx;
          }
        }
        else {
          node_idx_start(pidx) = NULL_IDX;
          node_idx_count(pidx) = 0;
          for (auto k=0; k<8; ++k) {
            node_kids(pidx, k) = NULL_IDX;
          }
        }
      }
    }
  }

//   KOKKOS_INLINE_FUNCTION
//   void operator() (const member_type team) const {
//     const auto i = team.league_rank();
//     bool new_grandparent = true;
//     if (i>0) new_grandparent = (node_address(i) > node_address(i-1));
//     if (new_grandparent) {
//       const auto gpkey = parent_key(parent_keys(i), level, max_depth);
//       const auto kid0_address = node_address(i)-8;
//
//       Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 8),
//         [=] (const int j) {
//           const auto new_idx = kid0_address+j;
//           const auto new_key = node_key(gpkey, j, level, max_depth);
//           node_keys(new_idx) = new_key;
//           node_parents(new_idx) = NULL_IDX;
//           const auto key_idx = binary_search_first(new_key, parent_keys);
//           const bool key_found = (key_idx != NULL_IDX);
//           if (key_found) {
//             node_idx_start(new_idx) = parent_idx0(key_idx);
//             node_idx_count(new_idx) = parent_idxn(key_idx);
//             const auto kid0_lower = binary_search_first(new_key, keys_from_lower);
//             LPM_KERNEL_ASSERT(kid0_lower != NULL_IDX);
//             Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, 8),
//              [=] (const int k) {
//               node_kids(new_idx,k) = kid0_lower+k;
//               parents_from_lower(kid0_lower+k) = new_idx;
//             });
//           }
//           else {
//             node_idx_start(new_idx) = NULL_IDX;
//             node_idx_count(new_idx) = 0;
//             Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,8),
//               [=] (const int k) {
//                 node_kids(new_idx,k) = NULL_IDX;
//               });
//           }
//         });
//     }
//   }
};


/** @brief kernel functor for listing 2 in [Z11]. Given neighbor lists
 for all nodes at level l-1 (parents) constructs neighbor lists for all nodes
 at level l.

  Parallel over the number of nodes in a level.
*/
struct NeighborhoodFunctor {
  // output
  Kokkos::View<Index*[27]> node_neighbors;
  // input
  Kokkos::View<key_type*> node_keys;
  Kokkos::View<id_type*> node_parents;
  Kokkos::View<id_type*[8]> node_kids;
  // local
  Kokkos::View<ParentLUT> parent_table;
  Kokkos::View<ChildLUT> child_table;
  Int level;
  Int max_depth;
  id_type base_address;

  /** @brief constructor

    @param [in/out] nn node_neighbors whole octree array
    @param [in] keys node_keys whole octree array
    @param [in] np node_parents whole octree array
    @param [in] nk node_kids whole octree array
    @param [in] l octree level whose neighbors need to be set
    @param [in] md maximum depth of octree
    @param [in] ba first idx of nodes at level l in whole octree array
  */
  NeighborhoodFunctor(Kokkos::View<Index*[27]> nn,
                const Kokkos::View<key_type*> keys,
                const Kokkos::View<id_type*> np,
                const Kokkos::View<id_type*[8]> nk,
                const Int l,
                const Int md,
                const id_type ba) :
    node_neighbors(nn),
    node_keys(keys),
    node_parents(np),
    node_kids(nk),
    parent_table("parent_lookup_table"),
    child_table("child_lookup_table"),
    level(l),
    max_depth(md),
    base_address(ba) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const member_type team) const {
    const auto t = team.league_rank() + base_address;
    const auto i = local_key(node_keys(t), level, max_depth);
    const auto p = node_parents(t);

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 27),
      [=] (const int j) {
        const auto pval = table_val(i, j, parent_table);
        const auto cval = table_val(i, j, child_table);
        const auto h = node_neighbors(p, pval);
        node_neighbors(t, j) = (h==NULL_IDX ? NULL_IDX :
          node_kids(h, cval));
      });
  }
};


/** @brief For each node in a level of the octree, determine the owner node of its 8 vertices in parallel.

*/
struct VertexOwnerFunctor {
  // output
  Kokkos::View<id_type*[8]> owner;
  // input
  Kokkos::View<key_type*> level_keys;
  Kokkos::View<Index*[27]> node_neighbors;
  id_type base_address;
  // local
  Kokkos::View<NeighborsAtVertexLUT> nvtable;

  VertexOwnerFunctor(Kokkos::View<id_type*[8]> lo,
                     Kokkos::View<key_type*> keys,
               const Kokkos::View<Index*[27]> nn,
               const Int ba) :
      owner(lo),
      level_keys(keys),
      node_neighbors(nn),
      base_address(ba),
      nvtable("nvtable") {}



  KOKKOS_INLINE_FUNCTION
  void operator() (const member_type team) const {
    const auto node_idx = base_address + team.league_rank();
    const auto node_level_idx = team.league_rank();
    std::cout << "node_idx = " << node_idx << "\n";
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 8),
      [=] (const int v) {
        key_type owner_key;
        Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, 8),
          [=] (const int j, key_type& okey) {
            const auto local_nbr = table_val(v, j, nvtable);
            const auto nbr_idx = node_neighbors(node_idx, local_nbr);
            if (nbr_idx != NULL_IDX) {
              const auto nbr_level_idx = nbr_idx - base_address;
              const auto nbr_key = level_keys(nbr_level_idx);
              if (nbr_key < okey) okey = nbr_key;
            }
          }, Kokkos::Min<key_type>(owner_key));

        team.team_barrier();

        const auto level_owner = binary_search_first(owner_key, level_keys);
        std::cout << "node: " << node_idx << " vertex " << v << " owner key = " << owner_key
          << "\n";
        LPM_KERNEL_ASSERT(level_owner != NULL_IDX);
        owner(node_level_idx, v) = base_address + level_owner;
      });
  }
};

struct VertexArrayFunctor {
  // output
  Kokkos::View<Real*[3]> vertex_crds;
  Kokkos::View<id_type*[8]> node_vertices;
  // input
  Kokkos::View<id_type*> vertex_address;
  Kokkos::View<id_type*[8]> vertex_owner;
  Kokkos::View<key_type*> node_keys;
  Kokkos::View<Index*[27]> node_neighbors;
  Int level;
  Int max_depth;
  // local
  Kokkos::View<NeighborsAtVertexLUT> nvtable;
  Kokkos::View<ChildLUT> child_table;

  VertexArrayFunctor(Kokkos::View<Real*[3]> vc,
                     Kokkos::View<id_type*[8]> nv,
               const Kokkos::View<id_type*> va,
               const Kokkos::View<id_type*[8]> vo,
               const Kokkos::View<key_type*> keys,
               const Kokkos::View<Index*[27]> nn,
               const Int l,
               const Int md) :
    vertex_crds(vc),
    node_vertices(nv),
    vertex_address(va),
    vertex_owner(vo),
    node_keys(keys),
    node_neighbors(nn),
    level(l),
    max_depth(md),
    nvtable("nvtable"),
    child_table("child_table") {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const member_type team) const {
    const auto node_idx = team.league_rank();

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 8),
      [=] (const int v) {
        // parallel loop over the vertices at this node
        // restrict work to only the vertices owned by this node
        if (vertex_owner(node_idx, v) == node_idx) {
          auto v_ct = 0;
          for (auto i=0; i<v; ++i) {
            if (vertex_owner(node_idx, i) == node_idx) {
              ++v_ct;
            }
          }
          const auto v_idx = vertex_address(node_idx) - 8 + v_ct;
          const auto box = box_from_key(node_keys(node_idx), level, max_depth);
          auto vxyz = Kokkos::subview(vertex_crds, v_idx, Kokkos::ALL);
          box.vertex_crds(vxyz, v);

          Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, 8),
            // parallel loop over the nodes that share this vertex
            [=] (const int j) {
              const auto local_nbr_idx = table_val(v, j, nvtable);
              const auto nbr_idx = node_neighbors(node_idx, local_nbr_idx);
              node_vertices(nbr_idx, table_val(v, local_nbr_idx, child_table)) = v_idx;
            });
        }
      });
  }
};


} // namespace octree
} // namespace Lpm

#endif
