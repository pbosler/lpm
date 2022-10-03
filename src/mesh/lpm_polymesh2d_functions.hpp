#ifndef LPM_POLYMESH2D_FUNCTIONS_HPP
#define LPM_POLYMESH2D_FUNCTIONS_HPP

#include "LpmConfig.h"
#include "lpm_assert.hpp"
#include "lpm_geometry.hpp"
#include "lpm_constants.hpp"
#include "lpm_kokkos_defs.hpp"
#include "lpm_coords.hpp"
#include "util/lpm_tuple.hpp"
#include "util/lpm_math.hpp"
#include "util/lpm_string_util.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#include "mesh/lpm_vertices.hpp"
#include "mesh/lpm_edges.hpp"
#include "mesh/lpm_faces.hpp"
#include <memory>

namespace Lpm {

KOKKOS_INLINE_FUNCTION
bool edge_is_positive(const Index edge_idx,
    const Index face_idx,
    const typename Edges::edge_view_type edge_lefts) {
    return face_idx == edge_lefts(edge_idx);
}

template <typename EdgeListType=Index*, typename SeedType> KOKKOS_INLINE_FUNCTION
void get_leaf_edges_from_parent(EdgeListType edge_list,
  Int& nleaves,
  const Index parent_edge_idx,
  const typename Edges::edge_tree_view edge_kids) {

  edge_list[0] = parent_edge_idx;
  nleaves = 1;
  bool keep_going = false;
  if (edge_kids(parent_edge_idx, 0) > 0) {
    keep_going = true;
  }

  while (keep_going) {
    Int n_new = 0;
    for (int i=0; i<nleaves; ++i) {
      if (edge_kids(edge_list[i],0) > 0) {
        const auto kid0 = edge_kids(edge_list[i], 0);
        const auto kid1 = edge_kids(edge_list[i], 1);
        // make room at idx i+1 in the list
        for (int j=i+1; j<nleaves; ++j) {
          edge_list[j+1] = edge_list[j];
        }
        // replace parent edge with its children
        edge_list[i] = kid0;
        edge_list[i+1] = kid1;
        ++n_new;
      }
    }
    nleaves += n_new;
    keep_going = false;
    for (int i=0; i<nleaves; ++i) {
      if (edge_kids(edge_list[i], 0) > 0) {
        keep_going = true;
        break;
      }
    }
  }
}




template <typename EV=Index*, typename SeedType> KOKKOS_INLINE_FUNCTION
void ccw_edges_around_face(EV leaf_edges, Int& n_edges, const Index face_idx,
  const typename Faces<typename SeedType::faceKind,
                       typename SeedType::geo>::edge_view_type face_edges,
  const typename Edges::edge_view_type edge_lefts,
  const typename Edges::edge_tree_view edge_kids) {

  n_edges = 0;
  for (int i=0; i<SeedType::nfaceverts; ++i) {
    Index leaf_edge_list[2*LPM_MAX_AMR_LIMIT];
    Int n_leaves_this_edge;
    get_leaf_edges_from_parent<Index*, SeedType>(leaf_edge_list,
            n_leaves_this_edge,
            face_edges(face_idx,i),
            edge_kids);
    if (edge_is_positive(face_edges(face_idx,i), face_idx, edge_lefts)) {
      for (int j=0; j<n_leaves_this_edge; ++j) {
        leaf_edges[n_edges + j] = leaf_edge_list[j];
      }
    }
    else {
      const int last_idx = n_edges + n_leaves_this_edge - 1;
      for (int j=0; j<n_leaves_this_edge; ++j) {
        leaf_edges[last_idx - j] = leaf_edge_list[j];
      }
    }
    n_edges += n_leaves_this_edge;
  }
}


template <typename FaceListType=Index*, typename SeedType> KOKKOS_INLINE_FUNCTION
void ccw_adjacent_faces(FaceListType adj_face_list,
    Int& n_adj,
    const Index face_idx,
    const typename Faces<typename SeedType::faceKind,
                         typename SeedType::geo>::edge_view_type face_edges,
    const typename Edges::edge_view_type edge_lefts,
    const typename Edges::edge_view_type edge_rights,
    const typename Edges::edge_tree_view edge_kids) {

    Index leaf_edge_list[2*SeedType::nfaceverts*LPM_MAX_AMR_LIMIT];
    Int n_leaf_edges;
    ccw_edges_around_face<Index*,SeedType>(leaf_edge_list, n_leaf_edges, face_idx,
      face_edges, edge_lefts, edge_kids);

    n_adj = n_leaf_edges;

    for (int i=0; i<n_leaf_edges; ++i) {
      if (edge_is_positive(leaf_edge_list[i], face_idx, edge_lefts)) {
        adj_face_list[i] = edge_rights(leaf_edge_list[i]);
      }
      else {
        adj_face_list[i] = edge_lefts(leaf_edge_list[i]);
      }
    }
}

template <typename CV, typename SeedType> KOKKOS_INLINE_FUNCTION
Index locate_point_walk_search(const CV query_pt,
  const Index face_start_idx,
  const typename Edges::edge_view_type edge_lefts,
  const typename Edges::edge_view_type edge_rights,
  const typename Edges::edge_tree_view edge_kids,
  const typename Faces<typename SeedType::faceKind,
                       typename SeedType::geo>::edge_view_type face_edges,
  const typename Coords<typename SeedType::geo>::crd_view_type face_crds) {

  Index face_idx = LPM_NULL_IDX;
  Index current_idx = face_start_idx;
  auto fcrd = Kokkos::subview(face_crds, current_idx, Kokkos::ALL);
  Real dist = SeedType::geo::distance(query_pt, fcrd);

  bool keep_going = true;
  while (keep_going) {
    Index adj_face_list[8*LPM_MAX_AMR_LIMIT];
    Int n_adj;
    face_idx = current_idx;
    ccw_adjacent_faces<Index*, SeedType>(adj_face_list,
        n_adj,
        face_idx,
        face_edges,
        edge_lefts,
        edge_rights,
        edge_kids);
    for (int i=0; i<n_adj; ++i) {
      fcrd = Kokkos::subview(face_crds, adj_face_list[i], Kokkos::ALL);
      Real test_dist = SeedType::geo::distance(query_pt, fcrd);
      if (test_dist < dist) {
        dist = test_dist;
        current_idx = adj_face_list[i];
      }
    }
    keep_going = current_idx != face_idx;
  }
  return face_idx;
}

// KOKKOS_INLINE_FUNCTION
// bool edge_on_boundary(const Index edge_idx,
//   const typename Edges::edge_view_type lefts,
//   const typename Edges::edge_view_type rights) {
//   return (lefts(edge_idx) < 0 or rights(edge_idx) < 0);
// }

template <typename CV, typename SeedType> KOKKOS_INLINE_FUNCTION
Index nearest_root_face(const CV query_pt,
  const typename SeedType::geo::crd_view_type face_crds) {

  Index result = 0;
  const auto xy0 = Kokkos::subview(face_crds, 0, Kokkos::ALL);
  Real dist = SeedType::geo::distance(query_pt, xy0);
  for (int i=1; i<SeedType::nfaces; ++i) {
    const auto txy = Kokkos::subview(face_crds, i, Kokkos::ALL);
    Real test_dist = SeedType::geo::distance(query_pt, txy);
    if (test_dist < dist) {
      dist = test_dist;
      result = i;
    }
  }
  return result;
}


template <typename CV, typename SeedType> KOKKOS_INLINE_FUNCTION
Index locate_point_tree_search(const CV query_pt,
      const Index root_face,
      const typename SeedType::geo::crd_view_type face_crds,
      const typename Faces<typename SeedType::faceKind,
        typename SeedType::geo>::face_tree_view face_kids) {

  bool keep_going = true;
  Index current_idx = root_face;
  Index next_idx = LPM_NULL_IDX;
  Real dist = std::numeric_limits<Real>::max();
  while (keep_going) {
    if (face_kids(current_idx, 0) > 0) {
      for (int k=0; k<4; ++k) {
        const auto fxy = Kokkos::subview(face_crds, face_kids(current_idx, k),
          Kokkos::ALL);
        const Real test_dist = SeedType::geo::distance(query_pt, fxy);
        if (test_dist < dist) {
          next_idx = face_kids(current_idx, k);
          dist = test_dist;
        }
      }
      current_idx = next_idx;
    }
    else {
      keep_going = false;
    }
  }
  return current_idx;
}

template <typename CV, typename SeedType> KOKKOS_INLINE_FUNCTION
Index locate_face_containing_pt(const CV query_pt,
  const typename Edges::edge_view_type edge_lefts,
  const typename Edges::edge_view_type edge_rights,
  const typename Edges::edge_tree_view edge_kids,
  const typename SeedType::geo::crd_view_type face_crds,
  const typename Faces<typename SeedType::faceKind,
                       typename SeedType::geo>::face_tree_view face_kids,
  const typename Faces<typename SeedType::faceKind,
                       typename SeedType::geo>::edge_view_type face_edges) {

  const auto tree_start = nearest_root_face<CV, SeedType>(query_pt, face_crds);
  const auto walk_start = locate_point_tree_search<CV, SeedType>(query_pt,
    tree_start, face_crds, face_kids);

  return locate_point_walk_search<CV, SeedType>(query_pt, walk_start, edge_lefts, edge_rights,
    edge_kids, face_edges, face_crds);
}


} // namespace Lpm

#endif
