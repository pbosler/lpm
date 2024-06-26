#ifndef LPM_OCTREE_LUT_HPP
#define LPM_OCTREE_LUT_HPP

#include <cassert>

#include "Kokkos_Core.hpp"
#include "LpmConfig.h"
#include "LpmDefs.hpp"

namespace Lpm {
namespace Octree {

/**
    These two lookup tables encode the neighbor/child relationships defined by
   the following orderings:

    Parent:        Children/Vertices:           Edges:                  Faces:

        .---------.        .-------.                .-----3----. .-----------.
   .---------. /         /|       /|1 / 3 /|               / |         /| /| /|
   /|         | /         / |      ---------                5  0       11 | /
   /5/    / |         / |         | /         /  |     /| 5 / 7/| /   |       /
   |        /           /  |       ./  |  |0|    |
    .---------.   |    .-------.                .------8----.   2 .-----------.
   |       |   |         | |         |   |    |       |                |
   |___1__|___|       |           ||3||       ||1||         | |         |   / |
   /       |   /       |           |   /       |   |_________| |         |  / 6
   4        9  10       |    |2|    |  /        |  /          / |         | / |
   /         | /         |           | /         | /   /4/    /
    |_________|/            |-------|           ./_____7____./ |/__________|/
   |/_________|/ / 0 / 2 /
                         |--------|
                       |/ 4 / 6 |/
                       .--------.


    Neighbors:  self = 13


          ---------------
         / 2 /  5  / 8  /|
        ---------------
       / 11 / 14 / 17 /|
      ---------------
     / 20 / 23 / 26 / |
    ----------------
    |    |    |    |

          |    |    |    |
          |----|----|----|
        |/ 1 |/  4|/  7|/|
        |----|----|----|
      |/ 10|/ 13|/ 16|/|
      |----|----|----|
    |/ 19|/ 22|/ 25|/|
    |----|----|----|
    |    |    |    |

          |___|____|_____|
         / 0 / 3  / 6   /
        |___|/___|/____|
       / 9 / 12 / 15  /
      |___|____|_____|
     / 18 / 21 / 24 /
    |____|____|____|
*/

/**
    For octree node t at level l whose parent is p, i.e.,

       p = node_parent(t),

    if node t's index in node_kids(p,:) is i, i.e., node t is the ith child of
   p, or,

       i = local_key(node_keys(t), l, max_depth),

    then the index of node t's jth neighbor's parent in node_neighbors(p,:) is
   table(i,j),

        node_parent(node_neighbors(t,j)) = node_neighbors(p, table(i,j))

    ...

    i = local_key of node t, relative to its parent
    j = neighbor of node t, in neighbor ordering [0,27]
    *** the parent of my jth neighbor is the tableth neighbor of my parent ***

*/
struct ParentLUT {
  static constexpr Int nrow_host = 8;
  static constexpr Int ncol_host = 27;
  const Int nrows = 8;
  const Int ncols = 27;
  const Int entries[216] = {
      0,  1,  1,  3,  4,  4,  3,  4,  4,  9,  10, 10, 12, 13, 13, 12, 13, 13,
      9,  10, 10, 12, 13, 13, 12, 13, 13, 1,  1,  2,  4,  4,  5,  4,  4,  5,
      10, 10, 11, 13, 13, 14, 13, 13, 14, 10, 10, 11, 13, 13, 14, 13, 13, 14,
      3,  4,  4,  3,  4,  4,  6,  7,  7,  12, 13, 13, 12, 13, 13, 15, 16, 16,
      12, 13, 13, 12, 13, 13, 15, 16, 16, 4,  4,  5,  4,  4,  5,  7,  7,  8,
      13, 13, 14, 13, 13, 14, 16, 16, 17, 13, 13, 14, 13, 13, 14, 16, 16, 17,
      9,  10, 10, 12, 13, 13, 12, 13, 13, 9,  10, 10, 12, 13, 13, 12, 13, 13,
      18, 19, 19, 21, 22, 22, 21, 22, 22, 10, 10, 11, 13, 13, 14, 13, 13, 14,
      10, 10, 11, 13, 13, 14, 13, 13, 14, 19, 19, 20, 22, 22, 23, 22, 22, 23,
      12, 13, 13, 12, 13, 13, 15, 16, 16, 12, 13, 13, 12, 13, 13, 15, 16, 16,
      21, 22, 22, 21, 22, 22, 24, 25, 25, 13, 13, 14, 13, 13, 14, 16, 16, 17,
      13, 13, 14, 13, 13, 14, 16, 16, 17, 22, 22, 23, 22, 22, 23, 25, 25, 26};
};

/**
    For octree node t at level l with local key i,

        i = local_key(node_keys(t), l, max_depth),

    if node t's jth neighbor has parent h,

        h = node_parents(node_neighbors(t,j)),

    then the jth neighbor has local key table(i,j), or,

        node_neighbors(t,j) = node_kids(h, table(i,j))

    ...

    i = local_key of node t, relative to its parent
    j = neighbor of node t, in neighbor ordering [0,27]
    *** my jth neighbor is the tableth child of its parent ***
*/
struct ChildLUT {
  static constexpr Int nrow_host = 8;
  static constexpr Int ncol_host = 27;
  const Int nrows = 8;
  const Int ncols = 27;
  const Int entries[216] = {
      7, 6, 7, 5, 4, 5, 7, 6, 7, 3, 2, 3, 1, 0, 1, 3, 2, 3, 3, 2, 3, 5, 4, 5,
      1, 0, 1, 6, 7, 6, 4, 5, 4, 6, 7, 6, 2, 3, 2, 0, 1, 0, 2, 3, 2, 6, 7, 6,
      4, 5, 4, 6, 7, 6, 5, 4, 5, 7, 6, 7, 5, 4, 5, 1, 0, 1, 3, 2, 3, 1, 0, 1,
      5, 4, 5, 7, 6, 7, 5, 4, 5, 4, 5, 4, 6, 7, 6, 4, 5, 4, 0, 1, 0, 2, 3, 2,
      0, 1, 0, 4, 5, 4, 6, 7, 6, 4, 5, 4, 3, 2, 3, 1, 0, 1, 3, 2, 3, 7, 6, 7,
      5, 4, 5, 7, 6, 7, 3, 2, 3, 1, 0, 1, 3, 2, 3, 2, 3, 2, 0, 1, 0, 2, 3, 2,
      6, 7, 6, 4, 5, 4, 6, 7, 6, 2, 3, 2, 0, 1, 0, 2, 3, 2, 1, 0, 1, 3, 2, 3,
      1, 0, 1, 5, 4, 5, 7, 6, 7, 5, 4, 5, 1, 0, 1, 3, 2, 3, 1, 0, 1, 0, 1, 0,
      2, 3, 2, 0, 1, 0, 4, 5, 4, 6, 7, 6, 4, 5, 4, 0, 1, 0, 2, 3, 2, 0, 1, 0};
};

/**
        node_vertices(t,i) =
   node_vertices(neighbors(t,NeighborsAtVertexLUT(i,7-i))) i = local index of
   vertex j = local index of node at vertex (arbitrarily ascending)
    NeighborsAtVertexLUT(i,j) = node neighbor at vertex
*/
struct NeighborsAtVertexLUT {
  static constexpr Int nrow_host = 8;
  static constexpr Int ncol_host = 8;
  const Int nrows = 8;
  const Int ncols = 8;
  const Int entries[64] = {0,  1,  3,  4,  9,  10, 12, 13, 1,  2,  4,  5,  10,
                           11, 13, 14, 3,  4,  6,  7,  12, 13, 15, 16, 4,  5,
                           7,  8,  13, 14, 16, 17, 9,  10, 12, 13, 18, 19, 21,
                           22, 10, 11, 13, 14, 19, 20, 22, 23, 12, 13, 15, 16,
                           21, 22, 24, 25, 13, 14, 16, 17, 22, 23, 25, 26};
};

/**
    For each local edge index i, NeighborsAtEdgeLUT(i,:) = the local indices of
   the 4 neighboring nodes that share the edge, arbitrarily ordered from lowest
   to highest
*/
struct NeighborsAtEdgeLUT {
  static constexpr Int nrow_host = 12;
  static constexpr Int ncol_host = 4;
  const Int nrows = 12;
  const Int ncols = 4;
  const Int entries[48] = {1,  4,  10, 13,   // 0
                           3,  4,  12, 13,   // 1
                           4,  7,  13, 16,   // 2
                           4,  5,  13, 14,   // 3
                           9,  10, 12, 13,   // 4
                           10, 11, 13, 14,   // 5
                           10, 13, 19, 22,   // 6
                           12, 13, 21, 22,   // 7
                           13, 14, 22, 23,   // 8
                           13, 16, 22, 25,   // 9
                           12, 13, 15, 16,   // 10
                           13, 14, 16, 17};  // 11
};

/**
    Node t, edge i, is shared by node_neighbors(t,NeighborsAtEdgeLUT(i,j)), for
   j in 0..3.

    let nbr = node_neighbors(t,NeighborsAtEdgeLUT(i,j))

    edge i is the NeighborEdgeComplementLUT(i,j) edge of nbr, so that:

    node_edges(t,i) = node_edges(node_neighbors(NeighborsAtEdgeLUT(i,j)),
   NeighborEdgeComplementLUT(i,j))
*/
struct NeighborEdgeComplementLUT {
  static constexpr Int nrow_host = 12;
  static constexpr Int ncol_host = 4;
  const Int nrows = 12;
  const Int ncols = 4;
  const Int entries[48] = {9,  6,  2, 0,   // 0
                           8,  7,  3, 1,   // 1
                           9,  6,  2, 0,   // 2
                           8,  7,  3, 1,   // 3
                           11, 10, 5, 4,   // 4
                           11, 10, 5, 4,   // 5
                           9,  6,  2, 0,   // 6
                           8,  7,  3, 1,   // 7
                           8,  7,  3, 1,   // 8
                           9,  6,  2, 0,   // 9
                           11, 10, 5, 4,   // 10
                           11, 10, 5, 4};  // 11
};

/**
    For local edge index i, EdgeVerticesLUT(i,:) = the local indices of the 2
   vertices that define the edge, so that, in the full octree,
        edge_vertices(node_edges(t,i),:) = node_vertices(t,
   EdgeVerticesLUT(i,:))
*/
struct EdgeVerticesLUT {
  static constexpr Int nrow_host = 12;
  static constexpr Int ncol_host = 2;
  const Int nrows = 12;
  const Int ncols = 2;
  const Int entries[24] = {1, 0,   // 0
                           0, 2,   // 1
                           2, 3,   // 2
                           3, 1,   // 3
                           0, 4,   // 4
                           1, 5,   // 5
                           5, 4,   // 6
                           4, 6,   // 7
                           5, 7,   // 8
                           7, 6,   // 9
                           6, 2,   // 10
                           7, 3};  // 11
};

/**
    node_faces(t,i) is shared by node_neighbors(t, NeighborsAtFaceLUT(i,j)) for
   j in 0..1
*/
struct NeighborsAtFaceLUT {
  static constexpr Int nrow_host = 6;
  static constexpr Int ncol_host = 2;
  const Int nrows = 6;
  const Int ncols = 2;
  const Int entries[12] = {4,  13,   // 0
                           10, 13,   // 1
                           13, 22,   // 2
                           13, 16,   // 3
                           12, 13,   // 4
                           13, 14};  // 5
};

/**
    let nbr = node_neighbors(t, NeighborsAtFaceLUT(i,j)) for j in 0..1

    face i is the NeighborFaceComplementLUT(i,j)th face of nbr

    node_faces(t,i) = node_faces(node_neighbors(t,NeighborsAtFaceLUT(i,j)),
   NeighborFaceComplementLUT(i,j))
*/
struct NeighborFaceComplementLUT {
  static constexpr Int nrow_host = 6;
  static constexpr Int ncol_host = 2;
  const Int nrows = 6;
  const Int ncols = 2;
  const Int entries[12] = {2, 0,   // 0
                           3, 1,   // 1
                           2, 0,   // 2
                           3, 1,   // 3
                           4, 5,   // 4
                           5, 4};  // 5
};

/**
    face_edges(node_faces(t,i),:) = node_edges(t, FaceEdgesLUT(i,:))
*/
struct FaceEdgesLUT {
  static constexpr Int nrow_host = 6;
  static constexpr Int ncol_host = 4;
  const Int nrows = 6;
  const Int ncols = 4;
  const Int entries[24] = {0, 1, 2,  3,    // 0
                           0, 4, 5,  6,    // 1
                           6, 7, 8,  9,    // 2
                           2, 9, 10, 11,   // 3
                           1, 4, 7,  10,   // 4
                           3, 5, 8,  11};  // 5
};

template <typename TableType>
KOKKOS_INLINE_FUNCTION Int table_val(const Int& i, const Int& j,
                                     const ko::View<TableType>& tableview) {
  assert(i >= 0 && i < tableview().nrows);
  assert(j >= 0 && j < tableview().ncols);
  return tableview().entries[tableview().ncols * i + j];
}

}  // namespace Octree
}  // namespace Lpm
#endif