#ifndef LPM_GPU_OCTREE_LOOKUP_TABLES_HPP
#define LPM_GPU_OCTREE_LOOKUP_TABLES_HPP

#include "LpmConfig.h"
#include "lpm_assert.hpp"

namespace Lpm {
namespace tree {

/**
    These lookup tables encode the neighbor/child relationships defined by the following orderings:

       z
       |
       |
       |
       o------y
      /
     /
    x


    Parent:        Children/Vertices:           Edges:                  Faces:

        .---------.        .-------.                .-----3----.            .-----------.           .---------.
       /         /|       /|1 / 3 /|               / |         /|          /|          /|          /|         |
      /         / |      ---------                5  0       11 |         /    /5/    / |         / |         |
     /         /  |     /| 5 / 7/|               /   |       /  |        /           /  |       ./  |  |0|    |
    .---------.   |    .-------.                .------8----.   2       .-----------.   |       |   |         |
    |         |   |    |       |                |    |___1__|___|       |           ||3||       ||1||         |
    |         |   /                             |   /       |   /       |           |   /       |   |_________|
    |         |  /                              6  4        9  10       |    |2|    |  /        |  /          /
    |         | /                               | /         | /         |           | /         | /   /4/    /
    |_________|/            |-------|           ./_____7____./          |/__________|/          |/_________|/
                           / 0 / 2 /
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

    if node t's index in node_kids(p,:) is i, i.e., node t is the ith child of p, or,

       i = local_key(node_keys(t), l, max_depth),

    then the index of node t's jth neighbor's parent in node_neighbors(p,:) is table(i,j),

        node_parent(node_neighbors(t,j)) = node_neighbors(p, table(i,j))

    ...

    i = local_key of node t, relative to its parent
    j = neighbor of node t, in neighbor ordering [0,27]
    *** I am the ith child of my parent;
      the parent of my jth neighbor is the table(i,j)th neighbor of my parent ***

*/
struct ParentLUT {
    const Int nrows = 8;
    const Int ncols = 27;
    const Int entries[216] = {0,1,1,3,4,4,3,4,4,9,10,10,12,13,13,12,13,13,9,10,10,12,13,13,12,13,13,
                            1,1,2,4,4,5,4,4,5,10,10,11,13,13,14,13,13,14,10,10,11,13,13,14,13,13,14,
                            3,4,4,3,4,4,6,7,7,12,13,13,12,13,13,15,16,16,12,13,13,12,13,13,15,16,16,
                            4,4,5,4,4,5,7,7,8,13,13,14,13,13,14,16,16,17,13,13,14,13,13,14,16,16,17,
                            9,10,10,12,13,13,12,13,13,9,10,10,12,13,13,12,13,13,18,19,19,21,22,22,21,22,22,
                            10,10,11,13,13,14,13,13,14,10,10,11,13,13,14,13,13,14,19,19,20,22,22,23,22,22,23,
                            12,13,13,12,13,13,15,16,16,12,13,13,12,13,13,15,16,16,21,22,22,21,22,22,24,25,25,
                            13,13,14,13,13,14,16,16,17,13,13,14,13,13,14,16,16,17,22,22,23,22,22,23,25,25,26};
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
    *** I am the ith child of my parent;
      my jth neighbor is the table(i,j)th child of its parent ***

      -- and --

    *** My ith vertex is the table(i,j)th vertex of my jth neighbor

*/
struct ChildLUT {
    const Int nrows = 8;
    const Int ncols = 27;
    const Int entries[216] = {7,6,7,5,4,5,7,6,7,3,2,3,1,0,1,3,2,3,3,2,3,5,4,5,1,0,1,
                             6,7,6,4,5,4,6,7,6,2,3,2,0,1,0,2,3,2,6,7,6,4,5,4,6,7,6,
                             5,4,5,7,6,7,5,4,5,1,0,1,3,2,3,1,0,1,5,4,5,7,6,7,5,4,5,
                             4,5,4,6,7,6,4,5,4,0,1,0,2,3,2,0,1,0,4,5,4,6,7,6,4,5,4,
                             3,2,3,1,0,1,3,2,3,7,6,7,5,4,5,7,6,7,3,2,3,1,0,1,3,2,3,
                             2,3,2,0,1,0,2,3,2,6,7,6,4,5,4,6,7,6,2,3,2,0,1,0,2,3,2,
                             1,0,1,3,2,3,1,0,1,5,4,5,7,6,7,5,4,5,1,0,1,3,2,3,1,0,1,
                             0,1,0,2,3,2,0,1,0,4,5,4,6,7,6,4,5,4,0,1,0,2,3,2,0,1,0};
};

/** @brief Given a node n with (local) vertex i, the table gives the neighbor
    indices of nodes that share the vertex.


    i = local index of vertex
    j = local index of node at vertex (arbitrarily ascending)
    NeighborsAtVertexLUT(i,j) = node neighbor at vertex

  ** my table(i,j)th neighbor shares my ith vertex, for j in [0,7] **

*/
struct NeighborsAtVertexLUT {
    const Int nrows = 8;
    const Int ncols = 8;
    const Int entries[64] = {0,1,3,4,9,10,12,13,
                            1,2,4,5,10,11,13,14,
                            3,4,6,7,12,13,15,16,
                            4,5,7,8,13,14,16,17,
                            9,10,12,13,18,19,21,22,
                            10,11,13,14,19,20,22,23,
                            12,13,15,16,21,22,24,25,
                            13,14,16,17,22,23,25,26};
};

template <typename TableType> KOKKOS_INLINE_FUNCTION
Int table_val(const Int& i, const Int& j, const ko::View<TableType>& tableview) {
    LPM_KERNEL_ASSERT(i >=0 && i < tableview().nrows);
    LPM_KERNEL_ASSERT(j >=0 && j < tableview().ncols);
    return tableview().entries[tableview().ncols*i+j];
}

} // namespace tree
} // namespace Lpm

#endif
