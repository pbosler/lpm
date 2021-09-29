#ifndef LPM_GPU_OCTREE_HPP
#define LPM_GPU_OCTREE_HPP

#include "LpmConfig.h"
#include "tree/lpm_gpu_octree_functions.hpp"
#include "tree/lpm_box3d.hpp"

namespace Lpm {
namespace tree {

struct GpuOctree {

  // point arrays
  Kokkos::View<id_type*> pt_in_leaf;
  Kokkos::View<Index*> pt_idx_orig;
  Kokkos::View<Real*[3]> unsorted_pts;
  Kokkos::View<Real*[3]> sorted_pts;

  Kokkos::View<Real*[3]> node_vertex_crds;

  // node arrays
  Kokkos::View<key_type*> node_keys;
  Kokkos::View<Index*> node_pt_idx_start;
  Kokkos::View<id_type*> node_pt_idx_count;
  Kokkos::View<id_type*> node_parents;
  Kokkos::View<id_type*[8]> node_kids;
  Kokkos::View<Index*[27]> node_neighbors;

  // node connectivity arrays
  Kokkos::View<Box3d*> node_boxes;

  Kokkos::View<id_type*[8]> node_vertices;
  Kokkos::View<id_type*[12]> node_edges;
  Kokkos::View<id_type*[6]> node_faces;

  Kokkos::View<id_type*[8]> vertex_nodes;
  Kokkos::View<id_type*[2]> edge_vertices;
  Kokkos::View<id_type*[4]> face_edges;

  // tree arrays
  Kokkos::View<id_type*> base_address;
  Kokkos::View<id_type*> nnodes_per_level;
  // tree scalars
  Index nnodes;
  Int max_depth;
  Kokkos::View<Box3d> bounding_box;
  bool do_connectivity;

  GpuOctree(const Kokkos::View<Real*[3]> pts,
            const Int md,
            const bool connect = false) :
    unsorted_pts(pts),
    base_address("base_address", md+1),
    nnodes_per_level("nnodes_per_level", md+1),
    nnodes(0),
    max_depth(md),
    bounding_box("bounding_box"),
    do_connectivity(connect) { init(); }


  void write_vtk(const std::string& ofname) const;

  std::string info_string(const int tab_level=0) const;

  protected:
    void init();
};


}
} // namespace Lpm

#endif
