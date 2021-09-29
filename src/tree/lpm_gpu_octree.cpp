#include "tree/lpm_gpu_octree.hpp"
#include "tree/lpm_node_array_d.hpp"
#include "tree/lpm_node_array_internal.hpp"
#include "tree/lpm_node_array_internal_impl.hpp"
#include "tree/lpm_gpu_octree_kernels.hpp"
#include "util/lpm_string_util.hpp"
#include "vtkPoints.h"
#include "vtkIntArray.h"
#include "vtkUnstructuredGrid.h"
#include "vtkCellData.h"
#include "vtkNew.h"
#include "vtkSmartPointer.h"
#include "vtkHexahedron.h"
#include "vtkXMLUnstructuredGridWriter.h"
#include <map>

namespace Lpm {
namespace tree {

void GpuOctree::init() {
  /**

    construct each level of the tree in parallel, starting with the leaves

  */
  NodeArrayD leaves(unsorted_pts, max_depth);

  auto h_bb = Kokkos::create_mirror_view(bounding_box);
  h_bb() = std_box();
  Kokkos::deep_copy(bounding_box, h_bb);
  sorted_pts = leaves.sorted_pts;
  pt_in_leaf = leaves.pt_in_node;
  pt_idx_orig = leaves.pt_idx_orig;

  std::map<int, NodeArrayInternal> levels;
  // level max_depth-1 is constructed from the leaves
  levels.emplace(max_depth-1, NodeArrayInternal(leaves));
  // other levels are constructed from internal level arrays
  for (int l=max_depth-2; l>=0; --l) {
    levels.emplace(l, NodeArrayInternal(levels.at(l+1)));
  }

  /**

    set up memory allocations for full tree

  */
  auto h_base_address = Kokkos::create_mirror_view(base_address);
  auto h_nnodes_per_level = Kokkos::create_mirror_view(nnodes_per_level);
  nnodes = leaves.nnodes;
  h_nnodes_per_level(max_depth) = leaves.nnodes;
  for (const auto& na : levels) {
    nnodes += na.second.nnodes;
    h_nnodes_per_level(na.first) = na.second.nnodes;
  }
  h_base_address(0) = 0;
  for (auto l=1; l<=max_depth; ++l) {
    h_base_address(l) = h_base_address(l-1) + h_nnodes_per_level(l-1);
  }
  LPM_ASSERT(h_nnodes_per_level(0) == 1);
  LPM_ASSERT(h_base_address(1) == 1);
  Kokkos::deep_copy(base_address, h_base_address);
  Kokkos::deep_copy(nnodes_per_level, h_nnodes_per_level);

  node_keys = Kokkos::View<key_type*>("node_keys", nnodes);
  node_pt_idx_start = Kokkos::View<Index*>("node_pt_idx_start", nnodes);
  node_pt_idx_count = Kokkos::View<id_type*>("node_pt_idx_count", nnodes);
  node_parents = Kokkos::View<id_type*>("node_parents", nnodes);
  node_kids = Kokkos::View<id_type*[8]>("node_kids", nnodes);
  node_neighbors = Kokkos::View<Index*[27]>("node_neighbors", nnodes);

  /**

    concatenate node arrays

  */
  for (auto l=0; l<max_depth; ++l) {
    const auto dest_range = std::make_pair(h_base_address(l),
      h_base_address(l) + h_nnodes_per_level(l));
    Kokkos::deep_copy(Kokkos::subview(node_keys, dest_range),
      levels.at(l).node_keys);
    Kokkos::deep_copy(Kokkos::subview(node_pt_idx_start, dest_range),
      levels.at(l).node_idx_start);
    Kokkos::deep_copy(Kokkos::subview(node_pt_idx_count, dest_range),
      levels.at(l).node_idx_count);
    Kokkos::deep_copy(Kokkos::subview(node_parents, dest_range),
      levels.at(l).node_parents);
    Kokkos::deep_copy(Kokkos::subview(node_kids, dest_range, Kokkos::ALL),
      levels.at(l).node_kids);
  }
  const auto dest_range = std::make_pair(h_base_address(max_depth),
    h_base_address(max_depth) + h_nnodes_per_level(max_depth));
  Kokkos::deep_copy(Kokkos::subview(node_keys, dest_range),
    leaves.node_keys);
  Kokkos::deep_copy(Kokkos::subview(node_pt_idx_start, dest_range),
    leaves.node_idx_start);
  Kokkos::deep_copy(Kokkos::subview(node_pt_idx_count, dest_range),
    leaves.node_idx_count);
  Kokkos::deep_copy(Kokkos::subview(node_parents, dest_range),
    leaves.node_parents);
  Kokkos::deep_copy(Kokkos::subview(node_kids, dest_range, Kokkos::ALL),
    NULL_IDX);

  /**

    compute node neighbor lists

    Listing 2 from
    [Z11] K. Zhou, et. al., 2011. Data-parallel octrees for surface reconstruction,
    IEEE Trans. Vis. Comput. Graphics 17(5): 669--681. DOI: 10.1109/TVCG.2010.75

  */
  auto root_neighbors = Kokkos::subview(node_neighbors, 0, Kokkos::ALL);
  auto h_root_neighbors = Kokkos::create_mirror_view(root_neighbors);
  for (auto j=0; j<27; ++j) {
    h_root_neighbors(j) = (j != 13 ? NULL_IDX : 0);
  }
  Kokkos::deep_copy(root_neighbors, h_root_neighbors);
  for (auto l=1; l<=max_depth; ++l) {
#ifdef LPM_USE_CUDA
    Kokkos::TeamPolicy<> neighbor_policy(h_nnodes_per_level(l), 32);
    Kokkos::parallel_for(neighbor_policy,
      NeighborhoodFunctor(node_neighbors, node_keys, node_parents, node_kids,
        l, max_depth, h_base_address(l)));
#else
    Kokkos::parallel_for(h_nnodes_per_level(l),
      NeighborhoodFunctor(node_neighbors, node_keys, node_parents, node_kids,
        l, max_depth, h_base_address(l)));
#endif
  }

  /**

    construct node vertices

  */
//   Kokkos::View<id_type*[8]> vertex_owner("vertex_owner", nnodes);
//   Kokkos::View<id_type*> vert_address("vert_address", nnodes);
//   for (auto l=0; l<=max_depth; ++l) {
//     const auto node_range = std::make_pair(h_base_address(l),
//       h_base_address(l) + h_nnodes_per_level(l));
//     Kokkos::TeamPolicy<> vertex_policy(h_nnodes_per_level(l), Kokkos::AUTO());
//     Kokkos::parallel_for(vertex_policy,
//       VertexOwnerFunctor(Kokkos::subview(vertex_owner, node_range, Kokkos::ALL),
//         Kokkos::subview(node_keys, node_range),
//         Kokkos::subview(node_neighbors, node_range, Kokkos::ALL),
//         h_base_address(l)));
//   }
//
//   for (auto n=0; n<nnodes; ++n) {
//     for (auto v=0; v<8; ++v) {
//       std::cout << "owner(" << n << " , " << v << ") = " << vertex_owner(n,v) << "\n";
//     }
//   }


}

void GpuOctree::write_vtk(const std::string& ofname) const {
  LPM_REQUIRE(filename_extension(ofname) == ".vtu");
  auto h_keys = Kokkos::create_mirror_view(node_keys);
  auto h_pt_ct = Kokkos::create_mirror_view(node_pt_idx_count);
  auto h_base_address = Kokkos::create_mirror_view(base_address);
  auto h_nnodes_per_level = Kokkos::create_mirror_view(nnodes_per_level);
  auto h_vert_crds = Kokkos::create_mirror_view(node_vertex_crds);
  auto h_node_verts = Kokkos::create_mirror_view(node_vertices);
  auto h_neighbors = Kokkos::create_mirror_view(node_neighbors);
  Kokkos::deep_copy(h_keys, node_keys);
  Kokkos::deep_copy(h_pt_ct, node_pt_idx_count);
  Kokkos::deep_copy(h_base_address, base_address);
  Kokkos::deep_copy(h_nnodes_per_level, nnodes_per_level);
  Kokkos::deep_copy(h_vert_crds, node_vertex_crds);
  Kokkos::deep_copy(h_node_verts, node_vertices);
  Kokkos::deep_copy(h_neighbors, node_neighbors);

  vtkNew<vtkPoints> pts;
  vtkNew<vtkUnstructuredGrid> ugrid;
  vtkNew<vtkIntArray> keys;
  vtkNew<vtkIntArray> npts;
  vtkNew<vtkIntArray> lev;
  vtkNew<vtkIntArray> nbrs;
  keys->SetName("node_keys");
  keys->SetNumberOfComponents(1);
  keys->SetNumberOfTuples(node_keys.extent(0));
  npts->SetName("pt_idx_count");
  npts->SetNumberOfComponents(1);
  npts->SetNumberOfTuples(node_keys.extent(0));
  lev->SetName("level");
  lev->SetNumberOfComponents(1);
  lev->SetNumberOfTuples(node_keys.extent(0));
  nbrs->SetName("neighbors");
  nbrs->SetNumberOfComponents(1);
  nbrs->SetNumberOfTuples(node_keys.extent(0));

  const id_type nbr_example = 2400;
  for (auto n=0; n<nnodes; ++n) {
    nbrs->InsertTuple1(n, -1);
  }
  for (auto j=0; j<27; ++j) {
    nbrs->SetValue(h_neighbors(nbr_example, j), 100);
  }

//   for (auto p=0; p<node_vertex_crds.extent(0); ++p) {
//     const auto vxyz = Kokkos::subview(h_vert_crds, p, Kokkos::ALL);
//     pts->InsertNextPoint(vxyz[0], vxyz[1], vxyz[2]);
//   }
//   for (auto n=0; n<nnodes; ++n) {
//     vtkNew<vtkHexahedron> hex;
//     hex->GetPointIds()->SetId(0, h_node_verts(n,0));
//     hex->GetPointIds()->SetId(1, h_node_verts(n,4));
//     hex->GetPointIds()->SetId(2, h_node_verts(n,6));
//     hex->GetPointIds()->SetId(3, h_node_verts(n,2));
//     hex->GetPointIds()->SetId(4, h_node_verts(n,1));
//     hex->GetPointIds()->SetId(5, h_node_verts(n,5));
//     hex->GetPointIds()->SetId(6, h_node_verts(n,7));
//     hex->GetPointIds()->SetId(7, h_node_verts(n,3));
//
//     ugrid->InsertNextCell(hex->GetCellType(), hex->GetPointIds());
//   }
  for (auto l=0; l<=max_depth; ++l) {
    for (auto i=0; i<h_nnodes_per_level(l); ++i) {
      const auto node_idx = h_base_address(l) + i;
      const auto box = box_from_key(h_keys(node_idx), l, max_depth);
      pts->InsertNextPoint(box.xmin, box.ymin, box.zmin);
      pts->InsertNextPoint(box.xmax, box.ymin, box.zmin);
      pts->InsertNextPoint(box.xmax, box.ymax, box.zmin);
      pts->InsertNextPoint(box.xmin, box.ymax, box.zmin);

      pts->InsertNextPoint(box.xmin, box.ymin, box.zmax);
      pts->InsertNextPoint(box.xmax, box.ymin, box.zmax);
      pts->InsertNextPoint(box.xmax, box.ymax, box.zmax);
      pts->InsertNextPoint(box.xmin, box.ymax, box.zmax);

      vtkNew<vtkHexahedron> hex;
      for (auto j=0; j<8; ++j) {
        hex->GetPointIds()->SetId(j, 8*node_idx + j);
      }
      ugrid->InsertNextCell(hex->GetCellType(), hex->GetPointIds());
      keys->InsertTuple1(node_idx, h_keys(node_idx));
      npts->InsertTuple1(node_idx, h_pt_ct(node_idx));
      lev->InsertTuple1(node_idx, l);
    }
  }
  ugrid->SetPoints(pts);
  ugrid->GetCellData()->AddArray(keys);
  ugrid->GetCellData()->AddArray(npts);
  ugrid->GetCellData()->AddArray(lev);
  ugrid->GetCellData()->AddArray(nbrs);

  vtkNew<vtkXMLUnstructuredGridWriter> writer;
  writer->SetInputData(ugrid);
  writer->SetFileName(ofname.c_str());
  writer->Write();
}

std::string GpuOctree::info_string(const int tab_level) const {
  std::ostringstream ss;
  auto tabstr = indent_string(tab_level);
  ss << tabstr << "GPUOctree info:\n";
  tabstr += "\t";
  ss << tabstr << "nnodes = " << nnodes << "\n";
  ss << tabstr << "max_depth = " << max_depth << "\n";
  ss << tabstr << "nvertices = " << node_vertex_crds.extent(0) << "\n";
  return ss.str();
}

} // namespace tree
} // namespace Lpm
