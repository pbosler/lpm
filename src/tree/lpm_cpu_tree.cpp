#include "tree/lpm_cpu_tree.hpp"
#include "lpm_assert.hpp"
#include "lpm_coords_impl.hpp"
#include "util/lpm_string_util.hpp"

#ifdef LPM_USE_VTK
#include "vtkPolyData.h"
#include "vtkVoxel.h"
#include "vtkCellData.h"
#include "vtkXMLPolyDataWriter.h"
#endif

#include <numeric>
#include <limits>
#include <sstream>
#include <algorithm>

namespace Lpm {
namespace tree {

template <typename Geo, typename NodeType>
std::string CpuTree<Geo,NodeType>::info_string(const int tab_level) const {
  std::ostringstream ss;
  auto tabstr = indent_string(tab_level);
  ss << tabstr << "CpuTree info:\n";
  ss << node_info_string(root.get());
  return ss.str();
}

template <typename Geo, typename NodeType>
CpuTree<Geo,NodeType>::CpuTree(const std::shared_ptr<Coords<Geo>> crds,
  const TreeDepthControl& ctl, const Int ctl_n,
  const bool shrink) :
  _crds(crds),
  depth(0),
  n_nodes(1),
  control(ctl),
  control_n(ctl_n),
  do_shrink(shrink) {

  const auto xminmax = crds->min_max_extent(0);
  const auto yminmax = crds->min_max_extent(1);
  const auto zminmax = crds->min_max_extent(2);

  Box3d root_box(xminmax.min_val, xminmax.max_val,
                 yminmax.min_val, yminmax.max_val,
                 zminmax.min_val, zminmax.max_val);

  std::vector<Index> root_inds(crds->nh());
  std::iota(root_inds.begin(), root_inds.end(), 0);

  root = std::unique_ptr<NodeType>(new NodeType(root_box, NULL, root_inds));

#ifndef NDEBUG
  std::cout << root->info_string();
#endif

  LPM_ASSERT(root->is_leaf());
  LPM_ASSERT(!root->has_kids());
  LPM_ASSERT(root->n() == crds->nh());

  switch (ctl) {
    case (MaxCoordsPerLeaf) : {
      generate_tree_max_coords_per_node(root.get(), ctl_n);
      break;
    }
    case (MaxDepth) : {
      LPM_REQUIRE(ctl_n <= MAX_CPU_TREE_DEPTH);
      generate_tree_max_depth(root.get(), ctl_n);
      break;
    }
  }
}

template <typename Geo, typename NodeType>
void CpuTree<Geo,NodeType>::shrink_box(NodeType* node) {
  Real xmin = std::numeric_limits<Real>::max();
  Real xmax = std::numeric_limits<Real>::lowest();
  Real ymin = xmin;
  Real ymax = xmax;
  Real zmin = xmin;
  Real zmax = xmax;
  const auto crd_view = _crds->get_host_crd_view();
  for (Index i=0; i<node->n(); ++i) {
    const auto cxyz = Kokkos::subview(crd_view, i, Kokkos::ALL);
    if (cxyz[0] < xmin) xmin = cxyz[0];
    if (cxyz[1] < ymin) ymin = cxyz[1];
    if (cxyz[2] < zmin) zmin = cxyz[2];
    if (cxyz[0] > xmax) xmax = cxyz[0];
    if (cxyz[1] > ymax) ymax = cxyz[1];
    if (cxyz[2] > zmax) zmax = cxyz[2];
  }
  node->box.xmin = xmin;
  node->box.xmax = xmax;
  node->box.ymin = ymin;
  node->box.ymax = ymax;
  node->box.zmin = zmin;
  node->box.zmax = zmax;
}

template <typename Geo, typename NodeType>
void CpuTree<Geo,NodeType>::divide_node(NodeType* node) {
  LPM_ASSERT(!node->empty());
  const auto kid_boxes = node->box.bisect_all();
  auto counted = std::vector<bool>(node->n(), false);
  const auto crd_view = _crds->get_host_crd_view();
  int empty_count = 0;
  int full_count = 0;
  for (int k=0; k<8; ++k) {
    std::vector<Index> kid_inds;
    for (int i=0; i<node->n(); ++i) {
      const auto cxyz = Kokkos::subview(crd_view, node->crd_inds[i], Kokkos::ALL);
      if (kid_boxes[k].contains_pt(cxyz)) {
        if (!counted[i]) {
          kid_inds.push_back(node->crd_inds[i]);
          counted[i] = true;
        }
      }
    }
    if (kid_inds.empty()) {
      ++empty_count;
    }
    else {
      ++full_count;
      kid_inds.shrink_to_fit();
    }
    node->kids.push_back(std::unique_ptr<Node>(
      new Node(kid_boxes[k], node, kid_inds)));
  }
#ifndef NDEBUG
  if (full_count == 0) {
    std::cout << "node has empty kids.\n";
    std::cout << node->info_string();
  }
#endif
  LPM_ASSERT(std::count(counted.begin(), counted.end(), true) == node->n());
  LPM_ASSERT(node->has_kids());
  LPM_ASSERT(full_count + empty_count == 8);
  LPM_ASSERT(full_count > 0);
  ++depth;
  n_nodes += full_count;
  if (do_shrink) {
    for (int k=0; k<8; ++k) {
      if (!node->kids[k]->empty()) {
        shrink_box(node->kids[k].get());
      }
    }
  }
}

template <typename Geo, typename NodeType>
void CpuTree<Geo,NodeType>::generate_tree_max_coords_per_node(NodeType* node,
  const Index max_coords_per_node) {
  if (node->n() <= max_coords_per_node or node->empty()) {
    return;
  }
  else {
    divide_node(node);
    for (int k=0; k<8; ++k) {
      generate_tree_max_coords_per_node(node->kids[k].get(),
        max_coords_per_node);
    }
  }
}

template <typename Geo, typename NodeType>
void CpuTree<Geo,NodeType>::generate_tree_max_depth(NodeType* node, const int max_depth) {
  if (node->level == max_depth or node->empty()) {
    return;
  }
  else {
    divide_node(node);
    for (int k=0; k<8; ++k) {
      generate_tree_max_depth(node->kids[k].get(), max_depth);
    }
  }
}

#ifdef LPM_USE_VTK
template <typename Geo>
void CpuTree<Geo>::insert_vtk_cell_points(
  vtkSmartPointer<vtkPoints> pts, const Index pt_offset,
  vtkSmartPointer<vtkCellArray> cells, const Index cell_offset,
  vtkSmartPointer<vtkIntArray> levels, const Node* node) {
  const auto box = node->box;
  const auto npts_in = pts->GetNumberOfPoints();
  const auto ncells_in = ugrid->GetNumberOfCells();

  pts->InsertNextPoint(box.xmin, box.ymin, box.zmin);
  pts->InsertNextPoint(box.xmax, box.ymin, box.zmin);
  pts->InsertNextPoint(box.xmax, box.ymax, box.zmin);
  pts->InsertNextPoint(box.xmin, box.ymax, box.zmin);

  pts->InsertNextPoint(box.xmin, box.ymin, box.zmax);
  pts->InsertNextPoint(box.xmax, box.ymin, box.zmax);
  pts->InsertNextPoint(box.xmax, box.ymax, box.zmax);
  pts->InsertNextPoint(box.xmin, box.ymax, box.zmax);

  vtkNew<vtkHexahedron> hex;
  for (auto i=0; i<8; ++i) {
    hex->GetPointIds()->SetId(i, npts_in+i);
  }
  levels->InsertTuple1(ncells_in, node->level);
  ugrid->InsertNextCell(hex->GetCellType(), hex->GetPointIds());

  if (node->has_kids()) {
    for (int k=0; k<8; ++k) {
      if (!node->kids[k]->empty()) {
        insert_vtk_cell_points(pts, ugrid, levels,
          node->kids[k].get());
      }
    }
  }
}

template <typename Geo, typename NodeType>
void CpuTree<Geo,NodeType>::write_vtk(const std::string& ofname) const {
  LPM_REQUIRE(filename_extension(ofname) == ".vtu");
  vtkNew<vtkPoints> pts;
  vtkNew<vtkUnstructuredGrid> ugrid;
  vtkNew<vtkIntArray> levels;
  levels->SetName("tree_level");
  levels->SetNumberOfComponents(1);
  levels->SetNumberOfTuples(n_nodes);
  insert_vtk_cell_points(pts, ugrid, levels, root.get());
  ugrid->SetPoints(pts);
  ugrid->GetCellData()->AddArray(levels);
  vtkNew<vtkXMLUnstructuredGridWriter> writer;
  writer->SetInputData(ugrid);
  writer->SetFileName(ofname.c_str());
  writer->Write();
}
#endif

} // namespace tree

template <typename Geo, typename NodeType>
std::string CpuTree<Geo,NodeType>::node_info_string(NodeType* node) const {
  auto nodestr = node->info_string(node->level+1);
  if (node->has_kids()) {
    for (int k=0; k<8; ++k) {
      if (!node->kids[k]->empty()) {
        nodestr += node_info_string(node->kids[k].get());
      }
    }
  }
  return nodestr;
}

// ETI
template class CpuTree<>;

}
} // namespace Lpm
