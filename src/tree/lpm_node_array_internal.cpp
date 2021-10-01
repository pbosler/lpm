#include "tree/lpm_node_array_internal.hpp"
#include "util/lpm_string_util.hpp"
#include "vtkPoints.h"
#include "vtkIntArray.h"
#include "vtkUnstructuredGrid.h"
#include "vtkCellData.h"
#include "vtkNew.h"
#include "vtkSmartPointer.h"
#include "vtkHexahedron.h"
#include "vtkXMLUnstructuredGridWriter.h"

namespace Lpm {
namespace octree {

void NodeArrayInternal::build_root(Kokkos::View<id_type*> parents_lower,
  const Kokkos::View<id_type*> idx_count_lower) {
  nnodes = 1;
  node_keys = Kokkos::View<key_type*>("node_keys",1);
  Kokkos::deep_copy(node_keys,0);
  node_parents = Kokkos::View<id_type*>("node_parents",1);
  Kokkos::deep_copy(node_parents, NULL_IDX);
  node_kids = Kokkos::View<id_type*[8]>("node_kids", 1);
  node_idx_start = Kokkos::View<Index*>("node_idx_start",1);
  Kokkos::deep_copy(node_idx_start,0);
  node_idx_count = Kokkos::View<id_type*>("node_idx_count",1);

  auto lkids = node_kids;
  Kokkos::parallel_reduce(8, KOKKOS_LAMBDA (const int k, id_type& ct) {
    parents_lower(k) = 0;
    lkids(0,k) = k;
    ct += idx_count_lower(k);
  }, node_idx_count(0));
}

std::string NodeArrayInternal::info_string(const int tab_level) const {
  std::ostringstream ss;
  auto tabstr = indent_string(tab_level);
  ss << tabstr << "NodeArrayInternal info:\n";
  tabstr += "\t";
  ss << tabstr << "nnodes = " << nnodes << "\n";
  ss << tabstr << "level = " << level << "\n";
  ss << tabstr << "max_depth = " << max_depth << "\n";
  const auto mm = min_max_pts_per_node();
  ss << tabstr << "min_max_pts_per_node() = (" << mm.min_val << ", "
     << mm.max_val << ")\n";
  return ss.str();
}

typename Kokkos::MinMax<id_type>::value_type
NodeArrayInternal::min_max_pts_per_node() const {
  typedef typename Kokkos::MinMax<id_type>::value_type result_type;
  const auto pt_cts = node_idx_count;
  result_type result;
  Kokkos::parallel_reduce(node_keys.extent(0),
    KOKKOS_LAMBDA (const id_type i, result_type& mm) {
      if (pt_cts(i) < mm.min_val) mm.min_val = pt_cts(i);
      if (pt_cts(i) > mm.max_val) mm.max_val = pt_cts(i);
  }, Kokkos::MinMax<id_type>(result));
  return result;
}

void NodeArrayInternal::write_vtk(const std::string& ofname) const {
  LPM_REQUIRE(filename_extension(ofname) == ".vtu");
  auto h_keys = Kokkos::create_mirror_view(node_keys);
  auto h_pt_ct = Kokkos::create_mirror_view(node_idx_count);
  Kokkos::deep_copy(h_keys, node_keys);
  Kokkos::deep_copy(h_pt_ct, node_idx_count);

  vtkNew<vtkPoints> pts;
  vtkNew<vtkUnstructuredGrid> ugrid;
  vtkNew<vtkIntArray> keys;
  vtkNew<vtkIntArray> npts;
  vtkNew<vtkIntArray> lev;
  keys->SetName("node_keys");
  keys->SetNumberOfComponents(1);
  keys->SetNumberOfTuples(node_keys.extent(0));
  npts->SetName("pt_idx_count");
  npts->SetNumberOfComponents(1);
  npts->SetNumberOfTuples(node_keys.extent(0));
  lev->SetName("level");
  lev->SetNumberOfComponents(1);
  lev->SetNumberOfTuples(node_keys.extent(0));

  for (auto i=0; i<node_keys.extent(0); ++i) {
    const auto box = box_from_key(h_keys(i), level, max_depth);
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
      hex->GetPointIds()->SetId(j, 8*i + j);
    }
    ugrid->InsertNextCell(hex->GetCellType(), hex->GetPointIds());
    keys->InsertTuple1(i, h_keys(i));
    npts->InsertTuple1(i, h_pt_ct(i));
    lev->InsertTuple1(i, level);
  }

  ugrid->SetPoints(pts);
  ugrid->GetCellData()->AddArray(keys);
  ugrid->GetCellData()->AddArray(npts);
  ugrid->GetCellData()->AddArray(lev);

  vtkNew<vtkXMLUnstructuredGridWriter> writer;
  writer->SetInputData(ugrid);
  writer->SetFileName(ofname.c_str());
  writer->Write();
}

}
}
