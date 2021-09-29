#include "lpm_geometry.hpp"
#include "tree/lpm_node_array_d.hpp"
#include "tree/lpm_gpu_octree_functions.hpp"
#include "tree/lpm_gpu_octree_kernels.hpp"
#include "lpm_assert.hpp"
#include "util/lpm_floating_point.hpp"
#include "util/lpm_string_util.hpp"
#include "vtkPoints.h"
#include "vtkIntArray.h"
#include "vtkUnstructuredGrid.h"
#include "vtkCellData.h"
#include "vtkNew.h"
#include "vtkSmartPointer.h"
#include "vtkHexahedron.h"
#include "vtkXMLUnstructuredGridWriter.h"
#include "Kokkos_Sort.hpp"

namespace Lpm {
namespace tree {

void NodeArrayD::init(const Kokkos::View<Real*[3]> unsorted_pts) {
  // step 1: compute bounding box
  // TODO: allow for boxes that are not the standard [-1,1]^3
  const Index npts = unsorted_pts.extent(0);
  auto hbox = Kokkos::create_mirror_view(bounding_box);
  hbox() = std_box();
  Kokkos::deep_copy(bounding_box, hbox);

  // step 2: compute shuffled xyz key and sorting code
  Kokkos::View<code_type*> sort_codes("sort_codes", npts);
  Kokkos::parallel_for("NodeArrayD::generate sort codes", npts,
    EncodeFunctor(sort_codes, unsorted_pts, level));

  // step 3: sort points
  Kokkos::sort(sort_codes);
  Kokkos::parallel_for("NodeArrayD::sort points", npts,
    SortPointsFunctor(sorted_pts, pt_idx_orig, unsorted_pts, sort_codes));

#ifndef NDEBUG
  Kokkos::View<Real*[3]> check_pts("check_pts", npts);
  Kokkos::parallel_for("unsort", npts,
    UnsortPointsFunctor(check_pts, sorted_pts, pt_idx_orig));
  Kokkos::parallel_for("check sort/unsort", npts,
    KOKKOS_LAMBDA (const Index i) {
      const auto pti_in = Kokkos::subview(unsorted_pts, i, Kokkos::ALL);
      const auto pti_ck = Kokkos::subview(check_pts, i, Kokkos::ALL);
      LPM_KERNEL_ASSERT(FloatingPoint<Real>::zero(
        SphereGeometry::square_euclidean_distance(pti_in, pti_ck)));
  });
#endif

  // step 4: find unique nodes
  //  aka: parallel stream compaction
  //  step 4a: mark and count unique keys
  Kokkos::View<id_type*> unique_flags("unique_flags", npts);
  Index unique_count = 0;
  Kokkos::parallel_reduce("mark & count unique nodes", npts,
    MarkUniqueFunctor(unique_flags, sort_codes), unique_count);
  // step 4b: scan (prefix sum)
  /**
     after scan, unique_flags(i) = the number of unique nodes in [0, i].
  */
  Kokkos::parallel_scan("scan unique flags", npts,
    KOKKOS_LAMBDA (const Index i, id_type& update, const bool is_final) {
      const auto flag_i = unique_flags(i);
      update += flag_i; // inclusive sum
      if (is_final) {
        unique_flags(i) = update;
      }
    });
  // step 4c: create unique nodes
  Kokkos::View<key_type*> unique_keys("unique_keys", unique_count);
  Kokkos::View<Index*>  unode_pt_idx_start("node_pt_idx_start", unique_count);
  Kokkos::View<id_type*>  unode_pt_idx_count("node_pt_idx_count", unique_count);
  Kokkos::parallel_for("setup unique nodes", npts,
    UniqueNodeFunctor(unique_keys, unode_pt_idx_start, unode_pt_idx_count,
      unique_flags, sort_codes));

#ifndef NDEBUG
  Kokkos::parallel_for("check unique", unique_count,
    KOKKOS_LAMBDA (const Index i) {
      const auto tgt_key = unique_keys(i);
      const auto first = binary_search_first(tgt_key, unique_keys);
      const auto last = binary_search_last(tgt_key, unique_keys);
      LPM_KERNEL_ASSERT(i == first);
      LPM_KERNEL_ASSERT(i == last);
    });
#endif

  // step 5: for each unique node, make sure its siblings are also included
  // (even if they're empty)
  //    step 5a: mark each unique parent with an 8 (its number of kids)
  Kokkos::View<id_type*> node_address("node_address", unique_count);
  Kokkos::parallel_for("mark unique parents", unique_count,
    MarkUniqueParentFunctor(node_address, unique_keys, level, level));
  //    step 5b: scan
  /**
    after scan, node_address is the address of each unique node in the final
      leaf array
    and node_address.last() is the total number of leaf nodes
  */
  Kokkos::parallel_scan("scan unique parents", unique_count,
    KOKKOS_LAMBDA (const Index i, key_type& update, const bool is_final) {
      const auto add_i = node_address(i);
      update += add_i; // inclusive sum
      if (is_final) {
        node_address(i) = update;
      }
    });
  const auto nnview = Kokkos::subview(node_address, unique_count-1);
  auto h_nnview = Kokkos::create_mirror_view(nnview);
  Kokkos::deep_copy(h_nnview, nnview);
  nnodes = h_nnview();

  // step 6: create NodeArrayD
  node_keys = Kokkos::View<key_type*>("node_keys", nnodes);
  node_idx_start = Kokkos::View<Index*>("node_idx_start", nnodes);
  node_idx_count = Kokkos::View<id_type*>("node_idx_count", nnodes);
  node_parents = Kokkos::View<id_type*>("node_parents", nnodes);
  Kokkos::TeamPolicy<> leaf_policy(unique_count, Kokkos::AUTO());
  Kokkos::parallel_for("create node array d", leaf_policy,
    NodeArrayDFunctor(node_keys, node_idx_start, node_idx_count, node_parents,
      pt_in_node, node_address, unique_keys,
      unode_pt_idx_start, unode_pt_idx_count, level));

}

std::string NodeArrayD::info_string(const int tab_level) const {
  std::ostringstream ss;
  auto tabstr = indent_string(tab_level);
  ss << tabstr << "NodeArrayD info:\n";
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
NodeArrayD::min_max_pts_per_node() const {
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

void NodeArrayD::write_vtk(const std::string& ofname) const {
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
    const auto box = box_from_key(h_keys(i), level, level);
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

} // namespace tree
} // namespace Lpm
