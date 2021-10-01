#include "tree/lpm_gpu_octree_lookup_tables.hpp"
#include "tree/lpm_box3d.hpp"

namespace Lpm {
namespace tree {

std::array<Int,216> parent_lookup_table_entries() {
  std::array<Int,216> result;
  const auto pbox = Box3d(-1,1,-1,1,-1,1,false);
  const auto parent_nbrs = pbox.neighbors();
  const auto kids = pbox.bisect_all();
  for (auto i=0; i<8; ++i) {

    LPM_ASSERT(pbox.contains_pt(kids[i].centroid()));

    const auto kid_nbrs = kids[i].neighbors();

    for (auto j=0; j<27; ++j) {
      const auto knbr_cntrd = kid_nbrs[j].centroid();
      auto ct = 0;
      for (auto k=0; k<27; ++k) {
        if (parent_nbrs[k].contains_pt(knbr_cntrd)) {
          ++ct;
          result[27*i+j] = k;
        }
      }
      LPM_ASSERT(ct == 1);
    }
  }
  return result;
}

std::array<Int,216> child_lookup_table_entries() {
  std::array<Int,216> result;

  const auto p = Box3d(-1,1,-1,1,-1,1,false);
  const auto p_nbrs = p.neighbors();
  const auto p_kids = p.bisect_all();

  for (auto i=0; i<8; ++i) {
    // node t with parent = p and index i
    const auto t = p_kids[i];
    const auto t_nbrs = t.neighbors();

    // loop over t_nbrs
    for (auto j=0; j<27; ++j) {
      // get a point contained by t_nbrs[j]
      auto c = t_nbrs[j].centroid();

      // that point will be contained by exactly 1 of the
      // neighbors of parent p.  That parent = h.
      auto npct = 0;
      int h;
      for (auto k=0; k<27; ++k) {
        if (p_nbrs[k].contains_pt(c)) {
          h = k;
          ++npct;
        }
      }
      LPM_REQUIRE(npct == 1);

      // set the table value: the child of h that contains pt c
      result[27*i+j] = p_nbrs[h].octree_child_idx(c);
    }
  }
  return result;
}

} // namespace tree
} // namespace Lpm
