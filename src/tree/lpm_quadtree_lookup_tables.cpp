#include "tree/lpm_quadtree_lookup_tables.hpp"
#include "tree/lpm_box2d.hpp"

namespace Lpm {
namespace quadtree {

std::array<Int,36> parent_lookup_table_entries() {
  std::array<Int,36> result;

  const auto pbox = Box2d(-1,1,-1,1,false);
  const auto p_neighbors = pbox.neighbors();
  const auto kids = pbox.bisect_all();

  for (auto i=0; i<4; ++i) {

    LPM_ASSERT(pbox.contains_pt(kids[i].centroid()));

    const auto kid_nbrs = kids[i].neighbors();
    for (int j=0; j<9; ++j) {
      const auto kc = kid_nbrs[j].centroid();
      int n=0;
      for (int k=0; k<9; ++k) {
        if (p_neighbors[k].contains_pt(kc)) {
          ++n;
          result[9*i + j] = k;
        }
      }
      LPM_ASSERT(n==1);
    }
  }
  return result;
}

std::array<Int,36> child_lookup_table_entries() {
  std::array<Int,36> result;

  const auto p = Box2d(-1,1,-1,1,false);
  const auto p_nbrs = p.neighbors();
  const auto kids = p.bisect_all();

  for (int i=0; i<4; ++i) {
    const auto t = kids[i];
    const auto t_nbrs = t.neighbors();

    for (int j=0; j<9; ++j) {
      const auto c = t_nbrs[j].centroid();

      int n=0;
      int h;
      for (int k=0; k<9; ++k) {
        if (p_nbrs[k].contains_pt(c)) {
          ++n;
          h = k;
        }
      }
      LPM_ASSERT(n==1);
      result[9*i + j] = p_nbrs[h].quadtree_child_idx(c);
    }
  }

  return result;
}


}
}
