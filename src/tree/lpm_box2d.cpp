#include "tree/lpm_box2d.hpp"

namespace Lpm {
namespace quadtree {

std::vector<Box2d> Box2d::bisect_all() const {
  std::vector<Box2d> result(4);
  const auto c = centroid();
  for (auto i=0; i<4; ++i) {
    result[i].xmin = ((i&2) == 0 ? xmin : c[0]);
    result[i].xmax = ((i&2) == 0 ? c[0] : xmax);
    result[i].ymin = ((i&1) == 0 ? ymin : c[1]);
    result[i].ymax = ((i&1) == 0 ? c[1] : ymax);
  }
  return result;
}

std::vector<Box2d> Box2d::neighbors() const {
  LPM_REQUIRE(is_square());
  const auto c = centroid();
  const auto l = square_edge_length();
  std::vector<Box2d> result;
  for (auto i=0; i<9; ++i) {
    Kokkos::Tuple<Real,2> cntrd(c[0] + ((i/3)-1)*l,
                                c[1] + ((i%3)-1)*l);
    result.push_back(Box2d(cntrd, l));
    LPM_ASSERT(result[i].contains_pt(cntrd));
  }
  LPM_ASSERT(result.size() == 9);
  return result;
}

Box2d parent_from_child(const Box2d& kid_box, const Int kid_idx) {
  LPM_REQUIRE(kid_box.is_square());
  const auto p_cntrd = kid_box.vertex_crds(3-kid_idx);
  return Box2d(p_cntrd, 2*kid_box.square_edge_length());
}

std::ostream& operator << (std::ostream& os, const Box2d& b) {
    os << "(" << std::setw(4) << b.xmin << " " << std::setw(4) << b.xmax << " " << std::setw(4) << b.ymin << " "
              << std::setw(4) << b.ymax << ")";
    return os;
}

}
}
