#include "tree/lpm_box3d.hpp"

namespace Lpm {

std::vector<Box3d> Box3d::bisect_all() const {
  std::vector<Box3d> result(8);
  const auto cntrd = centroid();
  for (int i=0; i<8; ++i) {
    result[i].xmin = ((i&4) == 0 ? xmin : cntrd[0]);
    result[i].xmax = ((i&4) == 0 ? cntrd[0] : xmax);
    result[i].ymin = ((i&2) == 0 ? ymin : cntrd[1]);
    result[i].ymax = ((i&2) == 0 ? cntrd[1] : ymax);
    result[i].zmin = ((i&1) == 0 ? zmin : cntrd[2]);
    result[i].zmax = ((i&1) == 0 ? cntrd[2] : zmax);
  }
  return result;
}

std::vector<Box3d> Box3d::neighbors() const {
  LPM_REQUIRE(is_cube());
  const auto c = centroid();
  const auto l = cube_edge_length();
  std::vector<Box3d> result;
  result.reserve(27);
  for (short j=0; j<27; ++j) {
    Kokkos::Tuple<Real,3> ctrd(c[0] + ((j/9)%3 - 1)*l,
                               c[1] + ((j/3)%3 - 1)*l,
                               c[2] + (    j%3 - 1)*l);
    result.push_back(Box3d(ctrd, l));
    LPM_ASSERT(result[j].contains_pt(ctrd));
  }
  return result;
}

Box3d parent_from_child(const Box3d& kid, const Int kid_idx) {
  LPM_REQUIRE(kid.is_cube());

  Kokkos::Tuple<Real,3> parent_ctrd;
  kid.vertex_crds(parent_ctrd, 7-kid_idx);
  return Box3d(parent_ctrd, 2*kid.cube_edge_length());
}

std::ostream& operator << (std::ostream& os, const Box3d& b) {
    os << "(" << std::setw(4) << b.xmin << " " << std::setw(4) << b.xmax << " " << std::setw(4) << b.ymin << " "
              << std::setw(4) << b.ymax << " " << std::setw(4) << b.zmin << " " << std::setw(4) << b.zmax << ")";
    return os;
}

}
