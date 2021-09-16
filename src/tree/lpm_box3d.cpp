#include "tree/lpm_box3d.hpp"

namespace Lpm {
namespace tree {

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

std::ostream& operator << (std::ostream& os, const Box3d& b) {
    os << "(" << std::setw(4) << b.xmin << " " << std::setw(4) << b.xmax << " " << std::setw(4) << b.ymin << " "
              << std::setw(4) << b.ymax << " " << std::setw(4) << b.zmin << " " << std::setw(4) << b.zmax << ")\n";
    return os;
}

}}
