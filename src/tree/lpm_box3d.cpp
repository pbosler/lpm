#include "LpmBox3d.hpp"

namespace Lpm {
namespace Octree {

std::ostream& operator << (std::ostream& os, const BBox& b) {
    os << "(" << std::setw(4) << b.xmin << " " << std::setw(4) << b.xmax << " " << std::setw(4) << b.ymin << " " 
              << std::setw(4) << b.ymax << " " << std::setw(4) << b.zmin << " " << std::setw(4) << b.zmax << ")\n";
    return os;
}

}}
