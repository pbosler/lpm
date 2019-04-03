#include <iostream>
#include "LpmTypeDefs.hpp"
#include "LpmUtilities.hpp"
#include "LpmRealVector.hpp"

#include <cmath>

namespace Lpm {

Real sphereTriArea(const RealVec<3>& a, const RealVec<3>& b, const RealVec<3>& c, const Real radius) {
    const Real s1 = a.sphereDist(b, radius);
    const Real s2 = b.sphereDist(c, radius);
    const Real s3 = c.sphereDist(a, radius);
    const Real halfPerim = 0.5*(s1 + s2 + s3);
    const Real zz = std::tan(0.5*halfPerim) * std::tan(0.5*(halfPerim-s1)) * std::tan(0.5*(halfPerim-s2)) *
        std::tan(0.5*(halfPerim-s3));
    return 4.0 * std::atan(std::sqrt(zz)) * radius*radius;
}

std::ostream& operator << (std::ostream& os, const RealVec<1>& vec) {
    os << "(" << vec.x[0] << ")" << std::endl;
    return os;
}

std::ostream& operator << (std::ostream& os, const RealVec<2>& vec) {
    os << "(" << vec.x[0] << ", " << vec.x[1] << ")";
    return os;
}

std::ostream& operator << (std::ostream& os, const RealVec<3>& vec) {
    os << "(" << vec.x[0] << ", " << vec.x[1] << ", " << vec.x[2] << ")";
    return os;
}

template class RealVec<2>;
template class RealVec<3>;
}

