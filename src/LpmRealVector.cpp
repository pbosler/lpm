#include <iostream>
#include "LpmAosTypes.hpp"
#include "LpmUtilities.h"

#include <cmath>

namespace Lpm {
namespace Aos {

scalar_type sphereTriArea(const Vec<3>& a, const Vec<3>& b, const Vec<3>& c, const scalar_type radius) {
    const scalar_type s1 = a.sphereDist(b, radius);
    const scalar_type s2 = b.sphereDist(c, radius);
    const scalar_type s3 = c.sphereDist(a, radius);
    const scalar_type halfPerim = 0.5*(s1 + s2 + s3);
    const scalar_type zz = std::tan(0.5*halfPerim) * std::tan(0.5*(halfPerim-s1)) * std::tan(0.5*(halfPerim-s2)) *
        std::tan(0.5*(halfPerim-s3));
    return 4.0 * std::atan(std::sqrt(zz)) * radius*radius;
}

std::ostream& operator << (std::ostream& os, const Vec<1>& vec) {
    os << "(" << vec.x[0] << ")" << std::endl;
}

std::ostream& operator << (std::ostream& os, const Vec<2>& vec) {
    os << "(" << vec.x[0] << ", " << vec.x[1] << ")";
    return os;
}

std::ostream& operator << (std::ostream& os, const Vec<3>& vec) {
    os << "(" << vec.x[0] << ", " << vec.x[1] << ", " << vec.x[2] << ")";
    return os;
}

template class Vec<2>;
template class Vec<3>;

}
}

