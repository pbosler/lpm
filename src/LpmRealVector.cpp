#include <iostream>
#include "LpmTypeDefs.hpp"
#include "LpmUtilities.hpp"
#include "LpmRealVector.hpp"

#include <cmath>

namespace Lpm {

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

