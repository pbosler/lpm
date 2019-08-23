#ifndef LPM_MATLAB_HPP
#define LPM_MATLAB_HPP

#include "LpmConfig.h"
#include "LpmLatLonMesh.hpp"
#include "LpmDefs.hpp"
#include "LpmUtilities.hpp"
#include "LpmKokkosUtil.hpp"
#include "Kokkos_Core.hpp"
#include <iostream>
#include <string>

namespace Lpm {

void writeVectorMatlab(std::ostream& os, const std::string name, const ko::View<const Real*,HostMem> v);

void writeArrayMatlab(std::ostream& os, const std::string name, const ko::View<const Real**,HostMem> a);

}
#endif 
