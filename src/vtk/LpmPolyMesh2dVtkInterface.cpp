#include "LpmPolyMesh2dVtkInterface.hpp"
#include "LpmPolyMesh2dVtkInterface_Impl.hpp"
#include <cassert>

namespace Lpm {

/// ETI
template class Polymesh2dVtkInterface<CubedSphereSeed>;
template class Polymesh2dVtkInterface<IcosTriSphereSeed>;

}
