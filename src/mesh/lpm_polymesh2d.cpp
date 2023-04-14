#include "mesh/lpm_polymesh2d.hpp"

#include "mesh/lpm_polymesh2d_impl.hpp"

namespace Lpm {

/// ETI
template class PolyMesh2d<TriHexSeed>;
template class PolyMesh2d<QuadRectSeed>;
template class PolyMesh2d<IcosTriSphereSeed>;
template class PolyMesh2d<CubedSphereSeed>;
}  // namespace Lpm
