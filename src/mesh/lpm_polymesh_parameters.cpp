#include "lpm_polymesh2d_parameters.hpp"
#include "lpm_polymesh2d_parameters_impl.hpp"

namespace Lpm {

/// ETI
template struct PolyMeshParameters<TriHexSeed>;
template struct PolyMeshParameters<QuadRectSeed>;
template struct PolyMeshParameters<IcosTriSphereSeed>;
template struct PolyMeshParameters<CubedSphereSeed>;
}
