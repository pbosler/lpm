#include "lpm_incompressible2d_rk2.hpp"

#include "lpm_incompressible2d_rk2_impl.hpp"

namespace Lpm {

template class Incompressible2DRK2<QuadRectSeed>;
template class Incompressible2DRK2<TriHexSeed>;
template class Incompressible2DRK2<CubedSphereSeed>;
template class Incompressible2DRK2<IcosTriSphereSeed>;

}  // namespace Lpm
