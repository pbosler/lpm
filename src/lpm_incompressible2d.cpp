#include "lpm_incompressible2d.hpp"

#include "lpm_incompressible2d_impl.hpp"

namespace Lpm {

// ETI
template class Incompressible2D<QuadRectSeed>;
template class Incompressible2D<TriHexSeed>;
template class Incompressible2D<CubedSphereSeed>;
template class Incompressible2D<IcosTriSphereSeed>;

}  // namespace Lpm
