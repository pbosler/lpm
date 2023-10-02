#include "lpm_swe.hpp"
#include "lpm_swe_impl.hpp"

namespace Lpm {

// ETI
template class SWE<QuadRectSeed>;
template class SWE<TriHexSeed>;
template class SWE<CubedSphereSeed>;
template class SWE<IcosTriSphereSeed>;

} // namespace Lpm
