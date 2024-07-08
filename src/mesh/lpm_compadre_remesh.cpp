#include "mesh/lpm_compadre_remesh.hpp"
#include "mesh/lpm_compadre_remesh_impl.hpp"

namespace Lpm {

template class CompadreRemesh<QuadRectSeed>;
template class CompadreRemesh<TriHexSeed>;
template class CompadreRemesh<CubedSphereSeed>;
template class CompadreRemesh<IcosTriSphereSeed>;
}
