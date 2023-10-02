#include "mesh/lpm_mesh_seed.hpp"
#include "lpm_swe_surface_laplacian.hpp"
#include "lpm_swe_surface_laplacian_impl.hpp"

namespace Lpm {

// ETI

template class SWEGMLSLaplacian<QuadRectSeed>;
template class SWEGMLSLaplacian<TriHexSeed>;
template class SWEGMLSLaplacian<CubedSphereSeed>;
template class SWEGMLSLaplacian<IcosTriSphereSeed>;


} // namespace Lpm
