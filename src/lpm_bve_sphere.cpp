#include "lpm_bve_sphere.hpp"
#include "lpm_bve_sphere_impl.hpp"


namespace Lpm {

/// ETI
template class BVESphere<IcosTriSphereSeed>;
template class BVESphere<CubedSphereSeed>;

template void output_vtk(const std::shared_ptr<BVESphere<CubedSphereSeed>>, const std::string& fname);
}
