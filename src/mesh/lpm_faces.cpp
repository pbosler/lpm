#include "mesh/lpm_faces_impl.hpp"

namespace Lpm {

/// ETI
template class Faces<TriFace, PlaneGeometry>;
template class Faces<TriFace, SphereGeometry>;
template class Faces<QuadFace, PlaneGeometry>;
template class Faces<QuadFace, SphereGeometry>;

template void Faces<TriFace, PlaneGeometry>::init_from_seed(
    const MeshSeed<TriHexSeed>& seed);
template void Faces<TriFace, SphereGeometry>::init_from_seed(
    const MeshSeed<IcosTriSphereSeed>& seed);
template void Faces<QuadFace, PlaneGeometry>::init_from_seed(
    const MeshSeed<QuadRectSeed>& seed);
template void Faces<QuadFace, SphereGeometry>::init_from_seed(
    const MeshSeed<CubedSphereSeed>& seed);

template struct FaceDivider<PlaneGeometry, TriFace>;
template struct FaceDivider<SphereGeometry, TriFace>;
template struct FaceDivider<PlaneGeometry, QuadFace>;
template struct FaceDivider<SphereGeometry, QuadFace>;

}  // namespace Lpm
