#include "LpmPolyMesh2d.hpp"

namespace Lpm {


/// ETI
template class PolyMesh2d<TriFace, TriHexSeed>;
template class PolyMesh2d<QuadFace, QuadRectSeed>;
template class PolyMesh2d<TriFace, IcosTriSphereSeed>;
template class PolyMesh2d<QuadFace, CubedSphereSeed>;
}
