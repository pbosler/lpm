#include "lpm_coords.hpp"
#include "lpm_coords_impl.hpp"


namespace Lpm {

template <>
void Coords<SphereGeometry>::init_random(const Real max_range, const Int ss) {
  // todo: replace this with kokkos random parallel_for.
  unsigned seed = 0 + ss;
  std::default_random_engine generator(seed);
  std::uniform_real_distribution<Real> randDist(-1.0, 1.0);
  for (Index i = 0; i < _nmax; ++i) {
    Real uu = randDist(generator);
    Real vv = randDist(generator);
    while (uu * uu + vv * vv > 1.0) {
      uu = randDist(generator);
      vv = randDist(generator);
    }
    const Real uv2 = uu * uu + vv * vv;
    const Real uvr = std::sqrt(1 - uv2);
    const Real cvec[3] = {2 * uu * uvr * max_range, 2 * vv * uvr * max_range,
                          (1 - 2 * uv2) * max_range};
    insert_host(cvec);
  }
  update_device();
}

/// ETI
template class Coords<PlaneGeometry>;
template class Coords<SphereGeometry>;
template class Coords<CircularPlaneGeometry>;

template void Coords<PlaneGeometry>::init_vert_crds_from_seed(
    const MeshSeed<TriHexSeed>& seed);
template void Coords<PlaneGeometry>::init_interior_crds_from_seed(
    const MeshSeed<TriHexSeed>& seed);
template void Coords<PlaneGeometry>::init_vert_crds_from_seed(
    const MeshSeed<QuadRectSeed>& seed);
template void Coords<PlaneGeometry>::init_interior_crds_from_seed(
    const MeshSeed<QuadRectSeed>& seed);
template void Coords<SphereGeometry>::init_vert_crds_from_seed(
    const MeshSeed<CubedSphereSeed>& seed);
template void Coords<SphereGeometry>::init_interior_crds_from_seed(
    const MeshSeed<CubedSphereSeed>& seed);
template void Coords<SphereGeometry>::init_vert_crds_from_seed(
    const MeshSeed<IcosTriSphereSeed>& seed);
template void Coords<SphereGeometry>::init_interior_crds_from_seed(
    const MeshSeed<IcosTriSphereSeed>& seed);
template void Coords<CircularPlaneGeometry>::init_vert_crds_from_seed(
    const MeshSeed<UnitDiskSeed>& seed);
template void Coords<CircularPlaneGeometry>::init_interior_crds_from_seed(
    const MeshSeed<UnitDiskSeed>& seed);

}  // namespace Lpm
