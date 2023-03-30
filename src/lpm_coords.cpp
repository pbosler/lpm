#include "lpm_coords.hpp"

#include <random>

#include "util/lpm_string_util.hpp"

namespace Lpm {

template <typename Geo>
std::string Coords<Geo>::info_string(const std::string& label,
                                     const short& tab_level,
                                     const bool& dump_all) const {
  std::ostringstream oss;
  const std::string tabstr = indent_string(tab_level);
  oss << tabstr << "Coords " << label << " info: nh = (" << _nh()
      << ") of nmax = " << _nmax << " in memory" << std::endl;
  if (dump_all) {
    for (Index i = 0; i < _nmax; ++i) {
      if (i == _nh()) oss << tabstr << "---------------------------------\n";
      oss << tabstr << "\t" << label << ": (" << i << ") : ";
      for (Int j = 0; j < Geo::ndim; ++j) oss << "\t" << _hostview(i, j) << " ";
      oss << std::endl;
    }
  }
  return oss.str();
}

template <typename Geo>
void Coords<Geo>::write_matlab(std::ostream& os,
                               const std::string& name) const {
  os << name << " = [";
  for (Index i = 0; i < _nh(); ++i) {
    for (int j = 0; j < Geo::ndim; ++j) {
      os << _hostview(i, j)
         << (j == Geo::ndim - 1 ? (i == _nh() - 1 ? "];" : ";") : ",");
    }
  }
  os << std::endl;
}

template <typename Geo>
void Coords<Geo>::init_random(const Real max_range, const Int ss) {
  // todo: replace this with kokkos random parallel_for.
  unsigned seed = 0 + ss;
  std::default_random_engine generator(seed);
  std::uniform_real_distribution<Real> randDist(-max_range, max_range);
  for (Index i = 0; i < _nmax; ++i) {
    Real cvec[Geo::ndim];
    for (Int j = 0; j < Geo::ndim; ++j) {
      cvec[j] = randDist(generator);
    }
    insert_host(cvec);
  }
  update_device();
}

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

template <typename Geo>
Real Coords<Geo>::max_radius() const {
  Real result = 0;
  const auto local_crds = this->view;
  Kokkos::parallel_reduce(
      _nh(),
      KOKKOS_LAMBDA(const Index i, Real& m) {
        const auto mcrd = Kokkos::subview(local_crds, i, Kokkos::ALL);
        const Real r = Geo::mag(mcrd);
        if (r > m) m = r;
      },
      Kokkos::Max<Real>(result));
  return result;
}

template <typename Geo>
template <typename SeedType>
void Coords<Geo>::init_vert_crds_from_seed(const MeshSeed<SeedType>& seed) {
  LPM_REQUIRE_MSG(_nmax >= SeedType::nverts,
                  "coords::init_vert_crds_from_seed memory limit error");
  for (int i = 0; i < SeedType::nverts; ++i) {
    for (int j = 0; j < Geo::ndim; ++j) {
      _hostview(i, j) = seed.seed_crds(i, j);
    }
  }
  _nh() = SeedType::nverts;
}

template <typename Geo>
template <typename SeedType>
void Coords<Geo>::init_interior_crds_from_seed(const MeshSeed<SeedType>& seed) {
  LPM_REQUIRE_MSG(_nmax >= SeedType::nfaces,
                  "Coords::init_interior_crds_from_seed memory limit error.");
  for (int i = 0; i < SeedType::nfaces; ++i) {
    for (int j = 0; j < Geo::ndim; ++j) {
      _hostview(i, j) = seed.seed_crds(SeedType::nverts + i, j);
    }
  }
  _nh() = SeedType::nfaces;
}

template <typename Geo>
Kokkos::MinMaxScalar<Real> Coords<Geo>::min_max_extent(const int dim) const {
  Kokkos::MinMaxScalar<Real> minmax;
  const auto local_crds = this->view;
  Kokkos::parallel_reduce(
      _nh(),
      KOKKOS_LAMBDA(const Index i, Kokkos::MinMaxScalar<Real>& mm) {
        if (local_crds(i, dim) < mm.min_val) mm.min_val = local_crds(i, dim);
        if (local_crds(i, dim) > mm.max_val) mm.max_val = local_crds(i, dim);
      },
      Kokkos::MinMax<Real>(minmax));
  return minmax;
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
