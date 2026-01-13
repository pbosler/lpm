#ifndef LPM_PARTICLES_IMPL_HPP
#define LPM_PARTICLES_IMPL_HPP

#include "lpm_coords_impl.hpp"
#include "lpm_particles.hpp"
#include "util/lpm_string_util.hpp"

namespace Lpm {

template <typename Geo>
Particles<Geo>::Particles(const Index nmax) : phys_crds(nmax), lag_crds(nmax) {}

template <typename Geo>
void Particles<Geo>::update_device() const {
  phys_crds.update_device();
  lag_crds.update_device();
}

template <typename Geo>
void Particles<Geo>::update_host() const {
  phys_crds.update_host();
  lag_crds.update_host();
}

template <typename Geo>
template <typename CV>
void insert_host(const CV& v) {
  phys_crds.insert_host(v);
  lag_crds.insert_host(v);
}

template <typename Geo>
std::string Particles<Geo>::info_string(const int tab_level) const {
  std::ostringstream ss;
  ss << phys_crds.info_string(tab_level);
  ss << lag_crds.info_string(tab_level);
  return ss.str();
}

template <typename Geo>
void Particles<Geo>::init_random(const Real max_range) {
  phys_crds.init_random(max_range);
  phys_crds.update_device();
  Kokkos::deep_copy(lag_crds.view, phys_crds.view);
  lag_crds.update_host();
}

#ifdef LPM_USE_NETCDF

#endif

}  // namespace Lpm

#endif
