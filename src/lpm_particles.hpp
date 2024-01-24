#ifndef LPM_PARTICLES_HPP
#define LPM_PARTICLES_HPP

#include "LpmConfig.h"
#include "lpm_coords.hpp"

namespace Lpm {

template <typename Geo>
class Particles {
  using coords_type = Coords<Geo>;
  using crd_view = typename Geo::crd_view_type;

  // Particles have physical coordinates
  Coords<Geo> phys_crds;
  // Particles have Lagrangian coordinates
  Coords<Geo> lag_crds;

  explicit Particles(const Index nmax);

#ifdef LPM_USE_NETCDF
  // TODO: Particles should init from a netcdf file
#endif

#ifdef LPM_USE_VTK
  // TODO: Particles should output vtk
#endif

  void update_host() const;

  void update_device() const;

  inline void nh() const {return _nh();}

  KOKKOS_INLINE_FUNCTION
  Index n_max() const {return phys_crds.view.extent(0);}

  template <typename CV>
  void insert_host(const CV& v);

  std::string info_string(const int tab_level=0) const;

  void init_random(const Real max_range = 1.0);
};

}  // namespace Lpm
#endif
