#ifndef LPM_INCOMPRESSIBLE2D_RK2_HPP
#define LPM_INCOMPRESSIBLE2D_RK2_HPP

#include "LpmConfig.h"
#include "lpm_incompressible2d.hpp"

namespace Lpm {

template <typename SeedType>
class Incompressible2DRK2 {
  public:
  using geo = typename SeedType::geo;
  using crd_view = typename SeedType::geo::crd_view_type;
  using vec_view = typename SeedType::geo::vec_view_type;

  Real dt;
  Incompressible2D<SeedType>& ic2d;

  Int t_idx;

  Index n_passive;
  Index n_active;


  Real eps; /// velocity kernel smoothing parameter

  std::string info_string(const int tab_level=0) const;

  void advance_timestep_impl();

  Incompressible2DRK2(const Real dt, Incompressible2D<SeedType>& ic2d);

  Incompressible2DRK2(const Real dt, Incompressible2D<SeedType>& ic2d, const Index t_idx);

  private:

    crd_view passive_x1;
    crd_view passive_x2;
    crd_view passive_xwork;

    scalar_view_type passive_rel_vort1;
    scalar_view_type passive_rel_vort2;
    scalar_view_type passive_rel_vortwork;

    crd_view active_x1;
    crd_view active_x2;
    crd_view active_xwork;

    scalar_view_type active_rel_vort1;
    scalar_view_type active_rel_vort2;
    scalar_view_type active_rel_vortwork;

    std::unique_ptr<Kokkos::TeamPolicy<>> passive_policy;
    std::unique_ptr<Kokkos::TeamPolicy<>> active_policy;
};


} // namespace Lpm

#endif
