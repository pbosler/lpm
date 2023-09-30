#ifndef LPM_SWE_SOLVERS_HPP
#define LPM_SWE_SOLVERS_HPP

#include "LpmConfig.h"
#include "lpm_swe.hpp"

namespace Lpm {

template <typename SeedType>
class SWERK4 {
  public:
  static constexpr Real sixth = 1.0/6;
  static constexpr Real third = 1.0/3;

  using typename SeedType::geo::crd_view_type = crd_view;
  using typename SeedType::geo::vec_view_type = vec_view;

  Real dt;

  SWERK4(SWE<SeedType>& swe, const Real dt);

  template <typename SurfaceLaplacianType>
  void initialize_tendencies(SurfaceLaplacianType& lap);

  template <typename SurfaceLaplacianType, typename BottomType>
  void advance_timestep(SurfaceLaplacianType& lap, const BottomType& topo);

  private:

  SWE<SeedType>& swe;

  crd_view x_passive;
  crd_view x_active;
  vec_view velocity_passive;
  vec_view velocity_active;
  scalar_view_type rel_vort_passive;
  scalar_view_type rel_vort_active;
  scalar_view_type div_passive;
  scalar_view_type div_active;
  scalar_view_type depth_passive;
  scalar_view_type area_active;
  scalar_view_type mass_active;

  crd_view x_passive_1;
  crd_view x_passive_2;
  crd_view x_passive_3;
  crd_view x_passive_4;
  crd_view x_passive_work;

  scalar_view_type rel_vort_passive_1;
  scalar_view_type rel_vort_passive_2;
  scalar_view_type rel_vort_passive_3;
  scalar_view_type rel_vort_passive_4;
  scalar_view_type rel_vort_passive_work;

  scalar_view_type div_passive_1;
  scalar_view_type div_passive_2;
  scalar_view_type div_passive_3;
  scalar_view_type div_passive_4;
  scalar_view_type div_passive_work;

  scalar_view_type depth_passive_1;
  scalar_view_type depth_passive_2;
  scalar_view_type depth_passive_3;
  scalar_view_type depth_passive_4;
  scalar_view_type depth_passive_work;

  scalar_view_type double_dot_passive;
  scalar_view_type surf_passive;
  scalar_view_type surf_laplacian_passive;

  crd_view x_active_1;
  crd_view x_active_2;
  crd_view x_active_3;
  crd_view x_active_4;
  crd_view x_active_work;

  scalar_view_type rel_vort_active_1;
  scalar_view_type rel_vort_active_2;
  scalar_view_type rel_vort_active_3;
  scalar_view_type rel_vort_active_4;
  scalar_view_type rel_vort_active_work;

  scalar_view_type div_active_1;
  scalar_view_type div_active_2;
  scalar_view_type div_active_3;
  scalar_view_type div_active_4;
  scalar_view_type div_active_work;

  scalar_view_type area_active_1;
  scalar_view_type area_active_2;
  scalar_view_type area_active_3;
  scalar_view_type area_active_4;
  scalar_view_type area_active_work;

  scalar_view_type double_dot_active;
  scalar_view_type surf_active;
  scalar_view_type depth_active;
  scalar_view_type surf_laplacian_active;

  struct PositionUpdate {
    crd_view x;
    crd_view x1;
    crd_view x2;
    crd_view x3;
    crd_view x4;

    PositionUpdate(crd_view& x, const crd_view& x1, const crd_view& x2, const crd_view& x3, const crd_view& x4) :
      x(x), x1(x1), x2(x2), x3(x3), x4(x4) {}

    KOKKOS_INLINE_FUNCTION
    void operator() (const Index i, const Index j) const {
      x(i,j) += sixth * (x1(i,j) + x4(i,j)) + third * (x2(i,j) + x3(i,j));
    }
  };

  struct ScalarUpdate {
    scalar_view_type s;
    scalar_view_type s1;
    scalar_view_type s2;
    scalar_view_type s3;
    scalar_view_type s4;

    ScalarUpdate(scalar_view_type s, const scalar_view_type s1,
      const scalar_view_type s2,
      const scalar_view_type s3,
      const scalar_view_type s4)  : s(s), s1(s1), s2(s2), s3(s3), s4(s4) {}

    KOKKOS_INLINE_FUNCTION
    void operator() (const Index i) const {
      s(i) += sixth * (s1(i) + s4(i)) + third * (s2(i) + s3(i));
    }
  };



};

} // namespace Lpm

#endif
