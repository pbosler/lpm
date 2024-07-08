#ifndef LPM_SWE_RK2_STAGGERED_IMPL_HPP
#define LPM_SWE_RK2_STAGGERED_IMPL_HPP

#include "LpmConfig.h"
#include "lpm_swe_kernels.hpp"
#include "lpm_swe_rk2_staggered.hpp"
#include "mesh/lpm_gather_mesh_data.hpp"
#include "mesh/lpm_scatter_mesh_data.hpp"

#include <KokkosBlas.hpp>

namespace Lpm {

template <typename Geo>
struct StaggeredSWETendencies {
  using crd_view = typename Geo::crd_view_type;
  using vec_view = typename Geo::vec_view_type;
  using coriolis_type = typename std::conditional<
    std::is_same<Geo, PlaneGeometry>::value, CoriolisBetaPlane,
      CoriolisSphere>::type;

  scalar_view_type dzeta; // output
  scalar_view_type dsigma; // output
  scalar_view_type darea; // output
  crd_view x; // input
  vec_view u; // input;
  scalar_view_type zeta; //input
  scalar_view_type sigma; // input
  scalar_view_type area; // input
  scalar_view_type grad_f_cross_u; // input
  scalar_view_type ddot; // input
  scalar_view_type laps; // input
  mask_view_type mask; // input
  coriolis_type coriolis; //input
  Real g; // input
  Real dt; // input

  StaggeredSWETendencies(scalar_view_type& dz, scalar_view_type& ds,
    scalar_view_type& da, const crd_view x, const vec_view u,
    const scalar_view_type zeta, const scalar_view_type sigma,
    const scalar_view_type area,
    const scalar_view_type gfcu, const scalar_view_type ddot,
    const scalar_view_type laps, const mask_view_type mask,
    const coriolis_type& coriolis,
    const Real g, const Real dt) :
    dzeta(dz),
    dsigma(ds),
    darea(da),
    x(x),
    u(u),
    zeta(zeta),
    sigma(sigma),
    area(area),
    grad_f_cross_u(gfcu),
    ddot(ddot),
    laps(laps),
    mask(mask),
    coriolis(coriolis),
    g(g),
    dt(dt) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index i) const {
    if (!mask(i)) {
      const auto xi = Kokkos::subview(x, i, Kokkos::ALL);
      const auto ui = Kokkos::subview(u, i, Kokkos::ALL);
      const Real f = coriolis.f(xi);
      dzeta(i) = (-coriolis.dfdt(ui) - (zeta(i) + f)*sigma(i)) * dt;
      dsigma(i) = (f*zeta(i) + grad_f_cross_u(i) - ddot(i) - g*laps(i))*dt;
      darea(i) = (sigma(i) * area(i))*dt;
    }
  }
};

template <typename SeedType, typename TopoType>
SWERK2Staggered<SeedType, TopoType>::SWERK2Staggered(const Real timestep,
  StaggeredSWE<SeedType, TopoType>& swe, const gmls::Params& gmls_params) :
  dt(timestep),
  t_idx(0),
  swe(swe),
  eps(swe.eps),
  gmls_params(gmls_params),
  x1("x1", swe.mesh.n_vertices_host()),
  x2("x2", swe.mesh.n_vertices_host()),
  xwork("xwork", swe.mesh.n_vertices_host()),
  rel_vort1("rel_vort1", swe.mesh.n_faces_host()),
  rel_vort2("rel_vort2", swe.mesh.n_faces_host()),
  rel_vortwork("rel_vortwork", swe.mesh.n_faces_host()),
  divergence1("divergence1", swe.mesh.n_faces_host()),
  divergence2("divergence2", swe.mesh.n_faces_host()),
  divergencework("divergencework", swe.mesh.n_faces_host()),
  area1("area1", swe.mesh.n_faces_host()),
  area2("area2", swe.mesh.n_faces_host()),
  areawork("areawork", swe.mesh.n_faces_host()) {
    vertex_policy = std::make_unique<Kokkos::TeamPolicy<>>(swe.mesh.n_vertices_host(), Kokkos::AUTO());
  }

template <typename SeedType, typename TopoType>
void SWERK2Staggered<SeedType, TopoType>::advance_timestep_impl() {
  const Index nverts = swe.mesh.n_vertices_host();
  const Index nfaces = swe.mesh.n_faces_host();
  const auto vert_range = std::make_pair(0, nverts);
  const auto face_range = std::make_pair(0, nfaces);
  auto velocity_view = Kokkos::subview(swe.velocity.view, vert_range, Kokkos::ALL);
  auto velocity_avg_view = Kokkos::subview(swe.velocity_avg.view, face_range, Kokkos::ALL);
  auto x_view = Kokkos::subview(swe.mesh.vertices.phys_crds.view, vert_range, Kokkos::ALL);
  auto x_avg_view = Kokkos::subview(swe.mesh.faces.phys_crds.view, face_range, Kokkos::ALL);
  auto zeta_view = Kokkos::subview(swe.relative_vorticity.view, face_range);
  auto sigma_view = Kokkos::subview(swe.divergence.view, face_range);
  auto area_view = Kokkos::subview(swe.mesh.faces.area, face_range);
  auto gfcu_view = Kokkos::subview(swe.grad_f_cross_u.view, vert_range);
  auto gfcu_avg_view = Kokkos::subview(swe.grad_f_cross_u_avg.view, face_range);
  auto ddot_view = Kokkos::subview(swe.double_dot.view, vert_range);
  auto ddot_avg_view = Kokkos::subview(swe.double_dot_avg.view, face_range);
  auto slap_view = Kokkos::subview(swe.surface_laplacian.view, face_range);
  auto depth_view = Kokkos::subview(swe.depth.view, face_range);
  const auto mass_view = Kokkos::subview(swe.mass.view, face_range);
  auto s_view = Kokkos::subview(swe.surface_height.view, face_range);
  auto b_view = Kokkos::subview(swe.bottom_height.view, face_range);
  auto mask_view = Kokkos::subview(swe.mesh.faces.mask, face_range);
  auto vert_view = Kokkos::subview(swe.mesh.faces.verts, face_range, Kokkos::ALL);

  constexpr bool do_velocity = true;

  // RK Stage 1
  KokkosBlas::scal(x1, dt, velocity_view);
  Kokkos::parallel_for("RK2-1 tendencies", nfaces,
    StaggeredSWETendencies<typename SeedType::geo>(rel_vort1, divergence1, area1,
      x_avg_view, velocity_avg_view, zeta_view, sigma_view,
      area_view, gfcu_avg_view,
      ddot_avg_view, slap_view, mask_view, swe.coriolis, swe.g, dt));

  // intermediate update to t + dt
  KokkosBlas::update(1, x_view, 1, x1, 0, xwork);
  KokkosBlas::update(1, zeta_view, 1, rel_vort1, 0, rel_vortwork);
  KokkosBlas::update(1, sigma_view, 1, divergence1, 0, divergencework);
  KokkosBlas::update(1, area_view, 1, area1, 0, areawork);

  Kokkos::parallel_for(nfaces,
    FacePositionAverages<SeedType>(x_avg_view, xwork, vert_view));

  Kokkos::parallel_for("RK2-2 surface input", nfaces,
    SetDepthAndSurfaceFromMassAndArea<typename SeedType::geo, TopoType>(
      depth_view, s_view, b_view, x_avg_view, mass_view, areawork, mask_view, swe.topography));
  swe.gmls_surface_laplacian(x_avg_view, gmls_params);

  Kokkos::parallel_for(*vertex_policy,
    SphereVertexSums(velocity_view,
      ddot_view, xwork, x_avg_view,
      rel_vortwork, divergencework, areawork, mask_view, eps,
      nfaces, do_velocity));
  KokkosBlas::scal(x2, dt, velocity_view);

  Kokkos::parallel_for(nverts,
    GradFCrossU<typename SeedType::geo>(gfcu_view,
      xwork, velocity_view, swe.coriolis));
  Kokkos::parallel_for(nfaces,
    VertexToFaceAverages<SeedType>(x_avg_view,
      ddot_avg_view, gfcu_avg_view,
      velocity_avg_view, xwork, ddot_view,
      gfcu_view, velocity_view, vert_view));

  Kokkos::parallel_for("RK2-2 tendencies", nfaces,
    StaggeredSWETendencies<typename SeedType::geo>(rel_vort2, divergence2, area2,
      x_avg_view, velocity_avg_view, zeta_view, sigma_view, area_view,
      gfcu_avg_view,
      ddot_avg_view, slap_view, mask_view, swe.coriolis, swe.g, dt));

  // final update
  KokkosBlas::update(0.5, x1, 0.5, x2, 1, x_view);
  KokkosBlas::update(0.5, rel_vort1, 0.5, rel_vort2, 1, zeta_view);
  KokkosBlas::update(0.5, divergence1, 0.5, divergence2, 1, sigma_view);
  KokkosBlas::update(0.5, area1, 0.5, area2, 1, area_view);

  Kokkos::parallel_for(nfaces,
    FacePositionAverages<SeedType>(x_avg_view, x_view, vert_view));
  Kokkos::parallel_for(nfaces,
    SetDepthAndSurfaceFromMassAndArea<typename SeedType::geo, TopoType>(
      depth_view, s_view, b_view, x_avg_view, mass_view, area_view, mask_view, swe.topography));
  swe.gmls_surface_laplacian(x_avg_view, gmls_params);

  Kokkos::parallel_for(*vertex_policy,
    SphereVertexSums(velocity_view,
      ddot_view, x_view, x_avg_view, zeta_view, sigma_view, area_view,
      mask_view, eps, nfaces, do_velocity));
  Kokkos::parallel_for(nverts,
    GradFCrossU<typename SeedType::geo>(gfcu_view,
      x_view, velocity_view, swe.coriolis));
  Kokkos::parallel_for(nfaces,
    VertexToFaceAverages<SeedType>(x_avg_view, ddot_avg_view, gfcu_avg_view,
      velocity_avg_view, x_view, ddot_view, gfcu_view, velocity_view, vert_view));

  ++t_idx;
}

} // namespace Lpm

#endif
