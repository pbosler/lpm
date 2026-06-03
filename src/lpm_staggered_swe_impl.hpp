#ifndef LPM_STAGGERED_SWE_IMPL_HPP
#define LPM_STAGGERED_SWE_IMPL_HPP

#include "LpmConfig.h"
#include "lpm_constants.hpp"
#include "lpm_field_impl.hpp"
#include "lpm_geometry.hpp"
#include "lpm_staggered_swe.hpp"
#include "lpm_swe_kernels.hpp"
#include "mesh/lpm_polymesh2d_impl.hpp"
#include "mesh/lpm_scatter_mesh_data.hpp"

namespace Lpm {

/** Compute face coordinates as barycenter of vertices,
  compute velocity tensor double dot product at face centers
  by averaging vertex values.
*/
template <typename SeedType>
struct VertexToFaceAverages {
  using geo       = typename SeedType::geo;
  using crd_view  = typename geo::crd_view_type;
  using vec_view  = typename geo::vec_view_type;
  using vert_view = Kokkos::View<Index * [SeedType::nfaceverts]>;

  crd_view face_phys_crds;               // output
  scalar_view_type face_double_dot;      // output
  scalar_view_type face_grad_f_cross_u;  // output
  vec_view face_vel_avg;                 // output
  crd_view vert_phys_crds;               // input
  scalar_view_type vert_double_dot;      // input
  scalar_view_type vert_grad_f_cross_u;  // input
  vec_view vert_vel;                     // input
  vert_view face_verts;                  // input

  /** Constructor.  To be called from a Kokkos::parallel_for kernel launch.

      @param [in/out] face_x physical coordinates of faces
      @param [in/out] face_dd velocity tensor gradient double dot product at
     faces
      @param [in] vert_x physical coordinates of vertices
      @param [in] vert_dd velocity gradient tensor double dot product vertex
     values
      @param [in] fv list of face vertices
  */
  VertexToFaceAverages(crd_view face_x, scalar_view_type face_dd,
                       scalar_view_type face_gfcu, vec_view face_vel,
                       const crd_view vert_x, const scalar_view_type vert_dd,
                       const scalar_view_type vert_gfcu,
                       const vec_view vert_vel, const vert_view fv)
      : face_phys_crds(face_x),
        face_double_dot(face_dd),
        face_grad_f_cross_u(face_gfcu),
        face_vel_avg(face_vel),
        vert_phys_crds(vert_x),
        vert_double_dot(vert_dd),
        vert_grad_f_cross_u(vert_gfcu),
        vert_vel(vert_vel),
        face_verts(fv) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index i) const {
    Index vertices[SeedType::nfaceverts];
    face_double_dot(i)     = 0;
    face_grad_f_cross_u(i) = 0;
    for (int j = 0; j < SeedType::nfaceverts; ++j) {
      const Index vert_idx = face_verts(i, j);
      vertices[j]          = vert_idx;
      face_double_dot(i) += vert_double_dot(vert_idx);
      face_grad_f_cross_u(i) += vert_grad_f_cross_u(vert_idx);
    }
    auto fxi = Kokkos::subview(face_phys_crds, i, Kokkos::ALL);
    auto fui = Kokkos::subview(face_vel_avg, i, Kokkos::ALL);
    // TODO: should these be geometric averages, instead of arithmetic averages?
    geo::barycenter(fxi, vert_phys_crds, vertices, SeedType::nfaceverts);
    geo::vector_average(fui, fxi, vert_vel, vertices, SeedType::nfaceverts);
    face_double_dot(i) /= SeedType::nfaceverts;
    face_grad_f_cross_u(i) /= SeedType::nfaceverts;
  }
};

template <typename SeedType>
struct FacePositionAverages {
  using geo       = typename SeedType::geo;
  using crd_view  = typename geo::crd_view_type;
  using vert_view = Kokkos::View<Index * [SeedType::nfaceverts]>;

  crd_view face_x_avg;   // output
  crd_view vert_x;       // input
  vert_view face_verts;  // input

  FacePositionAverages(crd_view fx_avg, const crd_view& vx, const vert_view& fv)
      : face_x_avg(fx_avg), vert_x(vx), face_verts(fv) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index i) const {
    Index vertices[SeedType::nfaceverts];
    for (int j = 0; j < SeedType::nfaceverts; ++j) {
      const Index vert_idx = face_verts(i, j);
      vertices[j]          = vert_idx;
    }
    auto fxi = Kokkos::subview(face_x_avg, i, Kokkos::ALL);
    geo::barycenter(fxi, vert_x, vertices, SeedType::nfaceverts);
  }
};

/** Initialize surface height, bottom height, depth, and mass values.

  These fields are all defined on faces.
*/
template <typename Geo, typename TopoType, typename SurfaceType>
struct SurfaceMassDepthInit {
  using crd_view = typename Geo::crd_view_type;

  scalar_view_type surface_height_view;  /// output
  scalar_view_type bottom_view;          /// output
  scalar_view_type depth_view;           /// output
  scalar_view_type mass_view;            /// output
  scalar_view_type area_view;            /// input
  crd_view lag_crd_view;                 /// input
  TopoType topo;                         /// input
  SurfaceType sfc;                       /// input

  /** Constrcutor, to be called from a Kokkos::parallel_for kernel launch.

    @param [in/out] s_view surface height view
    @param [in/out] b_view bottom height view
    @param [in/out] h_view depth view
    @param [in/out] m_view mass view
    @param [in] a_view face area view
    @param [in] lag_view face Lagrangian coordinate view
    @param [in] topo bottom topography functor (see lpm_surface_gallery.hpp)
    @param [in] sfc surface height functor (see lpm_surface_gallery.hpp)
  */
  SurfaceMassDepthInit(scalar_view_type s_view, scalar_view_type b_view,
                       scalar_view_type h_view, scalar_view_type m_view,
                       const scalar_view_type a_view, const crd_view lag_view,
                       const TopoType& topo, const SurfaceType& sfc)
      : surface_height_view(s_view),
        bottom_view(b_view),
        depth_view(h_view),
        mass_view(m_view),
        area_view(a_view),
        lag_crd_view(lag_view),
        topo(topo),
        sfc(sfc) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index i) const {
    const auto ai          = Kokkos::subview(lag_crd_view, i, Kokkos::ALL);
    const Real b           = topo(ai);
    const Real s           = sfc(ai);
    const Real h           = s - b;
    bottom_view(i)         = b;
    surface_height_view(i) = s;
    depth_view(i)          = h;
    mass_view(i)           = h * area_view(i);
  }
};

/** Initialize relative vorticity and potential vorticity.

  These fields are defined on faces.
*/
template <typename Geo, typename VorticityType>
struct VorticityInit {
  using crd_view = typename Geo::crd_view_type;
  using Coriolis =
      typename std::conditional<std::is_same<Geo, PlaneGeometry>::value,
                                CoriolisBetaPlane, CoriolisSphere>::type;

  scalar_view_type relative_vorticity_view;   /// output
  scalar_view_type potential_vorticity_view;  /// output
  scalar_view_type depth_view;                /// input
  crd_view lag_crd_view;                      /// input
  VorticityType vorticity;                    /// input
  Coriolis coriolis;                          /// input

  /** Constructor. to be called from a Kokkos::parallel_for kernel launch.

    @warning This functor must be called after depth values have been
    initialized.

    @param [in/out] zeta_view relative vorticity view
    @param [in/out] Q_view potential vorticity view
    @param [in] h_view depth view
    @param [in] lag_view face Lagrangian coordinate view
    @param [in] vorticity vorticity functor (see lpm_vorticity_gallery.hpp)
    @param [in] coriolis Coriolis functor (see lpm_coriolis.hpp)
  */
  VorticityInit(scalar_view_type zeta_view, scalar_view_type Q_view,
                const scalar_view_type h_view, const crd_view lag_view,
                const VorticityType& vorticity, const Coriolis& coriolis)
      : relative_vorticity_view(zeta_view),
        potential_vorticity_view(Q_view),
        depth_view(h_view),
        lag_crd_view(lag_view),
        vorticity(vorticity),
        coriolis(coriolis) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index i) const {
    const auto ai              = Kokkos::subview(lag_crd_view, i, Kokkos::ALL);
    const Real zeta            = vorticity(ai);
    relative_vorticity_view(i) = zeta;
    potential_vorticity_view(i) =
        (depth_view(i) > 0 ? (zeta + coriolis.f(ai)) / depth_view(i) : 0);
  }
};

/** Initialize divergence field

  Divergence is defined on faces.
*/
template <typename Geo, typename DivergenceType>
struct DivergenceInit {
  using crd_view = typename Geo::crd_view_type;
  scalar_view_type divergence_view;
  crd_view lag_crd_view;
  DivergenceType div;

  /** Constructor. to be called from a Kokkos::parallel_for kernel launch.

    @param [in/out] sigma_view divergence view
    @param [in] lag_view face Lagrangian coordinate view
    @param [in] div divergence functor
  */
  DivergenceInit(scalar_view_type sigma_view, const crd_view lag_view,
                 const DivergenceType& div)
      : divergence_view(sigma_view), lag_crd_view(lag_view), div(div) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index i) const {
    const auto ai      = Kokkos::subview(lag_crd_view, i, Kokkos::ALL);
    divergence_view(i) = div(ai);
  }
};

template <typename SeedType, typename TopoType>
StaggeredSWE<SeedType, TopoType>::StaggeredSWE(
    const PolyMeshParameters<SeedType>& mesh_params, const Coriolis& coriolis,
    const TopoType& topo, const std::shared_ptr<spdlog::logger> logger_in)
    : relative_vorticity("relative_vorticity", mesh_params.nmaxfaces),
      potential_vorticity("potential_vorticity", mesh_params.nmaxfaces),
      divergence("divergence", mesh_params.nmaxfaces),
      mass("mass", mesh_params.nmaxfaces),
      double_dot_avg("double_dot_avg", mesh_params.nmaxfaces),
      grad_f_cross_u_avg("grad_f_cross_u_avg", mesh_params.nmaxfaces),
      depth("depth", mesh_params.nmaxfaces),
      bottom_height("bottom", mesh_params.nmaxfaces),
      surface_height("surface_height", mesh_params.nmaxfaces),
      surface_laplacian("surface_laplacian", mesh_params.nmaxfaces),
      velocity_avg("velocity_avg", mesh_params.nmaxfaces),
      velocity("velocity", mesh_params.nmaxverts),
      double_dot("double_dot", mesh_params.nmaxverts),
      grad_f_cross_u("grad_f_cross_u", mesh_params.nmaxverts),
      mesh(mesh_params),
      coriolis(coriolis),
      g(constants::GRAVITY),
      t(0),
      eps(0),
      topography(topo),
      logger(logger_in) {
  if (!logger_in) {
    logger = lpm_logger();
  }
}

template <typename SeedType, typename TopoType>
template <typename SurfaceType, typename VorticityType, typename DivergenceType>
void StaggeredSWE<SeedType, TopoType>::init_fields(
    const SurfaceType& sfc, const VorticityType& vorticity,
    const DivergenceType& div, const gmls::Params gmls_params) {
  Kokkos::parallel_for(
      mesh.n_faces_host(),
      SurfaceMassDepthInit<geo, TopoType, SurfaceType>(
          surface_height.view, bottom_height.view, depth.view, mass.view,
          mesh.faces.area, mesh.faces.lag_crds.view, topography, sfc));

  gmls_surface_laplacian(gmls_params);

  Kokkos::parallel_for(
      mesh.n_faces_host(),
      VorticityInit<geo, VorticityType>(
          relative_vorticity.view, potential_vorticity.view, depth.view,
          mesh.faces.lag_crds.view, vorticity, coriolis));

  Kokkos::parallel_for(mesh.n_faces_host(),
                       DivergenceInit<geo, DivergenceType>(
                           divergence.view, mesh.faces.lag_crds.view, div));

  init_direct_sums();

  Kokkos::parallel_for(
      mesh.n_faces_host(),
      VertexToFaceAverages<SeedType>(
          mesh.faces.phys_crds.view, double_dot_avg.view,
          grad_f_cross_u_avg.view, velocity_avg.view,
          mesh.vertices.lag_crds.view, double_dot.view, grad_f_cross_u.view,
          velocity.view, mesh.faces.verts));
}

template <typename SeedType, typename TopoType>
void StaggeredSWE<SeedType, TopoType>::init_direct_sums(
    const bool do_velocity) {
  static_assert(std::is_same<geo, SphereGeometry>::value,
                "StaggeredSWE only implemented for spherical problems so far.");

  const auto vertex_policy =
      Kokkos::TeamPolicy<>(mesh.n_vertices_host(), Kokkos::AUTO());

  Kokkos::parallel_for(
      vertex_policy,
      SphereVertexSums(velocity.view, double_dot.view,
                       mesh.vertices.lag_crds.view, mesh.faces.lag_crds.view,
                       relative_vorticity.view, divergence.view,
                       mesh.faces.area, mesh.faces.mask, eps,
                       mesh.n_faces_host(), do_velocity));
  Kokkos::parallel_for(mesh.n_vertices_host(),
                       GradFCrossU<typename SeedType::geo>(
                           grad_f_cross_u.view, mesh.vertices.lag_crds.view,
                           velocity.view, coriolis));

  const Index nnan_vel = velocity.nan_count(mesh.n_vertices_host());
  const Index nnan_dd  = double_dot.nan_count(mesh.n_vertices_host());
  if (nnan_vel > 0) {
    logger->error("velocity contains {} nans", nnan_vel);
  }
  if (nnan_dd > 0) {
    logger->error("double_dot contains {} nans", nnan_dd);
  }
}

template <typename SeedType, typename TopoType>
std::string StaggeredSWE<SeedType, TopoType>::info_string(
    const int tab_level, const bool verbose) const {
  std::ostringstream ss;
  ss << "StaggeredSWE<" << SeedType::id_string() << "," << topography.name()
     << "> info:\n";
  ss << mesh.info_string("StaggeredSWE", tab_level + 1);
  if (verbose) {
    ss << relative_vorticity.info_string();
    ss << potential_vorticity.info_string();
    ss << divergence.info_string();
    ss << mass.info_string();
    ss << double_dot_avg.info_string();
    ss << depth.info_string();
    ss << bottom_height.info_string();
    ss << surface_height.info_string();
    ss << surface_laplacian.info_string();
    for (const auto& t : tracers) {
      ss << t.second.info_string();
    }
    ss << velocity.info_string();
    ss << double_dot.info_string();
  }
  return ss.str();
}

template <typename SeedType, typename TopoType>
void StaggeredSWE<SeedType, TopoType>::gmls_surface_laplacian(
    const gmls::Params& gmls_params) {
  crd_view face_leaf_crds("face_leaf_crds", mesh.faces.n_leaves_host());
  mesh.get_leaf_face_crds(face_leaf_crds);
  const auto h_face_leaf_crds = Kokkos::create_mirror_view(face_leaf_crds);
  Kokkos::deep_copy(h_face_leaf_crds, face_leaf_crds);

  const auto neighbors = gmls::Neighborhoods(h_face_leaf_crds, gmls_params);

  const auto gmls_ops = std::vector<Compadre::TargetOperation>(
      {Compadre::LaplacianOfScalarPointEvaluation});

  const auto sfc_vals = mesh.faces.leaf_field_vals(surface_height);

  if constexpr (std::is_same<geo, SphereGeometry>::value) {
    auto sfc_gmls = gmls::sphere_scalar_gmls(face_leaf_crds, face_leaf_crds,
                                             neighbors, gmls_params, gmls_ops);
    auto eval     = Compadre::Evaluator(&sfc_gmls);

    const auto sfc_lap_gmls =
        eval.applyAlphasToDataAllComponentsAllTargetSites<Real*, DevMemory>(
            sfc_vals, Compadre::LaplacianOfScalarPointEvaluation,
            Compadre::PointSample);

    Kokkos::parallel_for(mesh.n_faces_host(),
                         ScatterFaceLeafData<scalar_view_type>(
                             surface_laplacian.view, sfc_lap_gmls,
                             mesh.faces.mask, mesh.faces.leaf_idx));
  } else {
    auto sfc_gmls = gmls::plane_scalar_gmls(face_leaf_crds, face_leaf_crds,
                                            neighbors, gmls_params, gmls_ops);
    auto eval     = Compadre::Evaluator(&sfc_gmls);

    const auto sfc_lap_gmls =
        eval.applyAlphasToDataAllComponentsAllTargetSites<Real*, DevMemory>(
            sfc_vals, Compadre::LaplacianOfScalarPointEvaluation,
            Compadre::PointSample);

    Kokkos::parallel_for(
        mesh.n_faces_host(),
        ScatterFaceLeafData(surface_laplacian.view, sfc_lap_gmls,
                            mesh.faces.mask, mesh.faces.leaf_idx));
  }
}

template <typename SeedType, typename TopoType>
void StaggeredSWE<SeedType, TopoType>::gmls_surface_laplacian(
    const crd_view& face_x, const gmls::Params& gmls_params) {
  crd_view face_leaf_crds("face_leaf_crds", mesh.faces.n_leaves_host());
  mesh.get_leaf_face_crds(face_leaf_crds, face_x);

  const auto h_face_leaf_crds = Kokkos::create_mirror_view(face_leaf_crds);
  Kokkos::deep_copy(h_face_leaf_crds, face_leaf_crds);

  const auto neighbors = gmls::Neighborhoods(h_face_leaf_crds, gmls_params);
  const auto gmls_ops  = std::vector<Compadre::TargetOperation>(
      {Compadre::LaplacianOfScalarPointEvaluation});
  scalar_view_type sfc_vals("leaf_surface_heights", mesh.faces.n_leaves_host());
  mesh.faces.leaf_field_vals(sfc_vals, surface_height);
  if constexpr (std::is_same<geo, SphereGeometry>::value) {
    auto sfc_gmls = gmls::sphere_scalar_gmls(face_leaf_crds, face_leaf_crds,
                                             neighbors, gmls_params, gmls_ops);
    auto eval     = Compadre::Evaluator(&sfc_gmls);

    const auto sfc_lap_gmls =
        eval.applyAlphasToDataAllComponentsAllTargetSites<Real*, DevMemory>(
            sfc_vals, Compadre::LaplacianOfScalarPointEvaluation,
            Compadre::PointSample);

    Kokkos::parallel_for(mesh.n_faces_host(),
                         ScatterFaceLeafData<scalar_view_type>(
                             surface_laplacian.view, sfc_lap_gmls,
                             mesh.faces.mask, mesh.faces.leaf_idx));
  } else {
    auto sfc_gmls = gmls::plane_scalar_gmls(face_leaf_crds, face_leaf_crds,
                                            neighbors, gmls_params, gmls_ops);
    auto eval     = Compadre::Evaluator(&sfc_gmls);

    const auto sfc_lap_gmls =
        eval.applyAlphasToDataAllComponentsAllTargetSites<Real*, DevMemory>(
            sfc_vals, Compadre::LaplacianOfScalarPointEvaluation,
            Compadre::PointSample);

    Kokkos::parallel_for(mesh.n_faces_host(),
                         ScatterFaceLeafData<scalar_view_type>(
                             surface_laplacian.view, sfc_lap_gmls,
                             mesh.faces.mask, mesh.faces.leaf_idx));
  }
}

template <typename SeedType, typename TopoType>
void StaggeredSWE<SeedType, TopoType>::allocate_scalar_tracer(
    const std::string& name) {
  tracers.emplace(
      name, ScalarField<FaceField>(name, relative_vorticity.view.extent(0)));
}

template <typename SeedType, typename TopoType>
void StaggeredSWE<SeedType, TopoType>::allocate_scalar_diag(
    const std::string& name) {
  diags.emplace(name, ScalarField<VertexField>(name, velocity.view.extent(0)));
}

template <typename SeedType, typename TopoType>
void StaggeredSWE<SeedType, TopoType>::update_host() {
  relative_vorticity.update_host();
  potential_vorticity.update_host();
  divergence.update_host();
  mass.update_host();
  double_dot_avg.update_host();
  depth.update_host();
  bottom_height.update_host();
  surface_height.update_host();
  surface_laplacian.update_host();
  for (const auto& t : tracers) {
    t.second.update_host();
  }
  velocity.update_host();
  double_dot.update_host();
  mesh.update_host();
}

template <typename SeedType, typename TopoType>
void StaggeredSWE<SeedType, TopoType>::update_device() {
  relative_vorticity.update_device();
  potential_vorticity.update_device();
  divergence.update_device();
  mass.update_device();
  double_dot_avg.update_device();
  depth.update_device();
  bottom_height.update_device();
  surface_height.update_device();
  surface_laplacian.update_device();
  for (const auto& t : tracers) {
    t.second.update_device();
  }
  velocity.update_device();
  double_dot.update_device();
  mesh.update_device();
}

template <typename SeedType, typename TopoType>
template <typename SolverType>
void StaggeredSWE<SeedType, TopoType>::advance_timestep(SolverType& solver) {
  solver.advance_timestep_impl();
  t = solver.t_idx * solver.dt;
}

#ifdef LPM_USE_VTK
template <typename SeedType, typename TopoType>
VtkPolymeshInterface<SeedType> vtk_mesh_interface(
    const StaggeredSWE<SeedType, TopoType>& swe) {
  VtkPolymeshInterface<SeedType> vtk(swe.mesh);
  vtk.add_scalar_cell_data(swe.relative_vorticity.view);
  vtk.add_scalar_cell_data(swe.potential_vorticity.view);
  vtk.add_scalar_cell_data(swe.divergence.view);
  vtk.add_scalar_cell_data(swe.mass.view);
  vtk.add_scalar_cell_data(swe.double_dot_avg.view);
  vtk.add_scalar_cell_data(swe.grad_f_cross_u_avg.view);
  vtk.add_scalar_cell_data(swe.bottom_height.view);
  vtk.add_scalar_cell_data(swe.surface_height.view);
  vtk.add_scalar_cell_data(swe.surface_laplacian.view);
  vtk.add_scalar_cell_data(swe.depth.view);
  vtk.add_vector_cell_data(swe.velocity_avg.view);
  for (const auto& t : swe.tracers) {
    vtk.add_scalar_cell_data(t.second.view);
  }
  for (const auto& d : swe.diags) {
    vtk.add_scalar_point_data(d.second.view);
  }
  vtk.add_vector_point_data(swe.velocity.view);
  vtk.add_scalar_point_data(swe.double_dot.view);
  vtk.add_scalar_point_data(swe.grad_f_cross_u.view);
  return vtk;
}
#endif

}  // namespace Lpm

#endif
