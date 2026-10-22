#ifndef LPM_SWE_IMPL_HPP
#define LPM_SWE_IMPL_HPP

#include "LpmConfig.h"
#include "lpm_field_impl.hpp"
#include "lpm_kokkos_defs.hpp"
#include "lpm_swe.hpp"
#include "lpm_swe_kernels.hpp"
#include "mesh/lpm_gather_mesh_data_impl.hpp"
#include "mesh/lpm_scatter_mesh_data_impl.hpp"

namespace Lpm {

template <typename InitialCondition>
struct SWEInitializeProblem {
  using Coriolis =
      typename std::conditional<std::is_same<typename InitialCondition::geo, PlaneGeometry>::value,
                                CoriolisBetaPlane, CoriolisSphere>::type;

  view_2d<Real> lag_crds;
  scalar_view_type rel_vort;
  scalar_view_type pot_vort;
  scalar_view_type divergence;
  scalar_view_type surface_height;
  scalar_view_type depth;
  view_2d<Real> velocity;
  InitialCondition initial;
  Coriolis coriolis;


  SWEInitializeProblem(view_2d<Real> lx, scalar_view_type zeta,
                       scalar_view_type Q, scalar_view_type delta,
                       scalar_view_type s, scalar_view_type h, view_2d<Real> u,
                       const InitialCondition ic,
                       const Coriolis& coriolis)
      : lag_crds(lx),
        rel_vort(zeta),
        pot_vort(Q),
        divergence(delta),
        surface_height(s),
        depth(h),
        velocity(u),
        initial(ic),
        coriolis(coriolis) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index i) const {
    const auto mcrd   = Kokkos::subview(lag_crds, i, Kokkos::ALL);
    const auto mvel   = Kokkos::subview(velocity, i, Kokkos::ALL);
    const Real zeta   = initial.vorticity(mcrd);
    divergence(i)     = initial.divergence(mcrd);
    const Real s      = initial.surface_height(mcrd);
    surface_height(i) = s;
    const Real h      = s - initial.bottom_height(mcrd);
    depth(i) = h;
    rel_vort(i) = zeta;
    pot_vort(i) = (zeta + coriolis.f(mcrd)) / h;
    initial.velocity(mvel, mcrd);
  }
};

template <typename SeedType>
SWE<SeedType>::SWE(const PolyMeshParameters<SeedType>& params,
                   const Coriolis& coriolis)
    : rel_vort_passive("relative_vorticity", params.nmaxverts),
      rel_vort_active("relative_vorticity", params.nmaxfaces),
      pot_vort_passive("potential_vorticity", params.nmaxverts),
      pot_vort_active("potential_vorticity", params.nmaxfaces),
      div_passive("divergence", params.nmaxverts),
      div_active("divergence", params.nmaxfaces),
      surf_passive("surface_height", params.nmaxverts),
      surf_active("surface_height", params.nmaxfaces),
      surf_lap_passive("surface_laplacian", params.nmaxverts),
      surf_lap_active("surface_laplaican", params.nmaxfaces),
      bottom_passive("bottom_height", params.nmaxverts),
      bottom_active("bottom_height", params.nmaxfaces),
      depth_passive("depth", params.nmaxverts),
      depth_active("depth", params.nmaxfaces),
      stream_fn_passive("stream_function", params.nmaxverts),
      stream_fn_active("stream_function", params.nmaxfaces),
      potential_passive("potential", params.nmaxverts),
      potential_active("potential", params.nmaxfaces),
      velocity_passive("velocity", params.nmaxverts),
      velocity_active("velocity", params.nmaxfaces),
      double_dot_passive("double_dot", params.nmaxverts),
      double_dot_active("double_dot", params.nmaxfaces),
      du1dx1_passive("du1dx1", params.nmaxverts),
      du1dx1_active("du1dx1", params.nmaxfaces),
      du1dx2_passive("du1dx2", params.nmaxverts),
      du1dx2_active("du1dx2", params.nmaxfaces),
      du2dx1_passive("du2dx1", params.nmaxverts),
      du2dx1_active("du2dx1", params.nmaxfaces),
      du2dx2_passive("du2dx2", params.nmaxverts),
      du2dx2_active("du2dx2", params.nmaxfaces),
      mass_active("mass", params.nmaxfaces),
      mesh(params),
      g(1),
      t(0),
      coriolis(coriolis) {
          passive_gmls_input_scalar_fields.emplace("surface_height", surf_passive);
          active_gmls_input_scalar_fields.emplace("surface_height", surf_active);
          passive_gmls_output_scalar_fields.emplace("surface_laplacian", surf_lap_passive);
          active_gmls_output_scalar_fields.emplace("surface_laplacian", surf_lap_active);

          passive_gmls_input_vector_fields.emplace("velocity", velocity_passive);
          active_gmls_input_vector_fields.emplace("velocity", velocity_active);

          scalar_gmls_ops = {Compadre::ScalarPointEvaluation,
              Compadre::GradientOfScalarPointEvaluation,
              Compadre::LaplacianOfScalarPointEvaluation};
          vector_gmls_ops = {Compadre::VectorPointEvaluation,
              Compadre::GradientOfVectorPointEvaluation};
      }

template <typename SeedType>
void SWE<SeedType>::set_kernel_parameters(const Real vel_eps,
                                          const Real pse_eps) {
  LPM_ASSERT(vel_eps >= 0);
  if constexpr (std::is_same<typename SeedType::geo, PlaneGeometry>::value) {
    LPM_ASSERT(pse_eps > 0);
    this->eps     = vel_eps;
    this->pse_eps = pse_eps;
  }
}

template <typename SeedType>
void SWE<SeedType>::update_host() {
  rel_vort_passive.update_host();
  rel_vort_active.update_host();
  pot_vort_passive.update_host();
  pot_vort_active.update_host();
  div_passive.update_host();
  div_active.update_host();
  surf_passive.update_host();
  surf_active.update_host();
  surf_lap_passive.update_host();
  surf_lap_active.update_host();
  bottom_passive.update_host();
  bottom_active.update_host();
  depth_passive.update_host();
  depth_active.update_host();
  velocity_passive.update_host();
  velocity_active.update_host();
  double_dot_passive.update_host();
  double_dot_active.update_host();
  stream_fn_passive.update_host();
  stream_fn_active.update_host();
  potential_passive.update_host();
  potential_active.update_host();
  du1dx1_passive.update_host();
  du1dx1_active.update_host();
  du1dx2_passive.update_host();
  du1dx2_active.update_host();
  du2dx1_passive.update_host();
  du2dx1_active.update_host();
  du2dx2_passive.update_host();
  du2dx2_active.update_host();
  mass_active.update_host();
  mesh.update_host();
}

template <typename SeedType>
void SWE<SeedType>::update_device() {
  rel_vort_passive.update_device();
  rel_vort_active.update_device();
  pot_vort_passive.update_device();
  pot_vort_active.update_device();
  div_passive.update_device();
  div_active.update_device();
  surf_passive.update_device();
  surf_active.update_device();
  surf_lap_passive.update_device();
  surf_lap_active.update_device();
  bottom_passive.update_device();
  bottom_active.update_device();
  depth_passive.update_device();
  depth_active.update_device();
  stream_fn_passive.update_device();
  stream_fn_active.update_device();
  potential_passive.update_device();
  potential_active.update_device();
  velocity_passive.update_device();
  velocity_active.update_device();
  double_dot_passive.update_device();
  double_dot_active.update_device();
  du1dx1_passive.update_device();
  du1dx1_active.update_device();
  du1dx2_passive.update_device();
  du1dx2_active.update_device();
  du2dx1_passive.update_device();
  du2dx1_active.update_device();
  du2dx2_passive.update_device();
  du2dx2_active.update_device();
  mass_active.update_device();
  mesh.update_device();
}

template <typename SeedType>
template <typename InitialCondition>
void SWE<SeedType>::init_swe_problem(const InitialCondition& ic) {
  Kokkos::parallel_for(
      "init_swe_problem (passive)", mesh.n_vertices_host(),
      SWEInitializeProblem(mesh.vertices.lag_crds.view, rel_vort_passive.view,
                           pot_vort_passive.view, div_passive.view,
                           surf_passive.view, depth_passive.view,
                           velocity_passive.view, ic, coriolis));

  Kokkos::parallel_for(
      "init_swe_problem (active)", mesh.n_faces_host(),
      SWEInitializeProblem(mesh.faces.lag_crds.view, rel_vort_active.view,
                           pot_vort_active.view, div_active.view,
                           surf_active.view, depth_active.view,
                           velocity_active.view, ic, coriolis));
  auto mass = this->mass_active.view;
  auto depth = this->depth_active.view;
  auto area = this->mesh.faces.area;
  auto mask = this->mesh.faces.mask;

  Kokkos::parallel_for("ComputeMass", this->mesh.n_faces_host(),
    KOKKOS_LAMBDA(const int i) {
      mass(i) = (mask(i) ? 0.0 : depth(i) * area(i));
    }
  );
}

template <typename SeedType>
template <typename TopoType>
void SWE<SeedType>::set_bottom_topography(const TopoType& topo) {
  auto crds      = mesh.vertices.phys_crds.view;
  auto topo_view = bottom_passive.view;
  Kokkos::parallel_for(
      mesh.n_vertices_host(), KOKKOS_LAMBDA(const Index i) {
        const auto mcrd = Kokkos::subview(crds, i, Kokkos::ALL);
        topo_view(i)    = topo(mcrd);
      });

  crds      = mesh.faces.phys_crds.view;
  topo_view = bottom_active.view;
  Kokkos::parallel_for(
      mesh.n_faces_host(), KOKKOS_LAMBDA(const Index i) {
        const auto mcrd = Kokkos::subview(crds, i, Kokkos::ALL);
        topo_view(i)    = topo(mcrd);
      });
}

template <typename SeedType>
template <typename BottomType, typename SurfaceType>
void SWE<SeedType>::init_surface(const BottomType& topo,
                                 const SurfaceType& sfc) {
  auto crds       = mesh.vertices.phys_crds.view;
  auto topo_view  = bottom_passive.view;
  auto sfc_view   = surf_passive.view;
  auto depth_view = depth_passive.view;
  Kokkos::parallel_for(
      mesh.n_vertices_host(), KOKKOS_LAMBDA(const Index i) {
        const auto mcrd = Kokkos::subview(crds, i, Kokkos::ALL);
        const Real b    = topo(mcrd);
        const Real s    = sfc(mcrd);
        topo_view(i)    = b;
        sfc_view(i)     = s;
        depth_view(i)   = s - b;
      });

  crds           = mesh.faces.phys_crds.view;
  topo_view      = bottom_active.view;
  sfc_view       = surf_active.view;
  depth_view     = depth_active.view;
  auto mass_view = mass_active.view;
  auto area_view = mesh.faces.area;
  Kokkos::parallel_for(
      mesh.n_faces_host(), KOKKOS_LAMBDA(const Index i) {
        const auto mcrd = Kokkos::subview(crds, i, Kokkos::ALL);
        const Real b    = topo(mcrd);
        const Real s    = sfc(mcrd);
        topo_view(i)    = b;
        sfc_view(i)     = s;
        depth_view(i)   = s - b;
        mass_view(i)    = (s - b) * area_view(i);
      });
}

template <typename SeedType>
template <typename VorticityType>
void SWE<SeedType>::init_vorticity(const VorticityType& vorticity,
                                   const bool depth_set) {
  auto crds         = mesh.vertices.phys_crds.view;
  auto relvort_view = rel_vort_passive.view;
  auto potvort_view = pot_vort_passive.view;
  auto depth_view   = depth_passive.view;
  Kokkos::parallel_for(
      "SWE::init_vorticity (passive)", mesh.n_vertices_host(),
      KOKKOS_LAMBDA(const Index i) {
        const auto mcrd = Kokkos::subview(crds, i, Kokkos::ALL);
        const Real zeta = vorticity(mcrd);
        relvort_view(i) = zeta;
        if (depth_set) {
          potvort_view(i) = (zeta + coriolis.f(mcrd)) / depth_view(i);
        }
      });

  crds         = mesh.faces.phys_crds.view;
  relvort_view = rel_vort_active.view;
  potvort_view = pot_vort_active.view;
  depth_view   = depth_active.view;
  Kokkos::parallel_for(
      "SWE::init_vorticity (active)", mesh.n_faces_host(),
      KOKKOS_LAMBDA(const Index i) {
        const auto mcrd = Kokkos::subview(crds, i, Kokkos::ALL);
        const Real zeta = vorticity(mcrd);
        relvort_view(i) = zeta;
        if (depth_set) {
          potvort_view(i) = (zeta + coriolis.f(mcrd)) / depth_view(i);
        }
      });
}

template <typename SeedType>
template <typename DivergenceType>
void SWE<SeedType>::init_divergence(const DivergenceType& divergence) {
  auto crds       = mesh.vertices.lag_crds.view;
  auto delta_view = div_passive.view;
  Kokkos::parallel_for(
      "SWE::init_divergence (passive)", mesh.n_vertices_host(),
      KOKKOS_LAMBDA(const Index i) {
        const auto xi = Kokkos::subview(crds, i, Kokkos::ALL);
        delta_view(i) = divergence(xi);
      });

  crds       = mesh.faces.lag_crds.view;
  delta_view = div_active.view;
  Kokkos::parallel_for(
      "SWE::init_divergence (active)", mesh.n_faces_host(),
      KOKKOS_LAMBDA(const Index i) {
        const auto xi = Kokkos::subview(crds, i, Kokkos::ALL);
        delta_view(i) = divergence(xi);
      });
}

template <typename SeedType>
template <typename SolverType>
void SWE<SeedType>::advance_timestep(SolverType& solver) {
  solver.advance_timestep_impl();
  t = solver.t_idx * solver.dt;
}

template <typename SeedType>
void SWE<SeedType>::compute_velocity_direct_sum() {
  Kokkos::TeamPolicy<> vertex_policy(mesh.n_vertices_host(), Kokkos::AUTO());
  Kokkos::TeamPolicy<> face_policy(mesh.n_faces_host(), Kokkos::AUTO());

  Kokkos::parallel_for(vertex_policy,
    VelocityVertexSums<typename SeedType::geo>(
      velocity_passive.view,
      mesh.vertices.phys_crds.view,
      mesh.faces.phys_crds.view,
      rel_vort_active.view,
      div_active.view,
      mesh.faces.area,
      mesh.faces.mask,
      mesh.n_faces_host()));


  Kokkos::parallel_for(face_policy,
    VelocityFaceSums<typename SeedType::geo>(
      velocity_active.view,
      mesh.faces.phys_crds.view,
      rel_vort_active.view,
      div_active.view,
      mesh.faces.area,
      mesh.faces.mask,
      mesh.n_faces_host()));

}

template <typename SeedType>
void SWE<SeedType>::initialize_gmls(const int gmls_order, const Real gmls_radius_multiplier) {
  gmls_params = gmls::Params(gmls_order);
  gmls_params.eps_multiplier = gmls_radius_multiplier;
  gmls_params.manifold_order = 2;

  gathered_mesh = std::make_unique<GatherMeshData<SeedType>>(mesh);
  scatter_mesh = std::make_unique<ScatterMeshData<SeedType>>(*gathered_mesh, mesh);
  gathered_mesh->init_scalar_fields(passive_gmls_input_scalar_fields, active_gmls_input_scalar_fields);
  gathered_surface_laplacian = scalar_view_type("surface_laplacian", gathered_mesh->n());
  gathered_mesh->scalar_fields.emplace("surface_laplacian", gathered_surface_laplacian);

  mesh.vertices.phys_crds.update_host();
  mesh.faces.phys_crds.update_host();

  gmls_neighborhoods = gmls::Neighborhoods(gathered_mesh->h_phys_crds,
    gathered_mesh->h_phys_crds, gmls_params);

  const auto scalar_reconstruction_space = Compadre::ReconstructionSpace::ScalarTaylorPolynomial;
  const auto vector_reconstruction_space = Compadre::ReconstructionSpace::VectorTaylorPolynomial;
  const auto scalar_sampling_functional = Compadre::PointSample;
  const auto vector_sampling_functional = (std::is_same<typename SeedType::geo, SphereGeometry>::value ?
    Compadre::ManifoldVectorPointSample : Compadre::VectorPointSample);
  const auto problem_type = (std::is_same<typename SeedType::geo, SphereGeometry>::value ?
    Compadre::ProblemType::MANIFOLD : Compadre::ProblemType::STANDARD);
  const auto scalar_data_functional = scalar_sampling_functional;
  const auto vector_data_functional = vector_sampling_functional;
  const auto solver_type = Compadre::DenseSolverType::QR;
  const auto constraint_type = Compadre::ConstraintType::NO_CONSTRAINT;
  const auto weight_type = Compadre::WeightingFunctionType::Power;

  if (!scalar_gmls) {
    scalar_gmls = std::make_unique<Compadre::GMLS>(
      scalar_reconstruction_space,
      scalar_sampling_functional,
      scalar_data_functional,
      gmls_params.samples_order,
      SeedType::geo::ndim,
      solver_type,
      problem_type,
      constraint_type,
      gmls_params.manifold_order);

    vector_gmls = std::make_unique<Compadre::GMLS>(
      vector_reconstruction_space,
      vector_sampling_functional,
      vector_data_functional,
      gmls_params.samples_order,
      SeedType::geo::ndim,
      solver_type,
      problem_type,
      constraint_type,
      gmls_params.manifold_order);
  }
  else {
    scalar_gmls.reset(new Compadre::GMLS(
      scalar_reconstruction_space,
      scalar_sampling_functional,
      scalar_data_functional,
      gmls_params.samples_order,
      SeedType::geo::ndim,
      solver_type,
      problem_type,
      constraint_type,
      gmls_params.manifold_order));

    vector_gmls.reset(new Compadre::GMLS(
      vector_reconstruction_space,
      vector_sampling_functional,
      vector_data_functional,
      gmls_params.samples_order,
      SeedType::geo::ndim,
      solver_type,
      problem_type,
      constraint_type,
      gmls_params.manifold_order));
  }

  scalar_gmls->setProblemData(gmls_neighborhoods.neighbor_lists,
    gathered_mesh->phys_crds, gathered_mesh->phys_crds, gmls_neighborhoods.neighborhood_radii);
  vector_gmls->setProblemData(gmls_neighborhoods.neighbor_lists,
    gathered_mesh->phys_crds, gathered_mesh->phys_crds, gmls_neighborhoods.neighborhood_radii);

  scalar_gmls->addTargets(scalar_gmls_ops);
//   vector_gmls->addTargets(vector_gmls_ops);

  scalar_gmls->setWeightingType(weight_type);
//   vector_gmls->setWeightingType(weight_type);

  scalar_gmls->setWeightingParameter(gmls_params.samples_weight_pwr);
//   vector_gmls->setWeightingParameter(gmls_params.samples_weight_pwr);

  if constexpr (std::is_same<typename SeedType::geo, SphereGeometry>::value) {
    constexpr bool use_to_orient = true;
    scalar_gmls->setReferenceOutwardNormalDirection(gathered_mesh->phys_crds, use_to_orient);
    vector_gmls->setReferenceOutwardNormalDirection(gathered_mesh->phys_crds, use_to_orient);

    scalar_gmls->setCurvatureWeightingType(weight_type);
    vector_gmls->setCurvatureWeightingType(weight_type);

    scalar_gmls->setCurvatureWeightingParameter(gmls_params.manifold_weight_pwr);
    vector_gmls->setCurvatureWeightingParameter(gmls_params.manifold_weight_pwr);
  }

  scalar_gmls->generateAlphas();
//   vector_gmls->generateAlphas();

  gathered_mesh->gather_scalar_fields(passive_gmls_input_scalar_fields, active_gmls_input_scalar_fields);
  gathered_mesh->gather_vector_fields(passive_gmls_input_vector_fields, active_gmls_input_vector_fields);
}

template <typename SeedType>
void SWE<SeedType>::init_direct_sums(const bool do_velocity) {
  Kokkos::TeamPolicy<> vertex_policy(mesh.n_vertices_host(), Kokkos::AUTO());
  Kokkos::TeamPolicy<> face_policy(mesh.n_faces_host(), Kokkos::AUTO());

  if constexpr (std::is_same<typename SeedType::geo, PlaneGeometry>::value) {
    Kokkos::parallel_for(
        "initialize direct sums, passive", vertex_policy,
        PlanarSWEVertexSums(
            velocity_passive.view, double_dot_passive.view, du1dx1_passive.view,
            du1dx2_passive.view, du2dx1_passive.view, du2dx2_passive.view,
            surf_lap_passive.view, stream_fn_passive.view,
            potential_passive.view, mesh.vertices.phys_crds.view,
            surf_passive.view, mesh.faces.phys_crds.view, rel_vort_active.view,
            div_active.view, mesh.faces.area, mesh.faces.mask, surf_active.view,
            eps, pse_eps, mesh.n_faces_host(), do_velocity));

    Kokkos::parallel_for(
        "initialize direct sums, active", face_policy,
        PlanarSWEFaceSums(
            velocity_active.view, double_dot_active.view, du1dx1_active.view,
            du1dx2_active.view, du2dx1_active.view, du2dx2_active.view,
            surf_lap_active.view, stream_fn_active.view, potential_active.view,
            mesh.faces.phys_crds.view, rel_vort_active.view, div_active.view,
            mesh.faces.area, mesh.faces.mask, surf_active.view, eps, pse_eps,
            mesh.n_faces_host(), do_velocity));
  } else {
    if constexpr (std::is_same<typename SeedType::geo, SphereGeometry>::value) {
      Kokkos::parallel_for(
          "init direct sums, passive", vertex_policy,
          SphereVertexSums(velocity_passive.view, double_dot_passive.view,
                           mesh.vertices.lag_crds.view,
                           mesh.faces.lag_crds.view, rel_vort_active.view,
                           div_active.view, mesh.faces.area, mesh.faces.mask,
                           eps, mesh.n_faces_host(), do_velocity));

      Kokkos::parallel_for(
          "init direct sums, active", face_policy,
          SphereFaceSums(velocity_active.view, double_dot_active.view,
                         mesh.faces.lag_crds.view, rel_vort_active.view,
                         div_active.view, mesh.faces.area, mesh.faces.mask, eps,
                         mesh.n_faces_host(), do_velocity));
    }
  }
}

template <typename SeedType>
std::string SWE<SeedType>::info_string(const int tab_level,
                                       const bool verbose) const {
  std::ostringstream ss;
  std::string tabstr = indent_string(tab_level);
  ss << tabstr << "SWE info:\n";
  tabstr += "\t";
  ss << tabstr << "g = " << g << "\n"
     << tabstr << "t = " << t << "\n"
     << tabstr << "eps = " << eps << "\n"
     << tabstr << "pse_eps = " << pse_eps << "\n";
  ss << tabstr << mesh.info_string("swe mesh", tab_level, verbose);
  if (verbose) {
    ss << "passive fields:\n";
    ss << tabstr << rel_vort_passive.info_string() << tabstr
       << pot_vort_passive.info_string() << tabstr << div_passive.info_string()
       << tabstr << surf_passive.info_string() << tabstr
       << surf_lap_passive.info_string() << tabstr
       << depth_passive.info_string() << tabstr
       << double_dot_passive.info_string() << tabstr
       << velocity_passive.info_string();
    ss << "active fields:\n";
    ss << tabstr << rel_vort_active.info_string() << tabstr
       << pot_vort_active.info_string() << tabstr << div_active.info_string()
       << tabstr << surf_active.info_string() << tabstr
       << surf_lap_active.info_string() << tabstr << depth_active.info_string()
       << tabstr << double_dot_active.info_string() << tabstr
       << mass_active.info_string() << tabstr << velocity_active.info_string();
  }
  return ss.str();
}

template <typename SeedType>
void SWE<SeedType>::allocate_scalar_tracer(const std::string& name) {
  tracer_passive.emplace(
      name,
      ScalarField<VertexField>(name, this->rel_vort_passive.view.extent(0)));
  tracer_active.emplace(
      name, ScalarField<FaceField>(name, this->rel_vort_active.view.extent(0)));
}

template <typename SeedType>
void SWE<SeedType>::compute_surface_laplacian_gmls() {

  auto logger = lpm_logger();

  Compadre::Evaluator eval(scalar_gmls.get());
  const auto surf_vals = gathered_mesh->scalar_fields.at("surface_height");
  auto surf_lap = gathered_mesh->scalar_fields.at("surface_laplacian");

  logger->debug("SWE::compute_surface_laplacian_gmls: gathered input/output scalar fields.");

  scalar_view_type null_view;
  eval.applyAlphasToDataAllComponentsAllTargetSites<scalar_view_type, scalar_view_type>(
      surf_lap, null_view,
      surf_vals, Compadre::LaplacianOfScalarPointEvaluation, Compadre::PointSample);

  logger->debug("SWE::compute_surface_laplacian_gmls: GMLS eval complete");

  scatter_mesh->scatter_fields(passive_gmls_output_scalar_fields, active_gmls_output_scalar_fields);
}

template <typename SeedType>
VtkPolymeshInterface<SeedType> vtk_mesh_interface(const SWE<SeedType>& swe) {
//   if constexpr (std::is_same<typename SeedType::geo, PlaneGeometry>::value) {
//     VtkPolymeshInterface<SeedType> vtk(swe.mesh, swe.surf_passive.hview);
//   }
//   else {
    VtkPolymeshInterface<SeedType> vtk(swe.mesh);
//   }
  vtk.add_scalar_point_data(swe.rel_vort_passive.view);
  vtk.add_scalar_point_data(swe.pot_vort_passive.view);
  vtk.add_scalar_point_data(swe.div_passive.view);
  vtk.add_scalar_point_data(swe.surf_passive.view);
  vtk.add_scalar_point_data(swe.surf_lap_passive.view);
  vtk.add_scalar_point_data(swe.depth_passive.view);
  vtk.add_scalar_point_data(swe.double_dot_passive.view);
  vtk.add_scalar_point_data(swe.du1dx1_passive.view);
  vtk.add_scalar_point_data(swe.du1dx2_passive.view);
  vtk.add_scalar_point_data(swe.du2dx1_passive.view);
  vtk.add_scalar_point_data(swe.du2dx2_passive.view);
  vtk.add_scalar_point_data(swe.bottom_passive.view);
  vtk.add_scalar_point_data(swe.stream_fn_passive.view);
  vtk.add_scalar_point_data(swe.potential_passive.view);
  vtk.add_vector_point_data(swe.velocity_passive.view);
  vtk.add_scalar_cell_data(swe.rel_vort_active.view);
  vtk.add_scalar_cell_data(swe.pot_vort_active.view);
  vtk.add_scalar_cell_data(swe.div_active.view);
  vtk.add_scalar_cell_data(swe.surf_active.view);
  vtk.add_scalar_cell_data(swe.depth_active.view);
  vtk.add_scalar_cell_data(swe.surf_lap_active.view);
  vtk.add_scalar_cell_data(swe.double_dot_active.view);
  vtk.add_scalar_cell_data(swe.du1dx1_active.view);
  vtk.add_scalar_cell_data(swe.du1dx2_active.view);
  vtk.add_scalar_cell_data(swe.du2dx1_active.view);
  vtk.add_scalar_cell_data(swe.du2dx2_active.view);
  vtk.add_scalar_cell_data(swe.bottom_active.view);
  vtk.add_vector_cell_data(swe.velocity_active.view);
  vtk.add_scalar_cell_data(swe.mass_active.view);
  vtk.add_scalar_cell_data(swe.stream_fn_active.view);
  vtk.add_scalar_cell_data(swe.potential_active.view);
  for (const auto& tracer : swe.tracer_passive) {
    vtk.add_scalar_point_data(tracer.second.view, tracer.first);
    vtk.add_scalar_cell_data(swe.tracer_active.at(tracer.first).view,
                             tracer.first);
  }
  return vtk;
}


}  // namespace Lpm

#endif
