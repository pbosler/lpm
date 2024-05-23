#include "LpmConfig.h"
#include "lpm_comm.hpp"
#include "lpm_compadre.hpp"
#include "lpm_constants.hpp"
#include "lpm_error.hpp"
#include "lpm_error_impl.hpp"
#include "lpm_field.hpp"
#include "lpm_field_impl.hpp"
#include "lpm_geometry.hpp"
#include "lpm_incompressible2d.hpp"
#include "lpm_incompressible2d_impl.hpp"
#include "lpm_incompressible2d_rk2.hpp"
#include "lpm_incompressible2d_rk2_impl.hpp"
#include "lpm_input.hpp"
#include "lpm_logger.hpp"
#include "lpm_swe_kernels.hpp"
#include "lpm_tracer_gallery.hpp"
#include "lpm_vorticity_gallery.hpp"
#include "lpm_velocity_gallery.hpp"
#include "mesh/lpm_compadre_remesh.hpp"
#include "mesh/lpm_compadre_remesh_impl.hpp"
#include "util/lpm_floating_point.hpp"
#include "util/lpm_string_util.hpp"
#include "util/lpm_tuple.hpp"
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"

using namespace Lpm;

struct DDotReducer {
  using crd_view = typename SphereGeometry::crd_view_type;
  using value_type = Kokkos::Tuple<Real,9>;

  crd_view tgt_x;
  Index i;
  crd_view src_y;
  scalar_view_type src_vorticity;
  scalar_view_type src_area;
  mask_view_type src_mask;
  Real eps;
  bool collocated_src_tgt;

  DDotReducer(const crd_view tx, const Index tgt_idx,
    const crd_view sy, const scalar_view_type zeta, const scalar_view_type area,
    const mask_view_type mask, const Real eps, const bool colloc) :
    tgt_x(tx),
    i(tgt_idx),
    src_y(sy),
    src_vorticity(zeta),
    src_area(area),
    src_mask(mask),
    eps(eps),
    collocated_src_tgt(colloc) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index& j, value_type& r) const {
    if (!src_mask(j)) {
      if (!collocated_src_tgt or (i != j)) {
        const auto xcrd = Kokkos::subview(tgt_x, i, Kokkos::ALL);
        const auto ycrd = Kokkos::subview(src_y, j, Kokkos::ALL);
        const Real rot_str = - src_vorticity(j) * src_area(j);
        Real gkz[9];
        grad_kzeta(gkz, xcrd, ycrd, eps);
        for (int k=0; k<9; ++k) {
          r[k] += gkz[k]*rot_str;
        }
      }
    }
  }
};

struct DDotPassive {
  using crd_view = typename SphereGeometry::crd_view_type;

  scalar_view_type vert_ddot;
  crd_view vert_x;
  crd_view face_y;
  scalar_view_type face_zeta;
  scalar_view_type face_area;
  mask_view_type face_mask;
  Real eps;
  Index nfaces;

  DDotPassive(scalar_view_type dd, const crd_view vx,
    const crd_view fy, const scalar_view_type fz, const scalar_view_type fa,
    const mask_view_type fm, const Real eps, const Index nf):
    vert_ddot(dd),
    vert_x(vx),
    face_y(fy),
    face_zeta(fz),
    face_area(fa),
    face_mask(fm),
    eps(eps),
    nfaces(nf) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const member_type& thread_team) const {
    const Index i = thread_team.league_rank();
    Kokkos::Tuple<Real,9> sums;
    constexpr bool collocated = false;
    Kokkos::parallel_reduce(Kokkos::TeamVectorRange(thread_team, nfaces),
      DDotReducer(vert_x, i, face_y, face_zeta, face_area,
        face_mask, eps, collocated), sums);

    vert_ddot(i) = 0;
    for (int ii=0; ii<3; ++ii) {
      for (int jj=0; jj<3; ++jj) {
        vert_ddot(i) += sums[3*ii + jj] * sums[3*jj+ii];
      }
    }
  }

};

struct DDotActive {
  using crd_view = typename SphereGeometry::crd_view_type;

  scalar_view_type face_ddot;
  crd_view face_xy;
  scalar_view_type face_zeta;
  scalar_view_type face_area;
  mask_view_type face_mask;
  Real eps;
  Index nfaces;

  DDotActive(scalar_view_type dd, const crd_view fxy,
    const scalar_view_type fz, const scalar_view_type fa,
    const mask_view_type fm, const Real eps, const Index nf):
  face_ddot(dd),
  face_xy(fxy),
  face_zeta(fz),
  face_area(fa),
  face_mask(fm),
  eps(eps),
  nfaces(nf) {}


  KOKKOS_INLINE_FUNCTION
  void operator() (const member_type& thread_team) const {
    const Index i = thread_team.league_rank();
    Kokkos::Tuple<Real,9> sums;
    const bool colloc = FloatingPoint<Real>::zero(eps);
    Kokkos::parallel_reduce(Kokkos::TeamVectorRange(thread_team, nfaces),
      DDotReducer(face_xy, i, face_xy, face_zeta, face_area,
        face_mask, eps, colloc), sums);

    face_ddot(i) = 0;
    for (int ii=0; ii<3; ++ii) {
      for (int jj=0; jj<3; ++jj) {
        face_ddot(i) += sums[3*ii+jj]*sums[3*jj+ii];
      }
    }
  }
};

template <typename SeedType>
void do_velocity(Kokkos::View<Real*[3]> passive_exact, Kokkos::View<Real*[3]> active_exact,
  Kokkos::View<Real*[3]> passive_error, Kokkos::View<Real*[3]> active_error,
  const Incompressible2D<SeedType>& sphere) {

  RossbyWave54Velocity rh54_velocity;
  constexpr Real t_rh = 0; // TODO: fix this when rh54 velocity accounts for time

  const auto vx = sphere.mesh.vertices.phys_crds.view;
  const auto vvel = sphere.velocity_passive.view;
  Kokkos::parallel_for(sphere.mesh.n_vertices_host(),
    KOKKOS_LAMBDA (const Index i) {
      const auto xi = Kokkos::subview(vx, i, Kokkos::ALL);

      Kokkos::Tuple<Real,3> vel = rh54_velocity(xi, t_rh);
      for (int j=0; j<3; ++j) {
        passive_exact(i,j) = vel[j];
        passive_error(i,j) = vvel(i,j) - vel[j];
      }
    });

  const auto fx = sphere.mesh.faces.phys_crds.view;
  const auto fvel = sphere.velocity_active.view;
  Kokkos::parallel_for(sphere.mesh.n_faces_host(),
    KOKKOS_LAMBDA (const Index i) {
      const auto xi = Kokkos::subview(fx, i, Kokkos::ALL);

      Kokkos::Tuple<Real,3> vel = rh54_velocity(xi, t_rh);
      for (int j=0; j<3; ++j) {
        active_exact(i,j) = vel[j];
        active_error(i,j) = fvel(i,j) - vel[j];
      }
    });
}

template <typename PtType>
Real ddot_exact(const PtType& x) {
  const Real lon = SphereGeometry::longitude(x);
  const Real lat = SphereGeometry::latitude(x);

  return (pow(cos(lat),4)*(226 - 376*cos(2*lat) + 182*cos(4*lat) - 4*cos(2*(lat - 4*lon)) + 9*cos(4*(lat - 2*lon)) - 42*cos(8*lon) + 9*cos(4*(lat + 2*lon)) -
       4*cos(2*(lat + 4*lon))))/2.0;
}

template <typename SeedType>
void do_double_dot(Incompressible2D<SeedType>& sphere) {
  auto dd_passive = sphere.tracer_passive.at("double_dot").view;
  auto dd_passive_exact = sphere.tracer_passive.at("double_dot_exact").view;
  auto dd_passive_error = sphere.tracer_passive.at("double_dot_error").view;
  auto dd_active = sphere.tracer_active.at("double_dot").view;
  auto dd_active_exact = sphere.tracer_active.at("double_dot_exact").view;
  auto dd_active_error = sphere.tracer_active.at("double_dot_error").view;
  auto dd_gmls_passive = sphere.tracer_passive.at("double_dot_gmls").view;
  auto dd_gmls_active = sphere.tracer_active.at("double_dot_gmls").view;
  auto dd_gmls_err_view_passive = sphere.tracer_passive.at("double_dot_gmls_error").view;
  auto dd_gmls_err_view_active = sphere.tracer_active.at("double_dot_gmls_error").view;

  auto passive_policy = Kokkos::TeamPolicy<>(sphere.mesh.n_vertices_host(), Kokkos::AUTO());
  auto active_policy = Kokkos::TeamPolicy<>(sphere.mesh.n_faces_host(), Kokkos::AUTO());

  const auto passive_x = sphere.mesh.vertices.phys_crds.view;
  const auto active_x = sphere.mesh.faces.phys_crds.view;
  const auto active_zeta = sphere.rel_vort_active.view;
  const auto active_area = sphere.mesh.faces.area;
  const auto active_mask = sphere.mesh.faces.mask;

  Kokkos::parallel_for(passive_policy,
    DDotPassive(dd_passive, passive_x, active_x, active_zeta, active_area,
      active_mask, sphere.eps, sphere.mesh.n_faces_host()));
  Kokkos::parallel_for(active_policy,
    DDotActive(dd_active, active_x, active_zeta, active_area, active_mask,
      sphere.eps, sphere.mesh.n_faces_host()));

  std::map<std::string, ScalarField<VertexField>> vert_gmls_scalar_fields;
  std::map<std::string, ScalarField<FaceField>> face_gmls_scalar_fields;
  std::map<std::string, VectorField<typename SeedType::geo, VertexField>> vert_gmls_vec_fields;
  std::map<std::string, VectorField<typename SeedType::geo, FaceField>> face_gmls_vec_fields;
  vert_gmls_scalar_fields.emplace("double_dot_gmls", sphere.tracer_passive.at("double_dot_gmls"));
  face_gmls_scalar_fields.emplace("double_dot_gmls", sphere.tracer_active.at("double_dot_gmls"));
  vert_gmls_vec_fields.emplace("velocity", sphere.velocity_passive);
  face_gmls_vec_fields.emplace("velocity", sphere.velocity_active);

  GatherMeshData<SeedType> gather(sphere.mesh);
  gather.update_host();
  gather.init_scalar_fields(vert_gmls_scalar_fields, face_gmls_scalar_fields);
  gather.init_vector_fields(vert_gmls_vec_fields, face_gmls_vec_fields);
  gather.gather_vector_fields(vert_gmls_vec_fields, face_gmls_vec_fields);

  scalar_view_type u("u_gathered", gather.n());
  scalar_view_type v("v_gathered", gather.n());
  scalar_view_type w("w_gathered", gather.n());
  auto vel_view = gather.vector_fields.at("velocity");
  Kokkos::parallel_for(gather.n(),
    KOKKOS_LAMBDA (const Index i) {
      u(i) = vel_view(i,0);
      v(i) = vel_view(i,1);
      w(i) = vel_view(i,2);
    });

  constexpr int gmls_order = 4;
  const gmls::Params gmls_params(gmls_order);
  gmls::Neighborhoods neighbors(gather.h_phys_crds, gmls_params);

  const Compadre::ReconstructionSpace scalar_reconstruction_space =
      Compadre::ReconstructionSpace::VectorOfScalarClonesTaylorPolynomial; // Compadre::ReconstructionSpace::ScalarTaylorPolynomial;
  const Compadre::ReconstructionSpace vector_reconstruction_space =
      Compadre::ReconstructionSpace::VectorTaylorPolynomial;
  const Compadre::ProblemType problem_type = Compadre::ProblemType::MANIFOLD;
  const Compadre::DenseSolverType solver_type = Compadre::DenseSolverType::QR;
  const Compadre::SamplingFunctional scalar_sampling_functional = Compadre::PointSample;
  const Compadre::SamplingFunctional vector_sampling_functional = Compadre::ManifoldVectorPointSample;
  const Compadre::WeightingFunctionType weight_type = Compadre::WeightingFunctionType::Power;
  const Compadre::ConstraintType constraint_type = Compadre::ConstraintType::NO_CONSTRAINT;
  const auto scalar_data_functional = scalar_sampling_functional;
  const auto vector_data_functional = vector_sampling_functional;

  Compadre::GMLS dd_gmls(
    scalar_reconstruction_space,
    scalar_sampling_functional,
    scalar_data_functional,
    gmls_params.samples_order,
    SeedType::geo::ndim,
    solver_type,
    problem_type,
    constraint_type,
    gmls_params.manifold_order);

  std::vector<Compadre::TargetOperation> gmls_ops({Compadre::GradientOfScalarPointEvaluation});
  dd_gmls.setProblemData(
    neighbors.neighbor_lists,
    gather.phys_crds,
    gather.phys_crds,
    neighbors.neighborhood_radii);
  dd_gmls.addTargets(gmls_ops);
  dd_gmls.setWeightingType(weight_type);
  dd_gmls.setWeightingParameter(gmls_params.samples_weight_pwr);
  constexpr bool use_to_orient = true;
  dd_gmls.setReferenceOutwardNormalDirection(gather.phys_crds, use_to_orient);
  dd_gmls.setCurvatureWeightingType(weight_type);
  dd_gmls.setCurvatureWeightingParameter(gmls_params.manifold_weight_pwr);
  dd_gmls.generateAlphas();

  Compadre::Evaluator dd_eval(&dd_gmls);

  auto grad_u =
    dd_eval.applyAlphasToDataAllComponentsAllTargetSites<Real**,DevMemory>(
      u, Compadre::GradientOfScalarPointEvaluation);
  auto grad_v =
    dd_eval.applyAlphasToDataAllComponentsAllTargetSites<Real**,DevMemory>(
      v, Compadre::GradientOfScalarPointEvaluation);
  auto grad_w =
    dd_eval.applyAlphasToDataAllComponentsAllTargetSites<Real**,DevMemory>(
      w, Compadre::GradientOfScalarPointEvaluation);

  auto dd_view = gather.scalar_fields.at("double_dot_gmls");
  Kokkos::parallel_for(gather.n(),
    KOKKOS_LAMBDA (const Index i) {
      dd_view(i) = 0;
      for (int ii=0; ii<3; ++ii) {
        for (int jj=0; jj<3; ++jj) {
          const Real ij_fac = (ii == 0 ? grad_u(i, jj) : (ii == 1 ? grad_v(i, jj) : grad_w(i, jj)));
          const Real ji_fac = (jj == 0 ? grad_u(i, ii) : (jj == 1 ? grad_v(i, ii) : grad_w(i, ii)));
          dd_view(i) += ij_fac * ji_fac;
        }
      }
    });

  ScatterMeshData<SeedType> scatter(gather, sphere.mesh);
  scatter.scatter_fields(vert_gmls_scalar_fields, face_gmls_scalar_fields);

  Kokkos::parallel_for(sphere.mesh.n_vertices_host(),
    KOKKOS_LAMBDA (const Index i) {
      const auto xi = Kokkos::subview(passive_x, i, Kokkos::ALL);
      const Real lat = SphereGeometry::latitude(xi);
      const Real lon = SphereGeometry::longitude(xi);

      dd_passive_exact(i) = ddot_exact(xi);
      dd_passive_error(i) = abs(dd_passive(i) - dd_passive_exact(i));

      dd_gmls_err_view_passive(i) = abs(dd_passive_exact(i) - dd_gmls_passive(i));
    });
  const auto active_u = sphere.velocity_active.view;
  Kokkos::parallel_for(sphere.mesh.n_faces_host(),
    KOKKOS_LAMBDA (const Index i) {
      const auto xi = Kokkos::subview(active_x, i, Kokkos::ALL);
      const Real lat = SphereGeometry::latitude(xi);
      const Real lon = SphereGeometry::longitude(xi);

      dd_active_exact(i) = ddot_exact(xi);
      dd_active_error(i) = abs(dd_active(i) - dd_active_exact(i));

      dd_gmls_err_view_active(i) = abs(dd_active_exact(i) - dd_gmls_active(i));
    });
}

int main (int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  Comm comm(MPI_COMM_WORLD);
  Logger<> logger("bve_rh54", Log::level::debug, comm);
//   using seed_type = CubedSphereSeed;
  using seed_type = IcosTriSphereSeed;
  using Coriolis = CoriolisSphere;
  using Vorticity = RossbyHaurwitz54;
  using Lat0 = LatitudeTracer;
  using Solver = Incompressible2DRK2<seed_type>;
    Kokkos::initialize(argc, argv);
  {
    user::Input input("bve_vorticity");
    {
      user::Option tfinal_option("tfinal", "-tf", "--time-final", "time final", 0.0);
      input.add_option(tfinal_option);

      user::Option nsteps_option("nsteps", "-n", "--nsteps", "number of time steps", 0);
      input.add_option(nsteps_option);

      user::Option tree_depth_option("tree_depth", "-d", "--depth", "mesh tree initial uniform depth", 4);
      input.add_option(tree_depth_option);

      user::Option amr_refinement_buffer_option("amr_buffer", "-ab", "--amr-buffer", "amr memory buffer", 0);
      input.add_option(amr_refinement_buffer_option);

      user::Option amr_refinement_limit_option("amr_limit", "-al", "--amr-limit", "amr refinement limit", 0);
      input.add_option(amr_refinement_limit_option);

      user::Option max_circulation_option("max_circulation_tol", "-c", "--circuluation-max", "amr max circulation tolerance", std::numeric_limits<Real>::max());
      input.add_option(max_circulation_option);

      user::Option amr_both_option("amr_both", "-amr", "--amr-both", "both amr buffer and limit values", LPM_NULL_IDX);
      input.add_option(amr_both_option);

      user::Option output_write_frequency_option("output_write_frequency", "-of", "--output-frequency", "output write frequency", 1);
      input.add_option(output_write_frequency_option);

      user::Option kernel_smoothing_parameter_option("kernel_smoothing_parameter", "-eps", "--velocity-epsilon", "velocity kernel smoothing parameter", 0.0);
      input.add_option(kernel_smoothing_parameter_option);

      user::Option output_file_root_option("output_file_root", "-o", "--output-file-root", "output file root", std::string("rh54"));
      input.add_option(output_file_root_option);

      user::Option remesh_interval_option("remesh_interval", "-rm", "--remesh-interval", "number of timesteps allowed between remesh interpolations", std::numeric_limits<int>::max());
      input.add_option(remesh_interval_option);

      user::Option remesh_strategy_option("remesh_strategy", "-rs", "--remesh-strategy", "direct or indirect remeshing strategy", std::string("direct"), std::set<std::string>({"direct", "indirect"}));
      input.add_option(remesh_strategy_option);

      user::Option remesh_interpolation_order("remesh_interpolation_order", "-ro", "--remesh-order", "polynomial order for gmls-based remesh interpolation", 4);
      input.add_option(remesh_interpolation_order);
    }
    input.parse_args(argc, argv);
    if (input.help_and_exit) {
      std::cout << input.usage();
      Kokkos::finalize();
      MPI_Finalize();
      return 1;
    }

    const int nsteps = input.get_option("nsteps").get_int();
    const Real dt = input.get_option("tfinal").get_real() / nsteps;
    int frame_counter = 0;
    const int write_frequency = input.get_option("output_write_frequency").get_int();
    logger.info(input.info_string());
    logger.info("dt = {}", dt);

    /**
      AMR
    */
    Real max_circ_tol = input.get_option("max_circulation_tol").get_real();
    Int amr_buffer = input.get_option("amr_buffer").get_int();
    Int amr_limit = input.get_option("amr_limit").get_int();
    if (input.get_option("amr_both").get_int() > 0) {
      amr_buffer = input.get_option("amr_both").get_int();
      amr_limit = input.get_option("amr_both").get_int();
    }
    const bool amr = (amr_buffer > 0 and amr_limit > 0);

    /**
    Build the particle/panel mesh
    */
    constexpr Real sphere_radius = 1;
    PolyMeshParameters<seed_type> mesh_params(
        input.get_option("tree_depth").get_int(),
        sphere_radius,
        amr_buffer,
        amr_limit);

    Coriolis coriolis;
    Vorticity vorticity;
    const Real ddot_eps = input.get_option("kernel_smoothing_parameter").get_real();
    constexpr Real velocity_eps = 0;

    auto sphere = std::make_unique<Incompressible2D<seed_type>>(mesh_params,
      coriolis, velocity_eps);
    sphere->init_vorticity(vorticity);

    if (amr) {
      Refinement<seed_type> refiner(sphere->mesh);
      ScalarIntegralFlag max_circulation_flag(refiner.flags,
        sphere->rel_vort_active.view,
        sphere->mesh.faces.area,
        sphere->mesh.faces.mask,
        sphere->mesh.n_faces_host(),
        max_circ_tol);

        max_circulation_flag.set_tol_from_relative_value();
        max_circ_tol = max_circulation_flag.tol;

        logger.info("amr is enabled with limit {}, max_circ_tol = {}",
          amr_limit, max_circ_tol);

        Index face_start_idx = 0;
        for (int i=0; i<amr_limit; ++i) {
          const Index face_end_idx = sphere->mesh.n_faces_host();
          refiner.iterate(face_start_idx, face_end_idx, max_circulation_flag);

          logger.info("amr iteration {}: initial circulation refinement count = {}",
            i, refiner.count[0]);

          sphere->mesh.divide_flagged_faces(refiner.flags, logger);
          sphere->update_device();
          sphere->init_vorticity(vorticity);

          face_start_idx = face_end_idx;
        }
    }
    else {
      logger.info("amr is not enabled; using uniform meshes.");
    }
    sphere->init_direct_sums();
    sphere->eps = ddot_eps;

    Kokkos::View<Real*[3]> velocity_exact_passive("velocity_exact", sphere->mesh.n_vertices_host());
    Kokkos::View<Real*[3]> velocity_exact_active("velocity_exact", sphere->mesh.n_faces_host());
    Kokkos::View<Real*[3]> velocity_error_passive("velocity_error", sphere->mesh.n_vertices_host());
    Kokkos::View<Real*[3]> velocity_error_active("velocity_error", sphere->mesh.n_faces_host());

    Lat0 lat0;
    sphere->init_tracer(lat0);
    sphere->allocate_tracer(std::string("double_dot"));
    sphere->allocate_tracer(std::string("double_dot_exact"));
    sphere->allocate_tracer(std::string("double_dot_error"));
    sphere->allocate_tracer(std::string("double_dot_gmls"));
    sphere->allocate_tracer(std::string("double_dot_gmls_error"));
    logger.info(sphere->info_string());
    do_double_dot(*sphere);
    do_velocity(velocity_exact_passive, velocity_exact_active,
                velocity_error_passive, velocity_error_active, *sphere);

    ErrNorms velocity_err(velocity_error_active, velocity_exact_active,
                          sphere->mesh.faces.area);
    ErrNorms ddot_err(sphere->tracer_active.at("double_dot_error").view,
                      sphere->tracer_active.at("double_dot_exact").view,
                      sphere->mesh.faces.area);
    ErrNorms ddot_gmls_err(sphere->tracer_active.at("double_dot_gmls_error").view,
                           sphere->tracer_active.at("double_dot_exact").view,
                           sphere->mesh.faces.area);
    logger.info("velocity error : {}", velocity_err.info_string());
    logger.info("double dot convolution error: {}", ddot_err.info_string());
    logger.info("double dot gmls error : {}", ddot_gmls_err.info_string());

    const auto vel_range = sphere->velocity_active.range(sphere->mesh.n_faces_host());
    const Real cr = vel_range.second * dt / sphere->mesh.appx_mesh_size();
    logger.info("velocity magnitude (min, max) = ({}, {}); approximate Courant number = {}",
      vel_range.first, vel_range.second, cr);
    if constexpr (std::is_same<Solver, Incompressible2DRK2<seed_type>>::value) {
      if (cr > 0.5) {
        logger.warn("Courant number {} is likely too high.", cr);
      }
    }
    const int remesh_interval = input.get_option("remesh_interval").get_int();
    const std::string remesh_strategy = input.get_option("remesh_strategy").get_str();
    auto solver = std::make_unique<Incompressible2DRK2<seed_type>>(dt, *sphere);
    gmls::Params gmls_params(input.get_option("remesh_interpolation_order").get_int());

#ifdef LPM_USE_VTK
    std::string amr_str = "_";
    if (amr) {
      amr_str = "amr" + std::to_string(amr_limit) + "_";
      if (max_circ_tol < 1) {
        amr_str += "gamma_tol" + float_str(max_circ_tol);
      }
      amr_str += "_";
    }
    const std::string eps_str = "eps"+float_str(sphere->eps);
    const std::string resolution_str =
      std::to_string(input.get_option("tree_depth").get_int()) + dt_str(dt);
    const std::string remesh_str = (remesh_interval < nsteps ? remesh_strategy + "rm" + std::to_string(remesh_interval) : "no_rm");
    const std::string vtk_file_root = input.get_option("output_file_root").get_str()
      + "_" + seed_type::id_string() + resolution_str + eps_str + "_" + remesh_str + amr_str;
    {
      sphere->update_host();
      auto vtk = vtk_mesh_interface(*sphere);
      vtk.add_vector_point_data(velocity_error_passive);
      vtk.add_vector_cell_data(velocity_error_active);
      vtk.add_vector_point_data(velocity_exact_passive);
      vtk.add_vector_cell_data(velocity_exact_active);
      auto ctr_str = zero_fill_str(frame_counter);
      const std::string vtk_fname = vtk_file_root + ctr_str + vtp_suffix();
      logger.info("writing output at t = {} to file {}", sphere->t, vtk_fname);
      vtk.write(vtk_fname);
    }
#endif

    /**
    time stepping
    */
    int rm_counter = 0;
    for (int t_idx=0; t_idx<nsteps; ++t_idx) {
      if ( (t_idx+1)%remesh_interval == 0 ) {
        logger.debug("remesh {} triggered by remesh interval", ++rm_counter);

        auto new_sphere = std::make_unique<Incompressible2D<seed_type>>(mesh_params,
           coriolis, input.get_option("kernel_smoothing_parameter").get_real());
        new_sphere->t = sphere->t;
        new_sphere->allocate_tracer(lat0);
        new_sphere->allocate_tracer(std::string("double_dot"));
        new_sphere->allocate_tracer(std::string("double_dot_exact"));
        new_sphere->allocate_tracer(std::string("double_dot_error"));
        new_sphere->allocate_tracer(std::string("double_dot_gmls"));
        new_sphere->allocate_tracer(std::string("double_dot_gmls_error"));

        auto remesh = compadre_remesh(*new_sphere, *sphere, gmls_params);
        if (amr) {
          Refinement<seed_type> refiner(new_sphere->mesh);
            ScalarIntegralFlag max_circulation_flag(refiner.flags,
              new_sphere->rel_vort_active.view,
              new_sphere->mesh.faces.area,
              new_sphere->mesh.faces.mask,
              new_sphere->mesh.n_faces_host(),
              max_circ_tol);
          if (remesh_strategy == "direct") {
            remesh.adaptive_direct_remesh(refiner, max_circulation_flag);
          }
          else {
            remesh.adaptive_indirect_remesh(refiner, max_circulation_flag,
              vorticity, coriolis, lat0);
          }
        }
        else {
          if (remesh_strategy == "direct") {
            remesh.uniform_direct_remesh();
          }
          else {
            remesh.uniform_indirect_remesh(vorticity, coriolis, lat0);
          }
        }

        sphere = std::move(new_sphere);
        solver.reset(new Incompressible2DRK2<seed_type>(dt, *sphere, solver->t_idx));
      }

      sphere->advance_timestep(*solver);
      do_double_dot(*sphere);
      logger.debug("t = {}", sphere->t);

    #ifdef LPM_USE_VTK
      if ((t_idx+1)%write_frequency == 0) {
        sphere->update_host();
        auto vtk = vtk_mesh_interface(*sphere);
        auto ctr_str = zero_fill_str(++frame_counter);
        const std::string vtk_fname = vtk_file_root + ctr_str + vtp_suffix();
        logger.info("writing output at t = {} to file: {}", sphere->t, vtk_fname);
        vtk.write(vtk_fname);
      }
    #endif
    }


  } // kokkos scope
  Kokkos::finalize();
  MPI_Finalize();
}

