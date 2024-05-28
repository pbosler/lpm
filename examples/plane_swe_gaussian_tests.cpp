#include "LpmConfig.h"
#include "lpm_comm.hpp"
#include "lpm_coriolis.hpp"
#include "lpm_error.hpp"
#include "lpm_error_impl.hpp"
#include "lpm_geometry.hpp"
#include "lpm_input.hpp"
#include "lpm_logger.hpp"
#include "lpm_pse.hpp"
#include "lpm_surface_gallery.hpp"
#include "lpm_swe_problem_gallery.hpp"
#include "lpm_swe.hpp"
#include "lpm_swe_impl.hpp"
#include "lpm_swe_rk4.hpp"
#include "lpm_swe_rk4_impl.hpp"
#include "util/lpm_string_util.hpp"
#include "util/lpm_timer.hpp"
#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"
#endif

using namespace Lpm;

void input_init(user::Input& input);

template <typename PtType> KOKKOS_INLINE_FUNCTION
Kokkos::Tuple<Real,2> velocity_exact(const PtType& x);

int main (int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  Comm comm(MPI_COMM_WORLD);
  Logger<> logger("plane_gravity_wave", Log::level::debug, comm);

  // compile-time settings
  // mesh seed
  using seed_type = QuadRectSeed; // TriHexSeed;

  // plane Gaussian test problem setup
  using topography_type  = ZeroFunctor;
  using init_sfc_type = UniformDepthSurface;
  using coriolis_type = CoriolisBetaPlane;
  using geo = PlaneGeometry;
  using pse_type = pse::BivariateOrder8<geo>;
  using stream_fn_type = PlanarGaussian;
  using potential_fn_type = PlanarGaussian;
  using vorticity_type = PlanarNegativeLaplacianOfGaussian;
  using divergence_type = PlanarNegativeLaplacianOfGaussian;

  Kokkos::initialize(argc, argv);
  { // Kokkos scope
    // initialize command line input
    user::Input input("plane_gaussian_tests");
    input_init(input);
    // parse command line args
    input.parse_args(argc, argv);
    if (input.help_and_exit) {
      std::cout << input.usage();
      Kokkos::finalize();
      MPI_Finalize();
      return 1;
    }
    logger.info(input.info_string());

    const int nsteps = input.get_option("nsteps").get_int();
    const Real dt = input.get_option("tfinal").get_real() / nsteps;
    int frame_counter = 0;
    const int write_frequency = input.get_option("output_write_frequency").get_int();

    Timer total_time("total_time");

    // initialize planar particle/panel mesh
    PolyMeshParameters<seed_type> mesh_params(
      input.get_option("tree_depth").get_int(),
      input.get_option("mesh_radius").get_real(),
      input.get_option("amr_limit").get_int());

    coriolis_type coriolis(input.get_option("f-coriolis").get_real(),
      input.get_option("beta-coriolis").get_real());

    auto plane = std::make_unique<SWE<seed_type>>(mesh_params, coriolis);
    plane->set_kernel_parameters(input.get_option("kernel_smoothing_parameter").get_real(),
      pse_type::epsilon(plane->mesh.appx_mesh_size(), input.get_option("pse_kernel_width_power").get_real()));


    // set problem initial conditions
    topography_type topo;
    const Real h0 = input.get_option("h0").get_real();
    init_sfc_type sfc(h0);
    plane->init_surface(topo, sfc);

    stream_fn_type psi(input.get_option("zeta0").get_real(),
                       input.get_option("zetab").get_real(),
                       input.get_option("zetax").get_real(),
                       input.get_option("zetay").get_real());
    potential_fn_type phi(input.get_option("sigma0").get_real(),
                         input.get_option("sigmab").get_real(),
                         input.get_option("sigmax").get_real(),
                         input.get_option("sigmay").get_real());

    constexpr bool depth_set = true;
    vorticity_type vorticity(psi);
    divergence_type divergence(phi);
    plane->init_vorticity(vorticity, depth_set);
    plane->init_divergence(divergence);

    constexpr bool do_velocity = true;
    plane->init_direct_sums(do_velocity);

    logger.info("mesh initialized");
    logger.info(plane->info_string());

    /**
      t = 0 only: set up exact solutions
    */
    scalar_view_type psi_exact_passive("psi_exact", plane->mesh.n_vertices_host());
    scalar_view_type psi_exact_active("psi_exact", plane->mesh.n_faces_host());
    scalar_view_type psi_error_passive("psi_error", plane->mesh.n_vertices_host());
    scalar_view_type psi_error_active("psi_error", plane->mesh.n_faces_host());
    scalar_view_type phi_exact_passive("phi_exact", plane->mesh.n_vertices_host());
    scalar_view_type phi_exact_active("phi_exact", plane->mesh.n_faces_host());
    scalar_view_type phi_error_passive("phi_error", plane->mesh.n_vertices_host());
    scalar_view_type phi_error_active("phi_error", plane->mesh.n_faces_host());
    Kokkos::View<Real*[2]> velocity_exact_passive("velocity_exact", plane->mesh.n_vertices_host());
    scalar_view_type double_dot_exact_passive("double_dot_exact", plane->mesh.n_vertices_host());
    Kokkos::View<Real*[2]> velocity_exact_active("velocity_exact", plane->mesh.n_faces_host());
    scalar_view_type double_dot_exact_active("double_dot_exact", plane->mesh.n_faces_host());

    Kokkos::parallel_for(plane->mesh.n_vertices_host(),
      PlanarGaussianTestVelocity(velocity_exact_passive,
        double_dot_exact_passive,
        plane->mesh.vertices.phys_crds.view,
        vorticity, divergence));
    Kokkos::parallel_for(plane->mesh.n_faces_host(),
      PlanarGaussianTestVelocity(velocity_exact_active,
        double_dot_exact_active,
        plane->mesh.faces.phys_crds.view,
        vorticity, divergence));
    auto crds = plane->mesh.vertices.phys_crds.view;
    Kokkos::parallel_for(plane->mesh.n_vertices_host(),
      KOKKOS_LAMBDA (const Index i) {
        const auto xi = Kokkos::subview(crds, i, Kokkos::ALL);
        psi_exact_passive(i) = psi(xi);
        phi_exact_passive(i) = phi(xi);
      });
    crds = plane->mesh.faces.phys_crds.view;
    Kokkos::parallel_for(plane->mesh.n_faces_host(),
      KOKKOS_LAMBDA (const Index i) {
        const auto xi = Kokkos::subview(crds, i, Kokkos::ALL);
        psi_exact_active(i) = psi(xi);
        phi_exact_active(i) = phi(xi);
      });

    /**
      t = 0 only: compute velocity and double dot error
    */
    Kokkos::View<Real*[2]> velocity_error_passive("velocity_error", plane->mesh.n_vertices_host());
    compute_error(velocity_error_passive, plane->velocity_passive.view, velocity_exact_passive);

    Kokkos::View<Real*[2]> velocity_error_active("velocity_error", plane->mesh.n_faces_host());
    ErrNorms vel_err(velocity_error_active, plane->velocity_active.view,
      velocity_exact_active, plane->mesh.faces.area);
    logger.info("velocity error: {}", vel_err.info_string());

    scalar_view_type ddot_error_passive("double_dot_error", plane->mesh.n_vertices_host());
    scalar_view_type ddot_error_active("double_dot_error", plane->mesh.n_faces_host());
    compute_error(ddot_error_passive, plane->double_dot_passive.view, double_dot_exact_passive);
    ErrNorms ddot_err(ddot_error_active, plane->double_dot_active.view, double_dot_exact_active, plane->mesh.faces.area);
    logger.info("double dot error: {}", ddot_err.info_string());

    compute_error(psi_error_passive, plane->stream_fn_passive.view, psi_exact_passive);
    ErrNorms stream_err(psi_error_active, plane->stream_fn_active.view, psi_exact_active, plane->mesh.faces.area);
    logger.info("stream function error: {}", stream_err.info_string());

    compute_error(phi_error_passive, plane->potential_passive.view, phi_exact_passive);
    ErrNorms potential_err(phi_error_active, plane->potential_active.view, phi_exact_active,
      plane->mesh.faces.area);
    logger.info("potential function error: {}", potential_err.info_string());

#ifdef LPM_USE_VTK
    const std::string resolution_str = std::to_string(input.get_option("tree_depth").get_int());
    const std::string vtk_file_root = input.get_option("output_file_root").get_str()
      + "_" + seed_type::id_string() + resolution_str + "_";
    {
      plane->update_host();
      auto vtk = vtk_mesh_interface(*plane);
      vtk.add_vector_point_data(velocity_error_passive);
      vtk.add_vector_point_data(velocity_exact_passive);
      vtk.add_scalar_point_data(double_dot_exact_passive);
      vtk.add_scalar_point_data(ddot_error_passive);
      vtk.add_scalar_point_data(psi_error_passive);
      vtk.add_scalar_point_data(psi_exact_passive);
      vtk.add_scalar_point_data(phi_error_passive);
      vtk.add_scalar_point_data(phi_exact_passive);
      vtk.add_vector_cell_data(velocity_exact_active);
      vtk.add_vector_cell_data(velocity_error_active);
      vtk.add_scalar_cell_data(double_dot_exact_active);
      vtk.add_scalar_cell_data(ddot_error_active);
      vtk.add_scalar_cell_data(psi_error_active);
      vtk.add_scalar_cell_data(psi_exact_active);
      vtk.add_scalar_cell_data(phi_error_active);
      vtk.add_scalar_cell_data(phi_exact_active);
      auto ctr_str = zero_fill_str(frame_counter);
      const std::string vtk_fname = vtk_file_root + ctr_str + vtp_suffix();
      logger.info("writing output at t = {} to file: {}", plane->t, vtk_fname);
      vtk.write(vtk_fname);
    }
#endif

    // setup time stepper
    auto solver = std::make_unique<SWERK4<seed_type, topography_type>>(dt, *plane, topo);
    logger.info(solver->info_string());

    for (int t_idx=0; t_idx<nsteps; ++t_idx) {
      plane->advance_timestep(*solver);
      logger.debug("t = {}", plane->t);

#ifdef LPM_USE_VTK
      if ((t_idx+1)%write_frequency == 0) {
        plane->update_host();
        auto vtk = vtk_mesh_interface(*plane);
        auto ctr_str = zero_fill_str(++frame_counter);
        const std::string vtk_fname = vtk_file_root + ctr_str + vtp_suffix();
        logger.info("writing output at t = {} to file: {}", plane->t, vtk_fname);
        vtk.write(vtk_fname);
      }
#endif
    }

    total_time.stop();
    logger.info("total time: {}", total_time.info_string());
  } // kokkos scope
  Kokkos::finalize();
  MPI_Finalize();
}

void input_init(user::Input& input) {
  // define user parameters
  user::Option tfinal_option("tfinal", "-tf", "--time_final", "time final", 0.0);
  input.add_option(tfinal_option);

  user::Option nsteps_option("nsteps", "-n", "--nsteps", "number of steps", 0);
  input.add_option(nsteps_option);

  user::Option tree_depth_option("tree_depth", "-d", "--depth", "mesh tree depth", 4);

  input.add_option(tree_depth_option);
  user::Option f_coriolis_option("f-coriolis", "-f", "--f-coriolis", "f coriolis", 0.0);
  input.add_option(f_coriolis_option);

  user::Option beta_coriolis_option("beta-coriolis", "-b", "--beta-coriolis", "beta coriolis", 0.0);
  input.add_option(beta_coriolis_option);

  user::Option h0_option("h0", "-h0", "--init-depth", "initial uniform depth", 1.0);
  input.add_option(h0_option);

  user::Option zeta0_option("zeta0", "-zeta0", "--init-zeta-strength", "initial vorticity strength", 1.0);
  input.add_option(zeta0_option);

  user::Option zetax_option("zetax", "-zetax", "--init-zeta-x", "initial vorticity center, x-coordinate", 0.0);
  input.add_option(zetax_option);

  user::Option zetay_option("zetay", "-zetay", "--init-zeta-y", "initial vorticity center, y-coordinate", 0.0);
  input.add_option(zetay_option);

  user::Option zetab_option("zetab", "-zetab", "--init-zeta-b", "initial vorticity shape parameter", 0.25);
  input.add_option(zetab_option);

    user::Option sigma0_option("sigma0", "-sigma0", "--init-sigma-strength", "initial divergence strength", 0.1);
  input.add_option(sigma0_option);

  user::Option sigmax_option("sigmax", "-sigmax", "--init-sigma-x", "initial divergence center, x-coordinate", 0.5);
  input.add_option(sigmax_option);

  user::Option sigmay_option("sigmay", "-sigmay", "--init-sigma-y", "initial divergence center, y-coordinate", 0.0);
  input.add_option(sigmay_option);

  user::Option sigmab_option("sigmab", "-sigmab", "--init-sigma-b", "initial divergence shape parameter", 2.0);
  input.add_option(sigmab_option);

  user::Option mesh_radius_option("mesh_radius", "-r", "--radius", "mesh radius", 6.0);
  input.add_option(mesh_radius_option);

  user::Option amr_refinement_buffer_option("amr_buffer", "-ab", "--amr-buffer", "amr memory buffer", 0);
  input.add_option(amr_refinement_buffer_option);

  user::Option amr_refinement_limit_option("amr_limit", "-al", "--amr-limit", "amr refinement limit", 0);
  input.add_option(amr_refinement_limit_option);

  user::Option output_file_root_option("output_file_root", "-o", "--output-file-root", "output file root", std::string("gaussian_tests"));
  input.add_option(output_file_root_option);

  user::Option output_write_frequency_option("output_write_frequency", "-of", "--output-frequency", "output write frequency", 1);
  input.add_option(output_write_frequency_option);

  user::Option kernel_smoothing_parameter_option("kernel_smoothing_parameter", "-eps", "--velocity-epsilon", "velocity kernel smoothing parameter", 0.0);
  input.add_option(kernel_smoothing_parameter_option);

  user::Option pse_power_option("pse_kernel_width_power", "-pse", "--pse-kernel-width-power", "pse kernel width power",
    11.0/20);
  input.add_option(pse_power_option);
}
