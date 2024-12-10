#include "LpmConfig.h"
#include "lpm_comm.hpp"
#include "lpm_constants.hpp"
#include "lpm_error.hpp"
#include "lpm_error_impl.hpp"
#include "lpm_high_order_swe.hpp"
#include "lpm_logger.hpp"
#include "lpm_surface_gallery.hpp"
#include "lpm_swe.hpp"
#include "lpm_swe_problem_gallery.hpp"
#include "lpm_swe_impl.hpp"
#include "lpm_regularized_kernels.hpp"
#include "util/lpm_floating_point.hpp"
#include "util/lpm_math.hpp"
#include "util/lpm_timer.hpp"
#include "util/lpm_test_utils.hpp"
#include "util/lpm_stl_utils.hpp"
#include "util/lpm_string_util.hpp"
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using namespace Lpm;
using Catch::Approx;

template <typename SeedType, typename KernelType>
struct HighOrderTest {
  using Topography  = ZeroFunctor;
  using InitialSurface = UniformDepthSurface;
  using Coriolis = CoriolisBetaPlane;
  using Geo = PlaneGeometry;
  using StreamFunction = PlanarGaussian;
  using PotentialFunction = PlanarGaussian;
  using Vorticity = PlanarNegativeLaplacianOfGaussian;
  using Divergence = PlanarNegativeLaplacianOfGaussian;
  using vec_view = typename SeedType::geo::vec_view_type;
  using crd_view = typename SeedType::geo::crd_view_type;

  Int d_min; // minimum tree depth
  Int d_max; // maximum tree depth
  std::vector<Real> kernel_powers;
  static constexpr Real radius = 10;
  static constexpr Real zeta0 = 1; // stream function maximum
  static constexpr Real zetab = 0.25; // stream function shape parameter
  static constexpr Real zetax = -1; // stream function center x1
  static constexpr Real zetay = 0; // stream function center x2
  static constexpr Real delta0 = 1; // potential function maximum
  static constexpr Real deltab = 0.25; // potential function shape parameter
  static constexpr Real deltax = 1; // potential function center x1
  static constexpr Real deltay = 0; // potential function center x2
  static constexpr Real h0 = 1; // initial depth
  std::vector<Real> dxs;
  std::map<Real, std::vector<Real>> interp_l1;
  std::map<Real, std::vector<Real>> interp_l2;
  std::map<Real, std::vector<Real>> interp_linf;
  std::map<Real, std::vector<Real>> velocity_l1;
  std::map<Real, std::vector<Real>> velocity_l2;
  std::map<Real, std::vector<Real>> velocity_linf;
  std::map<Real, std::vector<Real>> ddot_l1;
  std::map<Real, std::vector<Real>> ddot_l2;
  std::map<Real, std::vector<Real>> ddot_linf;
  std::map<Real, std::vector<Real>> gradient_l1;
  std::map<Real, std::vector<Real>> gradient_l2;
  std::map<Real, std::vector<Real>> gradient_linf;
  std::map<Real, std::vector<Real>> left_gradient_l1;
  std::map<Real, std::vector<Real>> left_gradient_l2;
  std::map<Real, std::vector<Real>> left_gradient_linf;
  std::map<Real, std::vector<Real>> interior_gradient_l1;
  std::map<Real, std::vector<Real>> interior_gradient_l2;
  std::map<Real, std::vector<Real>> interior_gradient_linf;
  std::map<Real, std::vector<Real>> lap_l1;
  std::map<Real, std::vector<Real>> lap_l2;
  std::map<Real, std::vector<Real>> lap_linf;

  HighOrderTest(const Int dmin, const int dmax) : d_min(dmin), d_max(dmax) {
    for (int numer = 7; numer<=19; numer += 2) {
      kernel_powers.push_back(numer/20.0);
    }
  }

  void run() {
    Comm comm;

    std::string main_log_name;
    {
      std::ostringstream ss;
      ss << "high_order_kernels" << KernelType::order << "_tests";
      main_log_name = ss.str();
    }

    Logger<LogBasicFile<>> main_logger(main_log_name, Log::level::debug, comm);
    main_logger.debug("test run called.");
    main_logger.info("===================================");

    Timer test_time("test_time");
    test_time.start();

    StreamFunction psi(zeta0, zetab, zetax, zetay);
    PotentialFunction phi(delta0, deltab, deltax, deltay);
    Topography topo;
    InitialSurface sfc(h0);
    Vorticity vorticity(psi);
    Divergence divergence(phi);

    for (int i=0; i<(d_max - d_min)+1; ++i) {
      const int amr_limit = 0;
      const int depth = d_min + i;

      PolyMeshParameters<SeedType> mesh_params(depth, radius, amr_limit);
      Coriolis coriolis;
      auto plane = std::make_unique<SWE<SeedType>>(mesh_params, coriolis);
      const Real dx = plane->mesh.appx_mesh_size();
      dxs.push_back(dx);

      plane->init_surface(topo, sfc);
      constexpr bool depth_set = true;
      plane->init_vorticity(vorticity, depth_set);
      plane->init_divergence(divergence);
      Kokkos::TeamPolicy<> vertex_policy(plane->mesh.n_vertices_host(), Kokkos::AUTO());
      Kokkos::TeamPolicy<> face_policy(plane->mesh.n_faces_host(), Kokkos::AUTO());

      // exact solutions
      scalar_view_type psi_exact_passive("psi_exact", plane->mesh.n_vertices_host());
      scalar_view_type psi_exact_active("psi_exact", plane->mesh.n_faces_host());
      scalar_view_type psi_error_passive("psi_error", plane->mesh.n_vertices_host());
      scalar_view_type psi_error_active("psi_error", plane->mesh.n_faces_host());
      scalar_view_type lap_psi_exact_passive("lap_psi_exact", plane->mesh.n_vertices_host());
      scalar_view_type lap_psi_exact_active("lap_psi_exact", plane->mesh.n_faces_host());
      scalar_view_type lap_psi_error_passive("lap_psi_error", plane->mesh.n_vertices_host());
      scalar_view_type lap_psi_error_active("lap_psi_error", plane->mesh.n_faces_host());
      Kokkos::View<Real*[2]> phi_grad_exact_passive("phi_grad_exact", plane->mesh.n_vertices_host());
      Kokkos::View<Real*[2]> phi_grad_exact_active("phi_grad_exact", plane->mesh.n_faces_host());
      Kokkos::View<Real*[2]> phi_grad_error_passive("phi_grad_error", plane->mesh.n_vertices_host());
      Kokkos::View<Real*[2]> phi_grad_error_active("phi_grad_error", plane->mesh.n_faces_host());
      Kokkos::View<Real*[2]> left_phi_grad_error_passive("left_phi_grad_error", plane->mesh.n_vertices_host());
      Kokkos::View<Real*[2]> left_phi_grad_error_active("left_phi_grad_error", plane->mesh.n_faces_host());
      Kokkos::View<Real*[2]> velocity_exact_passive("velocity_exact", plane->mesh.n_vertices_host());
      Kokkos::View<Real*[2]> velocity_error_passive("velocity_error", plane->mesh.n_vertices_host());
      Kokkos::View<Real*[2]> velocity_exact_active("velocity_exact", plane->mesh.n_faces_host());
      Kokkos::View<Real*[2]> velocity_error_active("velocity_error", plane->mesh.n_faces_host());
      scalar_view_type double_dot_exact_passive("double_dot_exact", plane->mesh.n_vertices_host());
      scalar_view_type double_dot_error_passive("double_dot_error", plane->mesh.n_vertices_host());
      scalar_view_type double_dot_exact_active("double_dot_exact", plane->mesh.n_faces_host());
      scalar_view_type double_dot_error_active("double_dot_error", plane->mesh.n_faces_host());
      scalar_view_type du1dx1_exact_passive("du1dx1_exact", plane->mesh.n_vertices_host());
      scalar_view_type du1dx1_exact_active("du1dx1_exact", plane->mesh.n_faces_host());
      scalar_view_type du1dx2_exact_passive("du1dx2_exact", plane->mesh.n_vertices_host());
      scalar_view_type du1dx2_exact_active("du1dx2_exact", plane->mesh.n_faces_host());
      scalar_view_type du2dx1_exact_passive("du2dx1_exact", plane->mesh.n_vertices_host());
      scalar_view_type du2dx1_exact_active("du2dx1_exact", plane->mesh.n_faces_host());
      scalar_view_type du2dx2_exact_passive("du2dx2_exact", plane->mesh.n_vertices_host());
      scalar_view_type du2dx2_exact_active("du2dx2_exact", plane->mesh.n_faces_host());

      Kokkos::parallel_for(plane->mesh.n_vertices_host(),
        PlanarGaussianTestVelocity(velocity_exact_passive,
          double_dot_exact_passive,
          du1dx1_exact_passive,
          du1dx2_exact_passive,
          du2dx1_exact_passive,
          du2dx2_exact_passive,
          plane->mesh.vertices.phys_crds.view,
          vorticity, divergence));
      Kokkos::parallel_for(plane->mesh.n_faces_host(),
        PlanarGaussianTestVelocity(velocity_exact_active,
          double_dot_exact_active,
          du1dx1_exact_active,
          du1dx2_exact_active,
          du2dx1_exact_active,
          du2dx2_exact_active,
          plane->mesh.faces.phys_crds.view,
          vorticity, divergence));
      auto crds = plane->mesh.vertices.phys_crds.view;
      auto phi_view = plane->potential_passive.view;
      Kokkos::parallel_for(plane->mesh.n_vertices_host(),
        KOKKOS_LAMBDA (const Index i) {
          const auto x_i = Kokkos::subview(crds, i, Kokkos::ALL);
          psi_exact_passive(i) = psi(x_i);
          phi_grad_exact_passive(i,0) = phi.ddx1(x_i);
          phi_grad_exact_passive(i,1) = phi.ddx2(x_i);
          lap_psi_exact_passive(i) = -vorticity(x_i);
          phi_view(i) = phi(x_i);
        });
      crds = plane->mesh.faces.phys_crds.view;
      phi_view = plane->potential_active.view;
      Kokkos::parallel_for(plane->mesh.n_faces_host(),
        KOKKOS_LAMBDA (const Index i) {
          const auto x_i = Kokkos::subview(crds, i, Kokkos::ALL);
          psi_exact_active(i) = psi(x_i);
          phi_grad_exact_active(i,0) = phi.ddx1(x_i);
          phi_grad_exact_active(i,1) = phi.ddx2(x_i);
          lap_psi_exact_active(i) = -vorticity(x_i);
          phi_view(i) = phi(x_i);
        });

      Kokkos::View<Real*[2]> grad_passive("grad_phi", plane->mesh.n_vertices_host());
      Kokkos::View<Real*[2]> grad_active("grad_phi", plane->mesh.n_faces_host());
      Kokkos::View<Real*[2]> left_grad_passive("left_grad_phi", plane->mesh.n_vertices_host());
      Kokkos::View<Real*[2]> left_grad_active("left_grad_phi", plane->mesh.n_faces_host());
      for (int j=0; j< kernel_powers.size(); ++j) {
        Real eps;
        if (dx > 1) {
          eps = std::pow(dx, 1/kernel_powers[j]);
        }
        else {
          eps = std::pow(dx, kernel_powers[j]);
        }
        plane->set_kernel_parameters(eps, eps);

        std::ostringstream ss;
        ss << "high_order_" << SeedType::id_string() << depth << "_ord" << KernelType::order << "_pow" << kernel_powers[j];
        const auto test_name = ss.str();
        ss.str("");
        Logger<> logger(test_name, Log::level::debug, comm);
        Timer iteration_timer(test_name);
        iteration_timer.start();

        if (!map_contains(interp_l1, kernel_powers[j])) {
          interp_l1.emplace(kernel_powers[j], std::vector<Real>());
          interp_l2.emplace(kernel_powers[j], std::vector<Real>());
          interp_linf.emplace(kernel_powers[j], std::vector<Real>());
          velocity_l1.emplace(kernel_powers[j], std::vector<Real>());
          velocity_l2.emplace(kernel_powers[j], std::vector<Real>());
          velocity_linf.emplace(kernel_powers[j], std::vector<Real>());
          ddot_l1.emplace(kernel_powers[j], std::vector<Real>());
          ddot_l2.emplace(kernel_powers[j], std::vector<Real>());
          ddot_linf.emplace(kernel_powers[j], std::vector<Real>());
          gradient_l1.emplace(kernel_powers[j], std::vector<Real>());
          gradient_l2.emplace(kernel_powers[j], std::vector<Real>());
          gradient_linf.emplace(kernel_powers[j], std::vector<Real>());
          left_gradient_l1.emplace(kernel_powers[j], std::vector<Real>());
          left_gradient_l2.emplace(kernel_powers[j], std::vector<Real>());
          left_gradient_linf.emplace(kernel_powers[j], std::vector<Real>());
          interior_gradient_l1.emplace(kernel_powers[j], std::vector<Real>());
          interior_gradient_l2.emplace(kernel_powers[j], std::vector<Real>());
          interior_gradient_linf.emplace(kernel_powers[j], std::vector<Real>());
          lap_l1.emplace(kernel_powers[j], std::vector<Real>());
          lap_l2.emplace(kernel_powers[j], std::vector<Real>());
          lap_linf.emplace(kernel_powers[j], std::vector<Real>());
        }
        KernelType kernels(eps);

        logger.debug("kernel power: {}", kernel_powers[j]);
        logger.info("{} start.  dx = {}, eps = {}, eps / dx = {}", test_name, dx, eps, eps/dx);

        // interpolate
        Kokkos::parallel_for(vertex_policy,
          DirectSum<PlaneScalarInterpolationReducer<KernelType>>(
            plane->stream_fn_passive.view,
            plane->mesh.vertices.phys_crds.view,
            psi_exact_passive,
            plane->mesh.faces.phys_crds.view,
            psi_exact_active,
            kernels,
            plane->mesh.faces.area,
            plane->mesh.faces.mask,
            plane->mesh.n_faces_host())
          );
        Kokkos::parallel_for(face_policy,
          DirectSum<PlaneScalarInterpolationReducer<KernelType>>(
            plane->stream_fn_active.view,
            plane->mesh.faces.phys_crds.view,
            psi_exact_active,
            plane->mesh.faces.phys_crds.view,
            psi_exact_active,
            kernels,
            plane->mesh.faces.area,
            plane->mesh.faces.mask,
            plane->mesh.n_faces_host())
          );
        compute_error(psi_error_passive, plane->stream_fn_passive.view, psi_exact_passive);
        ErrNorms psi_err(psi_error_active, plane->stream_fn_active.view, psi_exact_active,
          plane->mesh.faces.area);
        // solve for velocity
        plane->init_velocity_direct_sum(kernels);
        compute_error(velocity_error_passive, plane->velocity_passive.view, velocity_exact_passive);
        ErrNorms vel_err(velocity_error_active, plane->velocity_active.view, velocity_exact_active,
          plane->mesh.faces.area);
        compute_error(double_dot_error_passive, plane->double_dot_passive.view, double_dot_exact_passive);
        ErrNorms ddot_err(double_dot_error_active, plane->double_dot_active.view, double_dot_exact_active,
          plane->mesh.faces.area);
        // compute a gradient
        Kokkos::parallel_for(vertex_policy,
          DirectSum<PlaneGradientReducer<KernelType>>(
            grad_passive,
            plane->mesh.vertices.phys_crds.view,
            plane->potential_passive.view,
            plane->mesh.faces.phys_crds.view,
            plane->potential_active.view,
            kernels,
            plane->mesh.faces.area,
            plane->mesh.faces.mask,
            plane->mesh.n_faces_host())
          );
        Kokkos::parallel_for(face_policy,
          DirectSum<PlaneGradientReducer<KernelType>>(
            grad_active,
            plane->mesh.faces.phys_crds.view,
            plane->potential_active.view,
            plane->mesh.faces.phys_crds.view,
            plane->potential_active.view,
            kernels,
            plane->mesh.faces.area,
            plane->mesh.faces.mask,
            plane->mesh.n_faces_host())
          );
        compute_error(phi_grad_error_passive, grad_passive, phi_grad_exact_passive);
        ErrNorms grad_err(phi_grad_error_active, grad_active, phi_grad_exact_active,
          plane->mesh.faces.area);
        // compute a 1-sided gradient
        Kokkos::parallel_for(vertex_policy,
          DirectSum<PlaneOneSidedInteriorGradientReducer<KernelType>>(
            left_grad_passive,
            plane->mesh.vertices.phys_crds.view,
            plane->potential_passive.view,
            plane->mesh.faces.phys_crds.view,
            plane->potential_active.view,
            kernels,
            plane->mesh.faces.area,
            plane->mesh.faces.mask,
            plane->mesh.n_faces_host())
          );
        Kokkos::parallel_for(face_policy,
          DirectSum<PlaneOneSidedInteriorGradientReducer<KernelType>>(
            left_grad_active,
            plane->mesh.faces.phys_crds.view,
            plane->potential_active.view,
            plane->mesh.faces.phys_crds.view,
            plane->potential_active.view,
            kernels,
            plane->mesh.faces.area,
            plane->mesh.faces.mask,
            plane->mesh.n_faces_host())
          );
        compute_error(left_phi_grad_error_passive, left_grad_passive, phi_grad_exact_passive);
        ErrNorms left_grad_err(left_phi_grad_error_active, left_grad_active, phi_grad_exact_active,
          plane->mesh.faces.area);
        // compute a surface laplacian
        Kokkos::parallel_for(vertex_policy,
          DirectSum<PlaneLaplacianReducer<KernelType>>(
            plane->surf_lap_passive.view,
            plane->mesh.vertices.phys_crds.view,
            psi_exact_passive,
            plane->mesh.faces.phys_crds.view,
            psi_exact_active,
            kernels,
            plane->mesh.faces.area,
            plane->mesh.faces.mask,
            plane->mesh.n_faces_host())
          );
        Kokkos::parallel_for(face_policy,
          DirectSum<PlaneLaplacianReducer<KernelType>>(
            plane->surf_lap_active.view,
            plane->mesh.faces.phys_crds.view,
            psi_exact_active,
            plane->mesh.faces.phys_crds.view,
            psi_exact_active,
            kernels,
            plane->mesh.faces.area,
            plane->mesh.faces.mask,
            plane->mesh.n_faces_host())
          );
        compute_error(lap_psi_error_passive, plane->surf_lap_passive.view, lap_psi_exact_passive);
        ErrNorms lap_psi_err(lap_psi_error_active, plane->surf_lap_active.view, lap_psi_exact_active,
          plane->mesh.faces.area);

        logger.info("interp error: {}", psi_err.info_string());
        interp_l1.at(kernel_powers[j]).push_back(psi_err.l1);
        interp_l2.at(kernel_powers[j]).push_back(psi_err.l2);
        interp_linf.at(kernel_powers[j]).push_back(psi_err.linf);
        logger.info("velocity error: {}", vel_err.info_string());
        velocity_l1.at(kernel_powers[j]).push_back(vel_err.l1);
        velocity_l2.at(kernel_powers[j]).push_back(vel_err.l2);
        velocity_linf.at(kernel_powers[j]).push_back(vel_err.linf);
        logger.info("ddot error: {}", ddot_err.info_string());
        ddot_l1.at(kernel_powers[j]).push_back(ddot_err.l1);
        ddot_l2.at(kernel_powers[j]).push_back(ddot_err.l2);
        ddot_linf.at(kernel_powers[j]).push_back(ddot_err.linf);
        logger.info("gradient error: {}", grad_err.info_string());
        gradient_l1.at(kernel_powers[j]).push_back(grad_err.l1);
        gradient_l2.at(kernel_powers[j]).push_back(grad_err.l2);
        gradient_linf.at(kernel_powers[j]).push_back(grad_err.linf);
        logger.info("left gradient error: {}", left_grad_err.info_string());
        left_gradient_l1.at(kernel_powers[j]).push_back(left_grad_err.l1);
        left_gradient_l2.at(kernel_powers[j]).push_back(left_grad_err.l2);
        left_gradient_linf.at(kernel_powers[j]).push_back(left_grad_err.linf);
        logger.info("laplacian error: {}", lap_psi_err.info_string());
        lap_l1.at(kernel_powers[j]).push_back(lap_psi_err.l1);
        lap_l2.at(kernel_powers[j]).push_back(lap_psi_err.l2);
        lap_linf.at(kernel_powers[j]).push_back(lap_psi_err.linf);

        plane->update_host();
        auto vtk = vtk_mesh_interface(*plane);
        vtk.add_vector_point_data(velocity_error_passive);
        vtk.add_vector_point_data(velocity_exact_passive);
        vtk.add_scalar_point_data(double_dot_exact_passive);
        vtk.add_scalar_point_data(double_dot_error_passive);
        vtk.add_scalar_point_data(psi_error_passive);
        vtk.add_scalar_point_data(psi_exact_passive);
        vtk.add_vector_point_data(grad_passive);
        vtk.add_vector_point_data(phi_grad_exact_passive);
        vtk.add_vector_point_data(phi_grad_error_passive);
        vtk.add_vector_point_data(left_grad_passive);
        vtk.add_vector_point_data(left_phi_grad_error_passive);
        vtk.add_scalar_point_data(lap_psi_exact_passive);
        vtk.add_scalar_point_data(lap_psi_error_passive);

        vtk.add_vector_cell_data(velocity_exact_active);
        vtk.add_vector_cell_data(velocity_error_active);
        vtk.add_scalar_cell_data(double_dot_exact_active);
        vtk.add_scalar_cell_data(double_dot_error_active);
        vtk.add_scalar_cell_data(psi_error_active);
        vtk.add_scalar_cell_data(psi_exact_active);
        vtk.add_vector_cell_data(grad_active);
        vtk.add_vector_cell_data(phi_grad_exact_active);
        vtk.add_vector_cell_data(phi_grad_error_active);
        vtk.add_vector_cell_data(left_grad_active);
        vtk.add_vector_cell_data(left_phi_grad_error_active);
        vtk.add_scalar_cell_data(lap_psi_error_active);
        vtk.add_scalar_cell_data(lap_psi_exact_active);

        const std::string vtk_fname = test_name + vtp_suffix();
        vtk.write(vtk_fname);

        iteration_timer.stop();
        logger.info(iteration_timer.info_string());
      }
    }

    main_logger.debug("dxs : {}", sprarr(dxs));
    for (const auto& eps_err : interp_l1) {
      main_logger.debug("kernel_power {} : {}", eps_err.first, sprarr(eps_err.second));
      const auto interpl1_rate = convergence_rates(dxs, eps_err.second);
      const auto interpl2_rate = convergence_rates(dxs, interp_l2.at(eps_err.first));
      const auto interplinf_rate = convergence_rates(dxs, interp_linf.at(eps_err.first));
      const auto velocityl1_rate = convergence_rates(dxs, velocity_l1.at(eps_err.first));
      const auto velocityl2_rate = convergence_rates(dxs, velocity_l2.at(eps_err.first));
      const auto velocitylinf_rate = convergence_rates(dxs, velocity_linf.at(eps_err.first));
      const auto ddotl1_rate = convergence_rates(dxs, ddot_l1.at(eps_err.first));
      const auto ddotl2_rate = convergence_rates(dxs, ddot_l2.at(eps_err.first));
      const auto ddotlinf_rate = convergence_rates(dxs, ddot_linf.at(eps_err.first));
      const auto gradientl1_rate = convergence_rates(dxs, gradient_l1.at(eps_err.first));
      const auto gradientl2_rate = convergence_rates(dxs, gradient_l2.at(eps_err.first));
      const auto gradientlinf_rate = convergence_rates(dxs, gradient_linf.at(eps_err.first));
      const auto left_gradientl1_rate = convergence_rates(dxs, left_gradient_l1.at(eps_err.first));
      const auto left_gradientl2_rate = convergence_rates(dxs, left_gradient_l2.at(eps_err.first));
      const auto left_gradientlinf_rate = convergence_rates(dxs, left_gradient_linf.at(eps_err.first));
      const auto lapl1_rate = convergence_rates(dxs, lap_l1.at(eps_err.first));
      const auto lapl2_rate = convergence_rates(dxs, lap_l2.at(eps_err.first));
      const auto laplinf_rate = convergence_rates(dxs, lap_linf.at(eps_err.first));

      main_logger.info(convergence_table("dx", dxs, "interp_l1", eps_err.second, interpl1_rate));
      main_logger.info(convergence_table("dx", dxs, "interp_l2", interp_l2.at(eps_err.first), interpl2_rate));
      main_logger.info(convergence_table("dx", dxs, "interp_linf", interp_linf.at(eps_err.first), interplinf_rate));
      main_logger.info(convergence_table("dx", dxs, "velocity_l1", velocity_l1.at(eps_err.first), velocityl1_rate));
      main_logger.info(convergence_table("dx", dxs, "velocity_l2", velocity_l2.at(eps_err.first), velocityl2_rate));
      main_logger.info(convergence_table("dx", dxs, "velocity_linf", velocity_linf.at(eps_err.first), velocitylinf_rate));
      main_logger.info(convergence_table("dx", dxs, "ddot_l1", ddot_l1.at(eps_err.first), ddotl1_rate));
      main_logger.info(convergence_table("dx", dxs, "ddot_l2", ddot_l2.at(eps_err.first), ddotl2_rate));
      main_logger.info(convergence_table("dx", dxs, "ddot_linf", ddot_linf.at(eps_err.first), ddotlinf_rate));
      main_logger.info(convergence_table("dx", dxs, "gradient_l1", gradient_l1.at(eps_err.first), gradientl1_rate));
      main_logger.info(convergence_table("dx", dxs, "gradient_l2", gradient_l2.at(eps_err.first), gradientl2_rate));
      main_logger.info(convergence_table("dx", dxs, "gradient_linf", gradient_linf.at(eps_err.first), gradientlinf_rate));
      main_logger.info(convergence_table("dx", dxs, "left_gradient_l1", left_gradient_l1.at(eps_err.first), left_gradientl1_rate));
      main_logger.info(convergence_table("dx", dxs, "left_gradient_l2", left_gradient_l2.at(eps_err.first), left_gradientl2_rate));
      main_logger.info(convergence_table("dx", dxs, "left_gradient_linf", left_gradient_linf.at(eps_err.first), left_gradientlinf_rate));
      main_logger.info(convergence_table("dx", dxs, "lap_l1", lap_l1.at(eps_err.first), lapl1_rate));
      main_logger.info(convergence_table("dx", dxs, "lap_l2", lap_l2.at(eps_err.first), lapl2_rate));
      main_logger.info(convergence_table("dx", dxs, "lap_linf", lap_linf.at(eps_err.first), laplinf_rate));
    }

    test_time.stop();
    main_logger.info(test_time.info_string());


  }
};

TEST_CASE("high order kernels' convergence", "[swe]") {
  constexpr Int start_depth = 4;
  Int end_depth = 4;

  auto& ts = TestSession::get();
  if (ts.params.find("end-depth") != ts.params.end()) {
    end_depth = std::stoi(ts.params["end-depth"]);
  }
  for (const auto& p : ts.params) {
    std::cout << p.first << " = " << p.second << "\n";
  }

  SECTION("2nd order kernels") {
    HighOrderTest<QuadRectSeed, Plane2ndOrder> h2(start_depth, end_depth);
    h2.run();
  }
  SECTION("4th order kernels") {
    HighOrderTest<QuadRectSeed, Plane4thOrder> h4(start_depth, end_depth);
    h4.run();
  }
  SECTION("6th order kernels") {
    HighOrderTest<QuadRectSeed, Plane6thOrder> h6(start_depth, end_depth);
    h6.run();
  }
  SECTION("8th order kernels") {
    HighOrderTest<QuadRectSeed, Plane8thOrder> h8(start_depth, end_depth);
    h8.run();
  }
}

// KOKKOS_INLINE_FUNCTION
// Real b2fn(const Real& r) {return exp(-square(r))/constants::PI;}
//
// KOKKOS_INLINE_FUNCTION
// Real b8fn(const Real& r) {
//   const Real rsq = square(r);
//   const Real coeff = 4 - 6*rsq + 2*square(rsq) - rsq*square(rsq)/6;
//   return coeff * exp(-rsq) / constants::PI;
// }
//
// KOKKOS_INLINE_FUNCTION
// Real bq2fn(const Real& r, const Real eps) {
//   return 1 - exp(-square(r)/square(eps));
// }
//
// KOKKOS_INLINE_FUNCTION
// Real bq8fn(const Real& r, const Real& eps) {
//   const Real rsq = square(r);
//   const Real epssq = square(eps);
//   const Real coeff = (6 - 18*rsq/epssq + 9*square(rsq)/square(epssq) - rsq*square(rsq)/(epssq*square(epssq)))/6;
//   return 1 - coeff * exp(-rsq/epssq);
// }

// TEST_CASE("high order kernels", "[swe]") {
//
//   Comm comm;
//   Logger<> logger("swe high order kernel values test", Log::level::debug, comm);
//
//   constexpr int N = 100;
//   constexpr Real rlim = 4;
//   constexpr Real dr = 2*rlim/N;
//
//   constexpr int n_eps = 4;
//   auto eps_vals = Kokkos::View<Real[n_eps]>("eps_vals");
//   auto h_eps_vals = Kokkos::create_mirror_view(eps_vals);
//   for (int i=0; i<n_eps; ++i) {
//     h_eps_vals(i) = 1/pow(2,i);
//   }
//   Kokkos::deep_copy(eps_vals, h_eps_vals);
//
//   auto b2_test = Kokkos::View<Real*>("b2_test", N+1);
//   auto b2_scaled_test = Kokkos::View<Real**>("b2_scaled_test", N+1, n_eps);
//   auto b8_test = Kokkos::View<Real*>("b8_test", N+1);
//   auto b8_scaled_test = Kokkos::View<Real**>("b8_scaled_test", N+1, n_eps);
//   auto q2_test = Kokkos::View<Real**>("q2_test", N+1, n_eps);
//   auto q8_test = Kokkos::View<Real**>("q8_test", N+1, n_eps);
//
//   auto b2 = Kokkos::View<Real*>("b2", N+1);
//   auto b2_scaled = Kokkos::View<Real**>("b2_scaled", N+1, n_eps);
//   auto b8 = Kokkos::View<Real*>("b8", N+1);
//   auto b8_scaled = Kokkos::View<Real**>("b8_scaled", N+1, n_eps);
//   auto q2 = Kokkos::View<Real**>("q2", N+1, n_eps);
//   auto q8 = Kokkos::View<Real**>("q8", N+1, n_eps);
//
//   scalar_view_type weights("weights", N+1);
//
//   Kokkos::parallel_for(N+1,
//     KOKKOS_LAMBDA (const int i) {
//       const Real rin = -rlim + i*dr;
//       weights(i) = abs(rin) * dr;
//
//       b2(i) = Blob2ndOrderPlane::value(rin);
//       b2_test(i) = b2fn(rin);
//
//       b8(i) = Blob8thOrderPlane::value(rin);
//       b8_test(i) = b8fn(rin);
//
//       const Real x[2] = {rin, 0};
//
//       for (int j=0; j<n_eps; ++j) {
//         b2_scaled_test(i,j) = b2fn(rin/eps_vals(j)) / square(eps_vals(j));
//         b2_scaled(i,j) = Blob2ndOrderPlane::scaled_value(x, eps_vals(j));
//
//         b8_scaled(i,j) = Blob8thOrderPlane::scaled_value(x, eps_vals(j));
//         b8_scaled_test(i,j) = b8fn(rin/eps_vals(j)) / square(eps_vals(j));
//
//         q2(i,j) = Blob2ndOrderPlane::qfn(x, eps_vals(j));
//         q2_test(i,j) = bq2fn(rin, eps_vals(j));
//
//         q8(i,j) = Blob8thOrderPlane::qfn(x, eps_vals(j));
//         q8_test(i,j) = bq8fn(rin, eps_vals(j));
//       }
//     });
//
//   auto b2_error = scalar_view_type("b2_error", N+1);
//   auto b8_error = scalar_view_type("b8_error", N+1);
//   auto b2_scaled_error = Kokkos::View<Real**>("b2_scaled_error", N+1, n_eps);
//   auto b8_scaled_error = Kokkos::View<Real**>("b8_scaled_error", N+1, n_eps);
//
//   ErrNorms b2err(b2_error, b2, b2_test, weights);
//   ErrNorms b8err(b8_error, b8, b8_test, weights);
//
//   logger.info("b2 error info: {}", b2err.info_string());
//   CHECK( FloatingPoint<Real>::zero(b2err.l1));
//   CHECK( FloatingPoint<Real>::zero(b2err.l2));
//   CHECK( FloatingPoint<Real>::zero(b2err.linf));
//
//   logger.info("b8 error info: {}", b8err.info_string());
//   CHECK( FloatingPoint<Real>::zero(b8err.l1));
//   CHECK( FloatingPoint<Real>::zero(b8err.l2));
//   CHECK( FloatingPoint<Real>::zero(b8err.linf));
//
//   std::vector<ErrNorms> b2_scaled_err;
//   std::vector<ErrNorms> b8_scaled_err;
//   for (int i=0; i<n_eps; ++i) {
//     auto err_view2 = Kokkos::subview(b2_scaled_error, Kokkos::ALL, i);
//     auto scaled_view2 = Kokkos::subview(b2_scaled, Kokkos::ALL, i);
//     auto test_view2 = Kokkos::subview(b2_scaled_test, Kokkos::ALL, i);
//     b2_scaled_err.push_back(
//       ErrNorms(err_view2, scaled_view2, test_view2, weights));
//     auto err_view8 = Kokkos::subview(b8_scaled_error, Kokkos::ALL, i);
//     auto scaled_view8 = Kokkos::subview(b8_scaled, Kokkos::ALL, i);
//     auto test_view8 = Kokkos::subview(b8_scaled_test, Kokkos::ALL, i);
//     b8_scaled_err.push_back(
//       ErrNorms(err_view8, scaled_view8, test_view8, weights));
//   }
//   for (int i=0; i<n_eps; ++i) {
//     logger.info("eps = {}, b2_scaled error info: {}", h_eps_vals(i), b2_scaled_err[i].info_string());
//     logger.info("eps = {}, b8_scaled error info: {}", h_eps_vals(i), b8_scaled_err[i].info_string());
//     CHECK( FloatingPoint<Real>::zero(b2_scaled_err[i].l1) );
//     CHECK( FloatingPoint<Real>::zero(b2_scaled_err[i].l2) );
//     CHECK( FloatingPoint<Real>::zero(b2_scaled_err[i].linf) );
//     CHECK( FloatingPoint<Real>::zero(b8_scaled_err[i].l1) );
//     CHECK( FloatingPoint<Real>::zero(b8_scaled_err[i].l2) );
//     CHECK( FloatingPoint<Real>::zero(b8_scaled_err[i].linf) );
//   }
//
//   auto q2_error = Kokkos::View<Real**>("q2_error", N+1, n_eps);
//   auto q8_error = Kokkos::View<Real**>("q8_error", N+1, n_eps);
//   std::vector<ErrNorms> q2err;
//   std::vector<ErrNorms> q8err;
//   for (int i=0; i<n_eps; ++i) {
//     auto err_view2 = Kokkos::subview(q2_error, Kokkos::ALL, i);
//     auto err_view8 = Kokkos::subview(q8_error, Kokkos::ALL, i);
//     auto qview2 = Kokkos::subview(q2, Kokkos::ALL, i);
//     auto qview8 = Kokkos::subview(q8, Kokkos::ALL, i);
//     auto qtest2 = Kokkos::subview(q2_test, Kokkos::ALL, i);
//     auto qtest8 = Kokkos::subview(q8_test, Kokkos::ALL, i);
//     q2err.push_back(
//       ErrNorms(err_view2, qview2, qtest2, weights));
//     q8err.push_back(
//       ErrNorms(err_view8, qview8, qtest8, weights));
//   }
//   for (int i=0; i<n_eps; ++i) {
//     logger.info("eps = {}, q2 err info: {}", h_eps_vals(i), q2err[i].info_string());
//     logger.info("eps = {}, q8 err info: {}", h_eps_vals(i), q8err[i].info_string());
//     CHECK( FloatingPoint<Real>::zero(q2err[i].l1) );
//     CHECK( FloatingPoint<Real>::zero(q2err[i].l2) );
//     CHECK( FloatingPoint<Real>::zero(q2err[i].linf) );
//     CHECK( FloatingPoint<Real>::zero(q8err[i].l1) );
//     CHECK( FloatingPoint<Real>::zero(q8err[i].l2) );
//     CHECK( FloatingPoint<Real>::zero(q8err[i].linf) );
//   }
// }
