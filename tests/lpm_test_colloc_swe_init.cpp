#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
#include "lpm_collocated_swe.hpp"
#include "lpm_collocated_swe_impl.hpp"
#include "lpm_coriolis.hpp"
#include "lpm_error.hpp"
#include "lpm_error_impl.hpp"
#include "lpm_surface_gallery.hpp"
#include "lpm_swe_problem_gallery.hpp"
#include "lpm_vorticity_gallery.hpp"
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"
#include "util/lpm_string_util.hpp"
#include "util/lpm_test_utils.hpp"

#include <catch2/catch_test_macros.hpp>

using namespace Lpm;
using namespace Lpm::colloc;

template <typename SeedType, typename KernelType>
struct TestCollocSweInit{
  using Topography  = ZeroFunctor;
  using InitialSurface = UniformDepthSurface;
  using Coriolis = CoriolisBetaPlane;
  using Geo = PlaneGeometry;
  using StreamFunction = PlanarGaussian;
  using PotentialFunction = PlanarGaussian;
  using Vorticity = PlanarNegativeLaplacianOfGaussian;
  using Divergence = PlanarNegativeLaplacianOfGaussian;
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

  Int tree_depth;
  Int amr_limit;
  std::string vtk_file;
  Real eps_power;

  TestCollocSweInit(const Int depth, const Int amr, const Real power) :
    tree_depth(depth),
    amr_limit(amr),
    eps_power(power)
    {
    vtk_file =
      "colloc_swe_" + SeedType::id_string() + "_ord" + std::to_string(KernelType::order) + "_pow" + std::to_string(eps_power);
    }

  void run() {

    Comm comm;
    Logger<> logger(vtk_file, Log::level::debug, comm);
    logger.debug("test run called.");

    PolyMeshParameters<SeedType> mesh_params(tree_depth, radius, amr_limit);
    Coriolis coriolis;

    auto plane = std::make_unique<CollocatedSWE<SeedType>>(mesh_params, coriolis);
    logger.debug(plane->mesh.info_string());

    Topography topo;
    InitialSurface sfc;
    StreamFunction psi(zeta0, zetab, zetax, zetay);
    PotentialFunction phi(delta0, deltab, deltax, deltay);
    Vorticity zeta(psi);
    Divergence delta(phi);
    plane->set_kernel_width_from_power(eps_power);
    KernelType kernels(plane->eps);
    plane->init_swe_problem(topo, sfc, zeta, delta, kernels);

    Kokkos::View<Real*[2]> velocity_error_active("velocity_error", plane->mesh.n_faces_host());
    Kokkos::View<Real*[2]> velocity_error_passive("velocity_error", plane->mesh.n_vertices_host());
    Kokkos::View<Real*[2]> velocity_exact_active("velocity_exact", plane->mesh.n_faces_host());
    Kokkos::View<Real*[2]> velocity_exact_passive("velocity_exact", plane->mesh.n_vertices_host());
    plane->allocate_scalar_tracer("double_dot_error");
    plane->allocate_scalar_tracer("double_dot_exact");
    plane->allocate_scalar_tracer("du1dx1_exact");
    plane->allocate_scalar_tracer("du1dx2_exact");
    plane->allocate_scalar_tracer("du2dx1_exact");
    plane->allocate_scalar_tracer("du2dx2_exact");
    plane->allocate_scalar_tracer("du1dx1_error");
    plane->allocate_scalar_tracer("du1dx2_error");
    plane->allocate_scalar_tracer("du2dx1_error");
    plane->allocate_scalar_tracer("du2dx2_error");

    scalar_view_type dummy_view("dummy", plane->mesh.n_vertices_host());

    Kokkos::parallel_for(plane->mesh.n_faces_host(),
      PlanarGaussianTestVelocity(velocity_exact_active,
      plane->tracers.at("double_dot_exact").view,
      plane->tracers.at("du1dx1_exact").view,
      plane->tracers.at("du1dx2_exact").view,
      plane->tracers.at("du2dx1_exact").view,
      plane->tracers.at("du2dx2_exact").view,
      plane->mesh.faces.phys_crds.view,
      zeta, delta));
    Kokkos::parallel_for(plane->mesh.n_vertices_host(),
      PlanarGaussianTestVelocity(velocity_exact_passive,
      dummy_view,
      dummy_view,
      dummy_view,
      dummy_view,
      dummy_view,
      plane->mesh.vertices.phys_crds.view,
      zeta, delta));
    logger.debug("fields initialized; computing error.");

    compute_error(velocity_error_passive, plane->velocity_passive.view, velocity_exact_passive);
    ErrNorms vel_err(velocity_error_active, plane->velocity_active.view, velocity_exact_active,
      plane->mesh.faces.area);
    logger.info("velocity error: {}", vel_err.info_string());
    ErrNorms ddot_err(plane->tracers.at("double_dot_error").view, plane->double_dot.view, plane->tracers.at("double_dot_exact").view,
      plane->mesh.faces.area);
    logger.info("double dot error: {}", ddot_err.info_string());

    ErrNorms d11_err(plane->tracers.at("du1dx1_error").view, plane->du1dx1.view, plane->tracers.at("du1dx1_exact").view,
      plane->mesh.faces.area);
    ErrNorms d12_err(plane->tracers.at("du1dx2_error").view, plane->du1dx1.view, plane->tracers.at("du1dx2_exact").view,
      plane->mesh.faces.area);
    ErrNorms d21_err(plane->tracers.at("du2dx1_error").view, plane->du1dx1.view, plane->tracers.at("du2dx1_exact").view,
      plane->mesh.faces.area);
    ErrNorms d22_err(plane->tracers.at("du2dx2_error").view, plane->du1dx1.view, plane->tracers.at("du2dx2_exact").view,
      plane->mesh.faces.area);
    logger.info("du1dx1 error: {}", d11_err.info_string());
    logger.info("du1dx2 error: {}", d12_err.info_string());
    logger.info("du2dx1 error: {}", d21_err.info_string());
    logger.info("du2dx2 error: {}", d22_err.info_string());


    plane->update_host();
    auto vtk = vtk_mesh_interface(*plane);
    vtk.add_vector_point_data(velocity_exact_passive);
    vtk.add_vector_point_data(velocity_error_passive);
    vtk.add_vector_cell_data(velocity_exact_active);
    vtk.add_vector_cell_data(velocity_error_active);
    vtk.write(vtk_file + vtp_suffix());
  }
};

TEST_CASE("colloc_swe_init", "") {
  int depth = 5;
  int amr = 0;
  Real eps_power = 0.85;

  const int start_depth = 2;
  int end_depth = 4;

  auto& ts = TestSession::get();
  if (ts.params.find("eps-pow") != ts.params.end()) {
    eps_power = std::stod(ts.params["eps-pow"]);
  }
  if (ts.params.find("depth") != ts.params.end()) {
    depth = std::stoi(ts.params["depth"]);
  }
  for (const auto& p : ts.params) {
    std::cout << p.first << " = " << p.second << "\n";
  }

  TestCollocSweInit<QuadRectSeed, Plane2ndOrder> swe_test2(depth, amr, eps_power);
  swe_test2.run();
  TestCollocSweInit<QuadRectSeed, Plane4thOrder> swe_test4(depth, amr, eps_power);
  swe_test4.run();
  TestCollocSweInit<QuadRectSeed, Plane6thOrder> swe_test6(depth, amr, eps_power);
  swe_test6.run();
  TestCollocSweInit<QuadRectSeed, Plane8thOrder> swe_test8(depth, amr, eps_power);
  swe_test8.run();
}
