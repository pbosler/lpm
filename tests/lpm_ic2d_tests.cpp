#include <catch2/catch_test_macros.hpp>

#include "LpmConfig.h"
#include "lpm_comm.hpp"
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
#include "lpm_tracer_gallery.hpp"
#include "lpm_vorticity_gallery.hpp"
#include "util/lpm_string_util.hpp"
#include "util/lpm_timer.hpp"

using namespace Lpm;

template <typename SeedType, typename SolverType>
struct TimeConvergenceTest {
  using geo = typename SeedType::geo;
  using Coriolis =
      typename std::conditional<std::is_same<geo, PlaneGeometry>::value,
                                CoriolisBetaPlane, CoriolisSphere>::type;
  using Vorticity =
      typename std::conditional<std::is_same<geo, PlaneGeometry>::value,
                                CollidingDipolePairPlane,
                                GaussianVortexSphere>::type;
  using CoordView = typename geo::crd_view_type;

  int tree_depth;
  Real radius;
  Real tfinal;
  std::vector<Real> epss;
  std::vector<Int> nsteps;
  std::string test_name;
  std::vector<Real> dts;
  std::vector<Real> crs;
  CoordView face_position_error;
  Real ref_dt;

  std::unique_ptr<Incompressible2D<SeedType>> reference_sol;
  std::unique_ptr<Incompressible2D<SeedType>> ic2d;

  std::vector<Real> l1;
  std::vector<Real> l2;
  std::vector<Real> linf;
  std::vector<Real> l1rate;
  std::vector<Real> l2rate;
  std::vector<Real> linfrate;

  TimeConvergenceTest(const int depth, const Real r, const Real tf,
                      const std::vector<Real>& epss,
                      const std::vector<int>& nsteps, const std::string& name)
      : tree_depth(depth),
        radius(r),
        tfinal(tf),
        epss(epss),
        nsteps(nsteps),
        test_name(name) {
    setup();
  }

  void run() {
    Comm comm;
    Logger<> logger(test_name, Log::level::debug, comm);
    Coriolis coriolis;
    Vorticity vorticity;
    for (int i = 0; i < nsteps.size() - 1; ++i) {
      const Real dt           = dts[i];
      const int ns            = nsteps[i];
      constexpr int amr_limit = 0;
      PolyMeshParameters<SeedType> mesh_params(tree_depth, radius, amr_limit);
      logger.debug("testing dt = {} with {} steps", dt, ns);
      const Real eps = epss[i];
      ic2d = std::make_unique<Incompressible2D<SeedType>>(mesh_params, coriolis,
                                                          eps);
      ic2d->init_vorticity(vorticity);
      ic2d->init_direct_sums();
      auto solver = std::make_unique<SolverType>(dt, *ic2d);
      for (int t_idx = 0; t_idx < ns; ++t_idx) {
        ic2d->advance_timestep(*solver);
      }
      auto ref_pos = reference_sol->mesh.faces.phys_crds.view;
      auto pos     = ic2d->mesh.faces.phys_crds.view;
      auto face_err_norms =
          ErrNorms(face_position_error, pos, ref_pos, ic2d->mesh.faces.area);
      logger.info("ns = {}, l1 = {}, l2 = {}, linf = {}", ns, face_err_norms.l1,
                  face_err_norms.l2, face_err_norms.linf);
      l1.push_back(face_err_norms.l1);
      l2.push_back(face_err_norms.l2);
      linf.push_back(face_err_norms.linf);
    }
    const auto l1rate   = convergence_rates(dts, l1);
    const auto l2rate   = convergence_rates(dts, l2);
    const auto linfrate = convergence_rates(dts, linf);

    logger.info(convergence_table("dt", dts, "l1", l1, l1rate));
    logger.info(convergence_table("dt", dts, "l2", l2, l2rate));
    logger.info(convergence_table("dt", dts, "linf", linf, linfrate));

    constexpr Real second_order_minimum = 1.95;
    REQUIRE(l1rate.back() > second_order_minimum);
    REQUIRE(l2rate.back() > second_order_minimum);
    REQUIRE(linfrate.back() > second_order_minimum);
  }

 private:
  void setup() {
    const auto setup_start = tic();
    Comm comm;
    Logger<> logger(test_name, Log::level::debug, comm);

    for (int i = 0; i < nsteps.size() - 1; ++i) {
      dts.push_back(tfinal / nsteps[i]);
    }
    ref_dt = tfinal / nsteps[nsteps.size() - 1];
    logger.debug("ref_dt = {} for {} steps", ref_dt, nsteps[nsteps.size() - 1]);

    // compute reference solution
    constexpr int amr_limit = 0;
    PolyMeshParameters<SeedType> mesh_params(tree_depth, radius, amr_limit);
    Coriolis coriolis;
    const Real eps = epss[nsteps.size() - 1];
    reference_sol  = std::make_unique<Incompressible2D<SeedType>>(mesh_params,
                                                                  coriolis, eps);
    Vorticity vorticity;
    reference_sol->init_vorticity(vorticity);
    reference_sol->init_direct_sums();
    const auto vel_range = reference_sol->velocity_active.range(
        reference_sol->mesh.n_faces_host());

    const Real dx = reference_sol->mesh.appx_mesh_size();
    for (int i = 0; i < nsteps.size() - 1; ++i) {
      crs.push_back(vel_range.second * dts[i] / dx);
    }
    crs.push_back(vel_range.second * ref_dt / dx);
    logger.info("testing crs from {} to {}", crs[0], crs[nsteps.size() - 1]);

    auto solver = std::make_unique<SolverType>(ref_dt, *reference_sol);
    for (int t_idx = 0; t_idx < nsteps[nsteps.size() - 1]; ++t_idx) {
      reference_sol->advance_timestep(*solver);
    }
    face_position_error =
        CoordView("face_position_error", reference_sol->mesh.n_faces_host());
    logger.info("setup complete for {}\n\tsetup time: {} seconds",
                SeedType::id_string(), toc(setup_start));
  }
};

// TEST_CASE("plane", "[conv]") {
//   SECTION("triangular panels") {
//     using seed_type = TriHexSeed;
//   }
//   SECTION("quadrilateral panels") {
//     using seed_type = QuadRectSeed;
//   }
// }

TEST_CASE("sphere", "[conv]") {
  constexpr int tree_depth     = 4;
  constexpr Real sphere_radius = 1;
  constexpr Real tfinal        = 0.5;
  //   const std::vector<int> nsteps = {15, 30, 60, 120};
  const std::vector<int> nsteps = {15, 30, 60};

  //   SECTION("triangular panels") {
  //     using seed_type = IcosTriSphereSeed;
  //   }

  SECTION("quadrilateral panels") {
    using seed_type             = CubedSphereSeed;
    using Solver                = Incompressible2DRK2<seed_type>;
    const std::string case_name = "dt_conv_" + seed_type::id_string();

    const std::vector<Real> eps(nsteps.size(), 0);

    TimeConvergenceTest<seed_type, Solver> test_case(
        tree_depth, sphere_radius, tfinal, eps, nsteps, case_name);

    test_case.run();
  }
}
