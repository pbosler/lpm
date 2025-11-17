#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
// #include "lpm_collocated_swe.hpp"
// #include "lpm_collocated_swe_impl.hpp"
#include "lpm_coriolis.hpp"
#include "lpm_error.hpp"
#include "lpm_error_impl.hpp"
#include "lpm_logger.hpp"
#include "lpm_regularized_kernels_1d.hpp"
#include "lpm_regularized_kernels_2d.hpp"
#include "lpm_swe_problem_gallery.hpp"
#include "lpm_vorticity_gallery.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#include "mesh/lpm_polymesh2d_impl.hpp"
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"
#include "util/lpm_matlab_io.hpp"
#include "util/lpm_string_util.hpp"
#include "util/lpm_test_utils.hpp"

#include <KokkosBlas.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using namespace Lpm;
using namespace Lpm::colloc;
using Catch::Approx;

KOKKOS_INLINE_FUNCTION
Real test_gaussian(const Real x) {
  constexpr Real sigma = 0.05;
  const Real denom = sqrt(constants::PI * square(sigma));
  return exp(-square(x)/square(sigma)) / denom;
}

KOKKOS_INLINE_FUNCTION
Real test_gaussian_2nd(const Real x) {
  constexpr Real sigma = 0.05;
  const Real xsq = square(x);
  const Real ssq = square(sigma);
  const Real denom = sqrt(constants::PI * pow(sigma,5));
  return -2 * exp(-xsq / ssq) * (-2*xsq + ssq ) / denom;
}

KOKKOS_INLINE_FUNCTION
Real test_packet(const Real x, const Real k0) {
  constexpr Real sigma = 0.1;
  const Real denom = 1; //sqrt(constants::PI * square(sigma));
  return exp(-square(x)/square(sigma))*cos(k0 * x)/denom;
}

TEST_CASE("planar gaussians", "[pse]") {
  using Seed = QuadRectSeed; // TriHexSeed;

  Comm comm;
  const std::string log_name = "eldrege2002b_planar_gaussians";
  Logger<> logger(log_name, Log::level::debug, comm);
  logger.debug("eldredge planar gaussians started.");

  constexpr Real r = 1;
  constexpr int min_depth = 3;
  constexpr int max_depth = 6;
  constexpr int amr_limit = 0;
  constexpr Real eps_pow = 0.85;

  constexpr Real sigmasq = 0.01;
  constexpr Real x0 = 0.15;
  constexpr Real y0 = 0.15;
  PlanarGaussian gaussian(1/(constants::PI * sigmasq), sigmasq, x0, y0);
  PlanarNegativeLaplacianOfGaussian neg_laplacian(gaussian);

  std::vector<Real> dxs;
  std::vector<Real> interp2_l2;
  std::vector<Real> interp4_l2;
  std::vector<Real> interp6_l2;
  std::vector<Real> interp8_l2;
  std::vector<Real> lap2_l2;
  std::vector<Real> lap4_l2;
  std::vector<Real> lap6_l2;
  std::vector<Real> lap8_l2;

  for (int d=min_depth; d<= max_depth; ++d) {
    PolyMeshParameters<Seed> mesh_params(d, r, amr_limit);
    PolyMesh2d<Seed> mesh(mesh_params);
    const Real dx = mesh.appx_mesh_size();
    dxs.push_back(dx);
    const Real eps = pow(dx, eps_pow);

    logger.info("dx = {}, eps = {}, eps/dx = {}", dx, eps, eps/dx);

    scalar_view_type gauss_pulse("gaussian", mesh.n_faces_host());
    scalar_view_type gauss_interp2("gauss_interp2", mesh.n_faces_host());
    scalar_view_type gauss_interp4("gauss_interp4", mesh.n_faces_host());
    scalar_view_type gauss_interp6("gauss_interp6", mesh.n_faces_host());
    scalar_view_type gauss_interp8("gauss_interp8", mesh.n_faces_host());
    scalar_view_type gauss_error2("gauss_error2", mesh.n_faces_host());
    scalar_view_type gauss_error4("gauss_error4", mesh.n_faces_host());
    scalar_view_type gauss_error6("gauss_error6", mesh.n_faces_host());
    scalar_view_type gauss_error8("gauss_error8", mesh.n_faces_host());
//     scalar_view_type gauss_dx0("gauss_dx0", mesh.n_faces_host());
//     scalar_view_type gauss_dx0_exact("gauss_dx0_exact", mesh.n_faces_host());
//     scalar_view_type gauss_dx0_error("gauss_dx0_error", mesh.n_faces_host());
//     scalar_view_type gauss_dx1("gauss_dx1", mesh.n_faces_host());
//     scalar_view_type gauss_dx1_exact("gauss_dx1_exact", mesh.n_faces_host());
//     scalar_view_type gauss_dx1_error("gauss_dx1_error", mesh.n_faces_host());
    scalar_view_type laplacian2("laplacian2", mesh.n_faces_host());
    scalar_view_type laplacian4("laplacian4", mesh.n_faces_host());
    scalar_view_type laplacian6("laplacian6", mesh.n_faces_host());
    scalar_view_type laplacian8("laplacian8", mesh.n_faces_host());
    scalar_view_type laplacian_exact("laplacian_exact", mesh.n_faces_host());
    scalar_view_type laplacian_error2("laplacian_error2", mesh.n_faces_host());
    scalar_view_type laplacian_error4("laplacian_error4", mesh.n_faces_host());
    scalar_view_type laplacian_error6("laplacian_error6", mesh.n_faces_host());
    scalar_view_type laplacian_error8("laplacian_error8", mesh.n_faces_host());

    Kokkos::parallel_for(mesh.n_faces_host(),
      KOKKOS_LAMBDA (const Index i) {
        const auto x_i = Kokkos::subview(mesh.faces.phys_crds.view, i, Kokkos::ALL);
        gauss_pulse(i) = gaussian(x_i);
        laplacian_exact(i) = - neg_laplacian(x_i);
      });

    const Plane2ndOrder kernels2(eps);
    const Plane4thOrder kernels4(eps);
    const Plane6thOrder kernels6(eps);
    const Plane8thOrder kernels8(eps);

    Kokkos::TeamPolicy<> thread_team(mesh.n_faces_host(), Kokkos::AUTO());

    Kokkos::parallel_for(thread_team,
      DirectSum<PlaneScalarInterpolationReducer<Plane2ndOrder>>(
        gauss_interp2, mesh.faces.phys_crds.view, gauss_pulse,
        mesh.faces.phys_crds.view, gauss_pulse,
        kernels2, mesh.faces.area, mesh.faces.mask, mesh.n_faces_host()));

    Kokkos::parallel_for(thread_team,
      DirectSum<PlaneScalarInterpolationReducer<Plane4thOrder>>(
        gauss_interp4, mesh.faces.phys_crds.view, gauss_pulse,
        mesh.faces.phys_crds.view, gauss_pulse,
        kernels4, mesh.faces.area, mesh.faces.mask, mesh.n_faces_host()));

    Kokkos::parallel_for(thread_team,
      DirectSum<PlaneScalarInterpolationReducer<Plane6thOrder>>(
        gauss_interp6, mesh.faces.phys_crds.view, gauss_pulse,
        mesh.faces.phys_crds.view, gauss_pulse,
        kernels6, mesh.faces.area, mesh.faces.mask, mesh.n_faces_host()));

    Kokkos::parallel_for(thread_team,
      DirectSum<PlaneScalarInterpolationReducer<Plane8thOrder>>(
        gauss_interp8, mesh.faces.phys_crds.view, gauss_pulse,
        mesh.faces.phys_crds.view, gauss_pulse,
        kernels8, mesh.faces.area, mesh.faces.mask, mesh.n_faces_host()));

    Kokkos::parallel_for(thread_team,
      DirectSum<PlaneLaplacianReducer<Plane2ndOrder>>(
        laplacian2, mesh.faces.phys_crds.view, gauss_pulse,
        mesh.faces.phys_crds.view, gauss_pulse,
        kernels2, mesh.faces.area, mesh.faces.mask, mesh.n_faces_host()));

    Kokkos::parallel_for(thread_team,
      DirectSum<PlaneLaplacianReducer<Plane4thOrder>>(
        laplacian4, mesh.faces.phys_crds.view, gauss_pulse,
        mesh.faces.phys_crds.view, gauss_pulse,
        kernels4, mesh.faces.area, mesh.faces.mask, mesh.n_faces_host()));

    Kokkos::parallel_for(thread_team,
      DirectSum<PlaneLaplacianReducer<Plane6thOrder>>(
        laplacian6, mesh.faces.phys_crds.view, gauss_pulse,
        mesh.faces.phys_crds.view, gauss_pulse,
        kernels6, mesh.faces.area, mesh.faces.mask, mesh.n_faces_host()));

    Kokkos::parallel_for(thread_team,
      DirectSum<PlaneLaplacianReducer<Plane8thOrder>>(
        laplacian8, mesh.faces.phys_crds.view, gauss_pulse,
        mesh.faces.phys_crds.view, gauss_pulse,
        kernels8, mesh.faces.area, mesh.faces.mask, mesh.n_faces_host()));

    const ErrNorms interp2_err(gauss_error2, gauss_interp2, gauss_pulse, mesh.faces.area);
    const ErrNorms interp4_err(gauss_error4, gauss_interp4, gauss_pulse, mesh.faces.area);
    const ErrNorms interp6_err(gauss_error6, gauss_interp6, gauss_pulse, mesh.faces.area);
    const ErrNorms interp8_err(gauss_error8, gauss_interp8, gauss_pulse, mesh.faces.area);

    const ErrNorms lap2_err(laplacian_error2, laplacian2, laplacian_exact, mesh.faces.area);
    const ErrNorms lap4_err(laplacian_error4, laplacian4, laplacian_exact, mesh.faces.area);
    const ErrNorms lap6_err(laplacian_error6, laplacian6, laplacian_exact, mesh.faces.area);
    const ErrNorms lap8_err(laplacian_error8, laplacian8, laplacian_exact, mesh.faces.area);

    interp2_l2.push_back(interp2_err.l2);
    interp4_l2.push_back(interp4_err.l2);
    interp6_l2.push_back(interp6_err.l2);
    interp8_l2.push_back(interp8_err.l2);

    lap2_l2.push_back(lap2_err.l2);
    lap4_l2.push_back(lap4_err.l2);
    lap6_l2.push_back(lap6_err.l2);
    lap8_l2.push_back(lap8_err.l2);

  }

  const auto interp_rate2 = convergence_rates(dxs, interp2_l2);
  const auto interp_rate4 = convergence_rates(dxs, interp4_l2);
  const auto interp_rate6 = convergence_rates(dxs, interp6_l2);
  const auto interp_rate8 = convergence_rates(dxs, interp8_l2);

  const auto lap_rate2 = convergence_rates(dxs, lap2_l2);
  const auto lap_rate4 = convergence_rates(dxs, lap4_l2);
  const auto lap_rate6 = convergence_rates(dxs, lap6_l2);
  const auto lap_rate8 = convergence_rates(dxs, lap8_l2);

  logger.info(convergence_table("dx", dxs, "interp2_l2", interp2_l2, interp_rate2));
  logger.info(convergence_table("dx", dxs, "interp4_l2", interp4_l2, interp_rate4));
  logger.info(convergence_table("dx", dxs, "interp6_l2", interp6_l2, interp_rate6));
  logger.info(convergence_table("dx", dxs, "interp8_l2", interp8_l2, interp_rate8));

  logger.info(convergence_table("dx", dxs, "lap2_l2", lap2_l2, lap_rate2));
  logger.info(convergence_table("dx", dxs, "lap4_l2", lap4_l2, lap_rate4));
  logger.info(convergence_table("dx", dxs, "lap6_l2", lap6_l2, lap_rate6));
  logger.info(convergence_table("dx", dxs, "lap8_l2", lap8_l2, lap_rate8));
}

// TEST_CASE("figure 1", "[pse]") {
// }



TEST_CASE("figure 2", "[pse]") {
  Comm comm;
  const std::string log_name = "eldrege2002b_figure2";
  Logger<> logger(log_name, Log::level::debug, comm);
  logger.debug("eldredge figure 2 started.");

  constexpr Real xmin = -constants::PI;
  constexpr Real xmax = constants::PI;
  Real eps_pow = 0.85;
  const std::vector<int> nx = {50, 100, 200, 400};
  constexpr Real tfinal = 1.5;

  std::ofstream ofile("lpm_eldredge_2002b_fig2.m");

  for (int i=0; i<nx.size(); ++i) {
    logger.debug("i = {}, n = {}", i, nx[i]);
    const int n = nx[i];
    const Real dx = (xmax - xmin)/n;
    const Real eps = pow(dx, eps_pow);
    const Real k0 = 0.32 * constants::PI / dx;
    scalar_view_type x("x", n);
    scalar_view_type length("length",n);
    scalar_view_type f0("f0", n);
    scalar_view_type f_low("f_low", n);
    scalar_view_type f_high("f_high", n);
    scalar_view_type dfdx_low("dfdx_low", n);
    scalar_view_type dfdx_high("dfdx_high", n);
    Kokkos::parallel_for(n,
      KOKKOS_LAMBDA (const int j) {
        x(j) = xmin + (j+0.5)*dx;
        length(j) = dx;
        f0(j) = test_packet(x(j), k0);
        f_low(j) = test_packet(x(j), k0);
        f_high(j) = test_packet(x(j), k0);
      });

    Real dt = tfinal / n;
    bool keep_going = (dt > eps);
    while (keep_going) {
      dt *= 0.5;
      keep_going = (dt > eps);
      logger.debug("dt = {}", dt);
    }
    const int nsteps = tfinal / dt;
    logger.info("n = {}, dx = {}, dt = {}, eps = {}, cr(dx) = {}, cr(eps) = {}, nsteps = {}",
       n, dx, dt, eps, dt/dx, eps/dx, nsteps);
    Real total_length;
    Kokkos::parallel_reduce(n,
      KOKKOS_LAMBDA (const Index j, Real& s) {
        s += length(j);
      }, total_length);
    logger.debug("total_length = {}", total_length);
    CHECK( total_length == Approx(2*constants::PI) );

    Kokkos::TeamPolicy<> thread_team(n, Kokkos::AUTO);

    Line2ndOrder kernels2(eps);
    Line8thOrder kernels8(eps);
    for (int m=0; m<nsteps; ++m) {

      Kokkos::parallel_for(thread_team,
        LineDirectSum<LineXDerivativeReducer<Line2ndOrder>>(dfdx_low, x, f_low, kernels2, length, n));
      Kokkos::parallel_for(thread_team,
        LineDirectSum<LineXDerivativeReducer<Line8thOrder>>(dfdx_high, x, f_high, kernels8, length, n));

      KokkosBlas::axpy(-dt, dfdx_low, f_low);
      KokkosBlas::axpy(-dt, dfdx_high, f_high);
    }

    auto h_x = Kokkos::create_mirror_view(x);
    auto h_f0 = Kokkos::create_mirror_view(f0);
    auto h_f_low = Kokkos::create_mirror_view(f_low);
    auto h_f_high = Kokkos::create_mirror_view(f_high);

    Kokkos::deep_copy(h_x, x);
    Kokkos::deep_copy(h_f0, f0);
    Kokkos::deep_copy(h_f_low, f_low);
    Kokkos::deep_copy(h_f_high, f_high);

    const std::string n_str = "_n" + std::to_string(n);
    write_vector_matlab(ofile, "x"+n_str, h_x);
    write_vector_matlab(ofile, "f0"+n_str, h_f0);
    write_vector_matlab(ofile, "f_low"+n_str, h_f_low);
    write_vector_matlab(ofile, "f_high"+n_str, h_f_high);
  }
  ofile.close();
}

TEST_CASE("figure 4", "[pse]") {
  Comm comm;
  const std::string log_name = "eldrege2002b_figure4";
  Logger<> logger(log_name, Log::level::debug, comm);
  logger.debug("eldredge figure 4 started.");

  std::vector<int> nx;
  std::vector<Real> dxs;
  std::vector<Real> l2_2;
  std::vector<Real> l2_4;
  std::vector<Real> l2_6;
  std::vector<Real> l2_8;

  std::ofstream ofile("lpm_eldredge_2002b_fig4.m");

  for (int i=1; i<=10; ++i) {
    const int n = i*100;
    nx.push_back(n);

    const Real dx = 1.0/n;
    dxs.push_back(dx);
    const Real eps = 2 * dx ;
//     const Real eps = pow(dx, 0.75);
    scalar_view_type x("x", n);
    scalar_view_type f("f", n);
    scalar_view_type length("length", n);
    scalar_view_type df2_exact("df2_dx2", n);
    scalar_view_type df2_2("df2_2", n);
    scalar_view_type df2_4("df2_4", n);
    scalar_view_type df2_6("df2_6", n);
    scalar_view_type df2_8("df2_8", n);
    scalar_view_type error2("error2", n);
    scalar_view_type error4("error4", n);
    scalar_view_type error6("error6", n);
    scalar_view_type error8("error8", n);
    Line2ndOrder kernels2(eps);
    Line4thOrder kernels4(eps);
    Line6thOrder kernels6(eps);
    Line8thOrder kernels8(eps);
    Kokkos::parallel_for(n,
      KOKKOS_LAMBDA (const int j) {
        x(j) = -0.5 + (j+0.5)*dx;
        f(j) = test_gaussian(x(j));
        df2_exact(j) = test_gaussian_2nd(x(j));
        length(j) = dx;
      });
    logger.info("dx = {}, eps = {}, eps/dx = {}", dx, eps, eps/dx);
    Real total_length;
    Kokkos::parallel_reduce(n,
      KOKKOS_LAMBDA (const Index j, Real& s) {
        s += length(j);
      }, total_length);
    logger.debug("total_length = {}", total_length);
    auto h_x = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(h_x, x);
    CHECK( total_length == Approx(1) );
    CHECK( h_x(0) ==  Approx(-0.5 + 0.5*dx));
    CHECK( h_x(n-1) == Approx(0.5 - 0.5*dx));

    Kokkos::TeamPolicy<> thread_team(n, Kokkos::AUTO);
    Kokkos::parallel_for(thread_team,
      LineDirectSum<Line2ndDerivativeReducer<Line2ndOrder>>(
        df2_2, x, f, kernels2, length, n));
    Kokkos::parallel_for(thread_team,
      LineDirectSum<Line2ndDerivativeReducer<Line4thOrder>>(
        df2_4, x, f, kernels4, length, n));
    Kokkos::parallel_for(thread_team,
      LineDirectSum<Line2ndDerivativeReducer<Line6thOrder>>(
        df2_6, x, f, kernels6, length, n));
    Kokkos::parallel_for(thread_team,
      LineDirectSum<Line2ndDerivativeReducer<Line8thOrder>>(
        df2_8, x, f, kernels8, length, n));

    ErrNorms err2(error2, df2_2, df2_exact, length);
    ErrNorms err4(error4, df2_4, df2_exact, length);
    ErrNorms err6(error6, df2_6, df2_exact, length);
    ErrNorms err8(error8, df2_8, df2_exact, length);

    l2_2.push_back(err2.l2);
    l2_4.push_back(err4.l2);
    l2_6.push_back(err6.l2);
    l2_8.push_back(err8.l2);


    auto h_f = Kokkos::create_mirror_view(f);
    Kokkos::deep_copy(h_f, f);
    auto h_df2_exact = Kokkos::create_mirror_view(df2_exact);
    Kokkos::deep_copy(h_df2_exact, df2_exact);
    auto h_df2_2 = Kokkos::create_mirror_view(df2_2);
    Kokkos::deep_copy(h_df2_2, df2_2);
    auto h_df2_4 = Kokkos::create_mirror_view(df2_4);
    Kokkos::deep_copy(h_df2_4, df2_4);
    auto h_df2_6 = Kokkos::create_mirror_view(df2_6);
    Kokkos::deep_copy(h_df2_6, df2_6);
    auto h_df2_8 = Kokkos::create_mirror_view(df2_8);
    Kokkos::deep_copy(h_df2_8, df2_8);

    const std::string n_str = "_n" + std::to_string(n);
    write_vector_matlab(ofile, "x" + n_str, h_x);
    write_vector_matlab(ofile, "f" + n_str, h_f);
    write_vector_matlab(ofile, "df2_exact" + n_str, h_df2_exact);
    write_vector_matlab(ofile, "df2_2" + n_str, h_df2_2);
    write_vector_matlab(ofile, "df2_4" + n_str, h_df2_4);
    write_vector_matlab(ofile, "df2_6" + n_str, h_df2_6);
    write_vector_matlab(ofile, "df2_8" + n_str, h_df2_8);
  }
  const auto rate2 = convergence_rates(dxs, l2_2);
  const auto rate4 = convergence_rates(dxs, l2_4);
  const auto rate6 = convergence_rates(dxs, l2_6);
  const auto rate8 = convergence_rates(dxs, l2_8);

  logger.info(convergence_table("dx", dxs, "l2_2", l2_2, rate2));
  logger.info(convergence_table("dx", dxs, "l2_4", l2_4, rate4));
  logger.info(convergence_table("dx", dxs, "l2_6", l2_6, rate6));
  logger.info(convergence_table("dx", dxs, "l2_8", l2_8, rate8));

  ofile.close();
}
