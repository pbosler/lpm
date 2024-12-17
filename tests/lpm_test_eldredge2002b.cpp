#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
// #include "lpm_collocated_swe.hpp"
// #include "lpm_collocated_swe_impl.hpp"
#include "lpm_coriolis.hpp"
#include "lpm_error.hpp"
#include "lpm_error_impl.hpp"
#include "lpm_logger.hpp"
#include "lpm_regularized_kernels_1d.hpp"
#include "lpm_surface_gallery.hpp"
#include "lpm_swe_problem_gallery.hpp"
#include "lpm_vorticity_gallery.hpp"
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"
#include "util/lpm_matlab_io.hpp"
#include "util/lpm_string_util.hpp"
#include "util/lpm_test_utils.hpp"

#include <KokkosBlas.hpp>

#include <catch2/catch_test_macros.hpp>

using namespace Lpm;
using namespace Lpm::colloc;

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
  const Real denom = sqrt(constants::PI * pow(sigma,5));
  return -2 * exp(-square(x)/square(sigma)) * (-2*xsq + square(sigma) ) / denom;
}

KOKKOS_INLINE_FUNCTION
Real test_packet(const Real x, const Real k0) {
  constexpr Real sigma = 0.1;
  const Real denom = 1; //sqrt(constants::PI * square(sigma));
  return exp(-square(x)/square(sigma))*cos(k0 * x)/denom;
}

TEST_CASE("figure 1", "[pse]") {
}



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
    scalar_view_type x("x", n+1);
    Kokkos::parallel_for(n+1,
      KOKKOS_LAMBDA (const int j) {
        x(j) = xmin + j*dx;
      });
    scalar_view_type f0("f0", n+1);
    scalar_view_type f_low("f_low", n+1);
    scalar_view_type f_high("f_high", n+1);
    scalar_view_type dfdx_low("dfdx_low", n+1);
    scalar_view_type dfdx_high("dfdx_high", n+1);
    scalar_view_type length("length",n+1);
    Kokkos::deep_copy(length, dx);
    Kokkos::parallel_for(n+1,
      KOKKOS_LAMBDA (const int j) {
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


    Kokkos::TeamPolicy<> thread_team(n+1, Kokkos::AUTO);

    Line2ndOrder kernels2(eps);
    Line8thOrder kernels8(eps);
    for (int m=0; m<nsteps; ++m) {

      Kokkos::parallel_for(thread_team,
        LineDirectSum<LineXDerivativeReducer<Line2ndOrder>>(dfdx_low, x, f_low, kernels2, length, n+1));
      Kokkos::parallel_for(thread_team,
        LineDirectSum<LineXDerivativeReducer<Line8thOrder>>(dfdx_high, x, f_high, kernels8, length, n+1));

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
    const Real eps = 2*dx;
    scalar_view_type x("x", n+1);
    scalar_view_type f("f", n+1);
    scalar_view_type length("length", n+1);
    scalar_view_type df2_exact("df2_dx2", n+1);
    scalar_view_type df2_2("df2_2", n+1);
    scalar_view_type df2_4("df2_4", n+1);
    scalar_view_type df2_6("df2_6", n+1);
    scalar_view_type df2_8("df2_8", n+1);
    scalar_view_type error2("error2", n+1);
    scalar_view_type error4("error4", n+1);
    scalar_view_type error6("error6", n+1);
    scalar_view_type error8("error8", n+1);
    Line2ndOrder kernels2(eps);
    Line4thOrder kernels4(eps);
    Line6thOrder kernels6(eps);
    Line8thOrder kernels8(eps);
    Kokkos::parallel_for(n+1,
      KOKKOS_LAMBDA (const int j) {
        x(j) = -0.5 + j*dx;
        f(j) = test_gaussian(x(j));
        df2_exact(j) = test_gaussian_2nd(x(j));
      });
    Kokkos::deep_copy(length, dx);
    Kokkos::TeamPolicy<> thread_team(n+1, Kokkos::AUTO);
    Kokkos::parallel_for(thread_team,
      LineDirectSum<Line2ndDerivativeReducer<Line2ndOrder>>(
        df2_2, x, f, kernels2, length, n+1));
    Kokkos::parallel_for(thread_team,
      LineDirectSum<Line2ndDerivativeReducer<Line4thOrder>>(
        df2_4, x, f, kernels4, length, n+1));
    Kokkos::parallel_for(thread_team,
      LineDirectSum<Line2ndDerivativeReducer<Line6thOrder>>(
        df2_6, x, f, kernels6, length, n+1));
    Kokkos::parallel_for(thread_team,
      LineDirectSum<Line2ndDerivativeReducer<Line8thOrder>>(
        df2_8, x, f, kernels8, length, n+1));

    ErrNorms err2(error2, df2_2, df2_exact, length);
    ErrNorms err4(error4, df2_4, df2_exact, length);
    ErrNorms err6(error6, df2_6, df2_exact, length);
    ErrNorms err8(error8, df2_8, df2_exact, length);

    l2_2.push_back(err2.l2);
    l2_4.push_back(err4.l2);
    l2_6.push_back(err6.l2);
    l2_8.push_back(err8.l2);

    auto h_x = Kokkos::create_mirror_view(x);
    Kokkos::deep_copy(h_x, x);
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
