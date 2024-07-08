#include "LpmConfig.h"
#include "lpm_comm.hpp"
#include "lpm_constants.hpp"
#include "lpm_error.hpp"
#include "lpm_error_impl.hpp"
#include "lpm_high_order_swe.hpp"
#include "lpm_logger.hpp"
#include "lpm_swe.hpp"
#include "lpm_swe_problem_gallery.hpp"
#include "lpm_swe_impl.hpp"
#include "util/lpm_floating_point.hpp"
#include "util/lpm_math.hpp"
#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"
#endif

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using namespace Lpm;
using Catch::Approx;

KOKKOS_INLINE_FUNCTION
Real b2fn(const Real& r) {return exp(-square(r))/constants::PI;}

KOKKOS_INLINE_FUNCTION
Real b8fn(const Real& r) {
  const Real rsq = square(r);
  const Real coeff = 4 - 6*rsq + 2*square(rsq) - rsq*square(rsq)/6;
  return coeff * exp(-rsq) / constants::PI;
}

KOKKOS_INLINE_FUNCTION
Real bq2fn(const Real& r, const Real eps) {
  return 1 - exp(-square(r)/square(eps));
}

KOKKOS_INLINE_FUNCTION
Real bq8fn(const Real& r, const Real& eps) {
  const Real rsq = square(r);
  const Real epssq = square(eps);
  const Real coeff = (6 - 18*rsq/epssq + 9*square(rsq)/square(epssq) - rsq*square(rsq)/(epssq*square(epssq)))/6;
  return 1 - coeff * exp(-rsq/epssq);
}

TEST_CASE("high order kernels", "[swe]") {

  Comm comm;
  Logger<> logger("swe high order kernel values test", Log::level::debug, comm);

  constexpr int N = 100;
  constexpr Real rlim = 4;
  constexpr Real dr = 2*rlim/N;

  constexpr int n_eps = 4;
  auto eps_vals = Kokkos::View<Real[n_eps]>("eps_vals");
  auto h_eps_vals = Kokkos::create_mirror_view(eps_vals);
  for (int i=0; i<n_eps; ++i) {
    h_eps_vals(i) = 1/pow(2,i);
  }
  Kokkos::deep_copy(eps_vals, h_eps_vals);

  auto b2_test = Kokkos::View<Real*>("b2_test", N+1);
  auto b2_scaled_test = Kokkos::View<Real**>("b2_scaled_test", N+1, n_eps);
  auto b8_test = Kokkos::View<Real*>("b8_test", N+1);
  auto b8_scaled_test = Kokkos::View<Real**>("b8_scaled_test", N+1, n_eps);
  auto q2_test = Kokkos::View<Real**>("q2_test", N+1, n_eps);
  auto q8_test = Kokkos::View<Real**>("q8_test", N+1, n_eps);

  auto b2 = Kokkos::View<Real*>("b2", N+1);
  auto b2_scaled = Kokkos::View<Real**>("b2_scaled", N+1, n_eps);
  auto b8 = Kokkos::View<Real*>("b8", N+1);
  auto b8_scaled = Kokkos::View<Real**>("b8_scaled", N+1, n_eps);
  auto q2 = Kokkos::View<Real**>("q2", N+1, n_eps);
  auto q8 = Kokkos::View<Real**>("q8", N+1, n_eps);

  scalar_view_type weights("weights", N+1);

  Kokkos::parallel_for(N+1,
    KOKKOS_LAMBDA (const int i) {
      const Real rin = -rlim + i*dr;
      weights(i) = abs(rin) * dr;

      b2(i) = Blob2ndOrderPlane::value(rin);
      b2_test(i) = b2fn(rin);

      b8(i) = Blob8thOrderPlane::value(rin);
      b8_test(i) = b8fn(rin);

      const Real x[2] = {rin, 0};

      for (int j=0; j<n_eps; ++j) {
        b2_scaled_test(i,j) = b2fn(rin/eps_vals(j)) / square(eps_vals(j));
        b2_scaled(i,j) = Blob2ndOrderPlane::scaled_value(x, eps_vals(j));

        b8_scaled(i,j) = Blob8thOrderPlane::scaled_value(x, eps_vals(j));
        b8_scaled_test(i,j) = b8fn(rin/eps_vals(j)) / square(eps_vals(j));

        q2(i,j) = Blob2ndOrderPlane::qfn(x, eps_vals(j));
        q2_test(i,j) = bq2fn(rin, eps_vals(j));

        q8(i,j) = Blob8thOrderPlane::qfn(x, eps_vals(j));
        q8_test(i,j) = bq8fn(rin, eps_vals(j));
      }
    });

  auto b2_error = scalar_view_type("b2_error", N+1);
  auto b8_error = scalar_view_type("b8_error", N+1);
  auto b2_scaled_error = Kokkos::View<Real**>("b2_scaled_error", N+1, n_eps);
  auto b8_scaled_error = Kokkos::View<Real**>("b8_scaled_error", N+1, n_eps);

  ErrNorms b2err(b2_error, b2, b2_test, weights);
  ErrNorms b8err(b8_error, b8, b8_test, weights);

  logger.info("b2 error info: {}", b2err.info_string());
  CHECK( FloatingPoint<Real>::zero(b2err.l1));
  CHECK( FloatingPoint<Real>::zero(b2err.l2));
  CHECK( FloatingPoint<Real>::zero(b2err.linf));

  logger.info("b8 error info: {}", b8err.info_string());
  CHECK( FloatingPoint<Real>::zero(b8err.l1));
  CHECK( FloatingPoint<Real>::zero(b8err.l2));
  CHECK( FloatingPoint<Real>::zero(b8err.linf));

  std::vector<ErrNorms> b2_scaled_err;
  std::vector<ErrNorms> b8_scaled_err;
  for (int i=0; i<n_eps; ++i) {
    auto err_view2 = Kokkos::subview(b2_scaled_error, Kokkos::ALL, i);
    auto scaled_view2 = Kokkos::subview(b2_scaled, Kokkos::ALL, i);
    auto test_view2 = Kokkos::subview(b2_scaled_test, Kokkos::ALL, i);
    b2_scaled_err.push_back(
      ErrNorms(err_view2, scaled_view2, test_view2, weights));
    auto err_view8 = Kokkos::subview(b8_scaled_error, Kokkos::ALL, i);
    auto scaled_view8 = Kokkos::subview(b8_scaled, Kokkos::ALL, i);
    auto test_view8 = Kokkos::subview(b8_scaled_test, Kokkos::ALL, i);
    b8_scaled_err.push_back(
      ErrNorms(err_view8, scaled_view8, test_view8, weights));
  }
  for (int i=0; i<n_eps; ++i) {
    logger.info("eps = {}, b2_scaled error info: {}", h_eps_vals(i), b2_scaled_err[i].info_string());
    logger.info("eps = {}, b8_scaled error info: {}", h_eps_vals(i), b8_scaled_err[i].info_string());
    CHECK( FloatingPoint<Real>::zero(b2_scaled_err[i].l1) );
    CHECK( FloatingPoint<Real>::zero(b2_scaled_err[i].l2) );
    CHECK( FloatingPoint<Real>::zero(b2_scaled_err[i].linf) );
    CHECK( FloatingPoint<Real>::zero(b8_scaled_err[i].l1) );
    CHECK( FloatingPoint<Real>::zero(b8_scaled_err[i].l2) );
    CHECK( FloatingPoint<Real>::zero(b8_scaled_err[i].linf) );
  }

  auto q2_error = Kokkos::View<Real**>("q2_error", N+1, n_eps);
  auto q8_error = Kokkos::View<Real**>("q8_error", N+1, n_eps);
  std::vector<ErrNorms> q2err;
  std::vector<ErrNorms> q8err;
  for (int i=0; i<n_eps; ++i) {
    auto err_view2 = Kokkos::subview(q2_error, Kokkos::ALL, i);
    auto err_view8 = Kokkos::subview(q8_error, Kokkos::ALL, i);
    auto qview2 = Kokkos::subview(q2, Kokkos::ALL, i);
    auto qview8 = Kokkos::subview(q8, Kokkos::ALL, i);
    auto qtest2 = Kokkos::subview(q2_test, Kokkos::ALL, i);
    auto qtest8 = Kokkos::subview(q8_test, Kokkos::ALL, i);
    q2err.push_back(
      ErrNorms(err_view2, qview2, qtest2, weights));
    q8err.push_back(
      ErrNorms(err_view8, qview8, qtest8, weights));
  }
  for (int i=0; i<n_eps; ++i) {
    logger.info("eps = {}, q2 err info: {}", h_eps_vals(i), q2err[i].info_string());
    logger.info("eps = {}, q8 err info: {}", h_eps_vals(i), q8err[i].info_string());
    CHECK( FloatingPoint<Real>::zero(q2err[i].l1) );
    CHECK( FloatingPoint<Real>::zero(q2err[i].l2) );
    CHECK( FloatingPoint<Real>::zero(q2err[i].linf) );
    CHECK( FloatingPoint<Real>::zero(q8err[i].l1) );
    CHECK( FloatingPoint<Real>::zero(q8err[i].l2) );
    CHECK( FloatingPoint<Real>::zero(q8err[i].linf) );
  }
}
