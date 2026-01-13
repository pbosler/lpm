#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <string>
#include <typeinfo>

#include "LpmConfig.h"
#include "lpm_comm.hpp"
#include "lpm_constants.hpp"
#include "lpm_geometry.hpp"
#include "lpm_logger.hpp"
#include "util/lpm_floating_point.hpp"
#include "util/lpm_math.hpp"

using namespace Lpm;

TEST_CASE("lpm_math", "") {
  Comm comm;

  Logger<> logger("math_test_log", Log::level::info, comm);
  logger.info("atan4(0,0) = {}, expected: 0", atan4(0, 0));
  CHECK(FloatingPoint<Real>::zero(atan4(0, 0)));

  const Real a   = 1;
  const Real b   = -3;
  const Real c   = 2;
  const Real d   = 4;
  const Real det = two_by_two_determinant(a, b, c, d);
  logger.info("2x2 determinant = {}; expected: 10", det);
  CHECK(Lpm::FloatingPoint<Real>::equiv(det, 10));

  Real r1, r2;
  quadratic_roots(r1, r2, a, b, c);
  logger.info("quadratic root 1 = {}; expected: 2", r1);
  logger.info("quadratic root 2 = {}; expected: 1", r2);
  CHECK(FloatingPoint<Real>::equiv(r1, 2));
  CHECK(FloatingPoint<Real>::equiv(r2, 1));

  constexpr int nlon     = 180;
  constexpr int nlat     = 91;
  constexpr Real dlambda = 2 * constants::PI / nlon;
  Real np[3];
  Real ones_sph[3];
  for (int i = 0; i < 3; ++i) {
    ones_sph[i] = 1 / sqrt(3.0);
  }

  Real xyz[3];
  Kokkos::Tuple<Real, 9> proj_mat;
  Kokkos::Tuple<Real, 9> np_mat;
  constexpr Real fp_tol = 1e-15;

  if (fp_tol > FloatingPoint<Real>::zero_tol) {
    logger.warn(
        "Using floating point tolerance of {}, rather than the default value "
        "{}; this fp_tol is {}x larger.",
        fp_tol, FloatingPoint<Real>::zero_tol,
        fp_tol / FloatingPoint<Real>::zero_tol);
  }

  for (int i = 0; i < nlat; ++i) {
    const Real lat = -0.5 * constants::PI + i * dlambda;
    xyz[2]         = sin(lat);
    for (int j = 0; j < nlon; ++j) {
      const Real lon = j * dlambda;
      xyz[0]         = cos(lat) * cos(lon);
      xyz[1]         = cos(lat) * sin(lon);

      proj_mat = spherical_tangent_projection_matrix(xyz);
      Real proj_ones[3];
      apply_3by3(proj_ones, proj_mat, ones_sph);

      if (!FloatingPoint<Real>::zero(SphereGeometry::dot(proj_ones, xyz),
                                     fp_tol)) {
        logger.error(
            "tangent projection error at (lon, lat) = ({}, {}), xyz = ({}, {}, "
            "{})",
            lon, lat, xyz[0], xyz[1], xyz[2]);
      }
      CHECK(FloatingPoint<Real>::zero(SphereGeometry::dot(proj_ones, xyz),
                                      fp_tol));

      np_mat = north_pole_rotation_matrix(xyz);
      Real np[3];
      apply_3by3(np, np_mat, xyz);

      if (!FloatingPoint<Real>::zero(np[0], fp_tol) or
          (!FloatingPoint<Real>::zero(np[0], fp_tol) or
           !FloatingPoint<Real>::equiv(np[2], 1, fp_tol))) {
        logger.error(
            "north pole rotation failed for i = {}, j = {}: np = ({}, {}, {})",
            i, j, np[0], np[1], np[2]);
      }

      CHECK(FloatingPoint<Real>::zero(np[0], fp_tol));
      CHECK(FloatingPoint<Real>::zero(np[1], fp_tol));
      CHECK(FloatingPoint<Real>::equiv(np[2], 1, fp_tol));

      Real xyz_check[3];
      apply_3by3_transpose(xyz_check, np_mat, np);

      if (!FloatingPoint<Real>::equiv(xyz[0], xyz_check[0], fp_tol) or
          (!FloatingPoint<Real>::equiv(xyz[1], xyz_check[1], fp_tol) or
           !FloatingPoint<Real>::equiv(xyz[2], xyz_check[2], fp_tol))) {
        logger.error(
            "rotation inversion failed for i = {}, j = {}: xyz = ({}, {}, {}), "
            "xyz_check = ({}, {}, {})",
            i, j, xyz[0], xyz[1], xyz[2], xyz_check[0], xyz_check[1],
            xyz_check[2]);
      }

      CHECK(FloatingPoint<Real>::equiv(xyz[0], xyz_check[0], fp_tol));
      CHECK(FloatingPoint<Real>::equiv(xyz[1], xyz_check[1], fp_tol));
      CHECK(FloatingPoint<Real>::equiv(xyz[2], xyz_check[2], fp_tol));
    }
  }  // lon-lat loop

  Kokkos::Tuple<Real, 9> matrix_a;
  Kokkos::Tuple<Real, 9> matrix_b;
  Kokkos::Tuple<Real, 9> matrix_c;
  const Kokkos::Tuple<Real, 9> matrix_c_exact = {150, 153, 156, 420, 432,
                                                 444, 690, 711, 732};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      matrix_a[3 * i + j] = 3 * i + j;
      matrix_b[3 * i + j] = 30 * i + j;
    }
  }
  matmul_3by3(matrix_c, matrix_a, matrix_b);
  std::ostringstream ss;
  ss << " matrix_a = " << matrix_a << "\n"
     << " matrix_b = " << matrix_b << "\n"
     << " matrix_c_exact = " << matrix_c_exact << "\n"
     << " matrix_c       = " << matrix_c;
  logger.info(ss.str());
  for (int i = 0; i < 9; ++i) {
    CHECK(FloatingPoint<Real>::equiv(matrix_c[i], matrix_c_exact[i]));
  }

  Real mat1[4]                      = {3, 1, 0, -2};
  Real eye2[4]                      = {1, 0, 0, 1};
  Kokkos::Tuple<Real, 2> mat1_evals = two_by_two_real_eigenvalues(mat1);
  Kokkos::Tuple<Real, 2> eye2_evals = two_by_two_real_eigenvalues(eye2);
  CHECK(FloatingPoint<Real>::equiv(mat1_evals[0], 3));
  CHECK(FloatingPoint<Real>::equiv(mat1_evals[1], -2));
  logger.info("mat1 eigenvalue 1: {} (expected 3)", mat1_evals[0]);
  logger.info("mat2 eigenvalue 2: {} (expected -2)", mat1_evals[1]);
  CHECK(FloatingPoint<Real>::equiv(eye2_evals[0], 1));
  CHECK(FloatingPoint<Real>::equiv(eye2_evals[1], 1));

#ifdef LPM_USE_BOOST
  logger.info("Checking that BesselJ0 is even");
  CHECK(
      FloatingPoint<Real>::equiv(cyl_bessel_j(0, 0.5), cyl_bessel_j(0, -0.5)));
  logger.info("Checking that BesselJ1 is odd");
  CHECK(
      FloatingPoint<Real>::equiv(cyl_bessel_j(1, 0.5), -cyl_bessel_j(1, -0.5)));

  logger.info("Checking that LegendreP(l,1) = 1 for l in {0,1,2,3,4}");
  for (int l = 0; l < 5; ++l) {
    CHECK(FloatingPoint<Real>::equiv(legendre_p(l, 1), 1));
  }
#endif
}
