#include "LpmConfig.h"
#include "lpm_comm.hpp"
#include "lpm_constants.hpp"
#include "lpm_logger.hpp"
#include "lpm_swe_kernels.hpp"
#include "catch.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>

using namespace Lpm;

TEST_CASE("kernel values", "") {

  Real x[3];
  Real y[3];

  const Real xlon = constants::PI/4;
  const Real xlat = constants::PI/8;

  const Real ylon = constants::PI/6;
  const Real ylat = constants::PI/20;

  x[0] = cos(xlat) * cos(xlon);
  x[1] = cos(xlat) * sin(xlon);
  x[2] = sin(xlat);

  y[0] = cos(ylat) * cos(ylon);
  y[1] = cos(ylat) * sin(ylon);
  y[2] = sin(ylat);

  const Real eps0 = 0;
  const Real eps1 = 0.01;

  Comm comm;
  Logger<> logger("swe kernel values test", Log::level::debug, comm);

  logger.info("pts on sphere: \n\t x = [{}, {}, {}] \n\t y = [{}, {}, {}]",
    x[0], x[1], x[2], y[0], y[1], y[2]);

  Real kzeta0[3];
  Real kzeta1[3];
  Real ksigma0[3];
  Real ksigma1[3];
  const Real kzeta_exact[3] = { 0.11761244724273439212, -0.30509169352258549181, 0.32004709265607662522};
  const Real ksigma_exact[3] = {0.32583437560369497620, -0.16407250414068629817, -0.27614478771192878158};
  for (int i=0; i<3; ++i) {
    kzeta0[i] = 0;
    kzeta1[i] = 0;
    ksigma0[i] = 0;
    ksigma1[i] = 0;
  }
  const Real vorticity_unit = 1;
  const Real divergence_unit = 1;
  const Real area_unit = 1;
  kzeta_sphere(kzeta0, x, y, vorticity_unit, area_unit, eps0);
  kzeta_sphere(kzeta1, x, y, vorticity_unit, area_unit, eps1);
  ksigma_sphere(ksigma0, x, y, divergence_unit, area_unit, eps0);
  ksigma_sphere(ksigma1, x, y, divergence_unit, area_unit, eps1);

  logger.info("biot-savart kernel with eps = 0    : [{}, {}, {}]",
    kzeta0[0], kzeta0[1], kzeta0[2]);

  logger.info("biot-savart kernel with eps = 0.01 : [{}, {}, {}]",
    kzeta1[0], kzeta1[1], kzeta1[2]);

  logger.info("potential kernel with eps = 0    : [{}, {}, {}]",
    ksigma0[0], ksigma0[1], ksigma0[2]);

  logger.info("potential kernel with eps = 0.01 : [{}, {}, {}]",
    ksigma1[0], ksigma1[1], ksigma1[2]);

  const Real kz0_dot_x = SphereGeometry::dot(kzeta0, x);
  const Real kz1_dot_x = SphereGeometry::dot(kzeta1, x);
  const Real ks0_dot_x = SphereGeometry::dot(ksigma0, x);
  const Real ks1_dot_x = SphereGeometry::dot(ksigma1, x);
  logger.info("kzeta0 . x = {}", kz0_dot_x);
  logger.info("kzeta1 . x = {}", kz1_dot_x);
  logger.info("ksigma0 . x = {}", ks0_dot_x);
  logger.info("ksigma1 . x = {}", ks1_dot_x);

  for (int i=0; i<3; ++i) {
    CHECK( kzeta0[i] == Approx(kzeta_exact[i]) );
    CHECK( kzeta1[i] == Approx(kzeta_exact[i]).epsilon(20*square(eps1)) );
    CHECK( ksigma0[i] == Approx(ksigma_exact[i]) );
    CHECK( ksigma1[i] == Approx(ksigma_exact[i]).epsilon(20*square(eps1)) );
    CHECK( FloatingPoint<Real>::zero(kz0_dot_x) );
    CHECK( FloatingPoint<Real>::zero(kz1_dot_x) );
    CHECK( FloatingPoint<Real>::zero(ks0_dot_x) );
    CHECK( FloatingPoint<Real>::zero(ks1_dot_x) );
  }
}
