#ifndef LPM_F_INTERP_HPP
#define LPM_F_INTERP_HPP

#include "LpmConfig.h"
#include "mesh/lpm_gather_mesh_data.hpp"
// #include "lpm_fortran_c.h"

#ifdef __cplusplus
extern "C" {
#endif

extern void idbvip(int* md, int*, double*, double*, double*,
    int*, double*, double*, double*, int*, double*);

#ifdef __cplusplus
}
#endif

void c_idbvip(int md, int nsrc, double* xin, double* yin, double* zin,
  int ndst, double* xout, double* yout, double* zout, int* iwk, double* rwk);

namespace Lpm {

struct BivarInterface {
  int nsrc;
  int ndst;
  int md;
  Kokkos::View<Real*, Host> x_in;
  Kokkos::View<Real*, Host> y_in;
  Kokkos::View<Real*, Host> z_in;
  Kokkos::View<Real*, Host> x_out;
  Kokkos::View<Real*, Host> y_out;
  Kokkos::View<Real*, Host> z_out;
  Kokkos::View<Int*, Host> integer_work;
  Kokkos::View<Real*, Host> real_work;

  template <typename RealViewIn, typename RealViewOut>
  BivarInterface(const RealViewIn& ix,
    const RealViewIn& iy,
    const RealViewIn& iz,
    const RealViewOut& ox,
    const RealViewOut& oy,
    const RealViewOut& oz):
    nsrc(ix.extent(0)),
    ndst(ox.extent(0)),
    md(1),
    x_in(ix),
    y_in(iy),
    z_in(iz),
    x_out(ox),
    y_out(oy),
    z_out(oz),
    integer_work("bivar_integer_work", 31*ix.extent(0) + ox.extent(0)),
    real_work("bivar_real_work", 8*ix.extent(0)) {}

  template <typename SeedType>
  BivarInterface(const Index nd,
    const GatherSourceData<SeedType>& sdata) :
  nsrc(sdata.n()),
  ndst(nd),
  md(1),
  x_in("bivar_x_in", sdata.n()),
  y_in("bivar_y_in", sdata.n()),
  z_in("bivar_z_in", sdata.n()),
  x_out("bivar_x_out", nd),
  y_out("bivar_y_out", nd),
  z_out("bivar_z_out", nd),
  integer_work("bivar_integer_work", 31*sdata.n() + nd),
  real_work("bivar_real_work", 8*nd) {}

  template <typename RealViewIn, typename RealViewOut>
  inline void reset_output_pts(const RealViewIn& xout, const RealViewIn& yout,
    const RealViewOut& zout) {
      x_out = xout;
      y_out = yout;
      z_out = zout;
      md = 2;
    }

  template <typename RealViewIn, typename RealViewOut>
  inline void reset_output_vals(const RealViewIn& iz, const RealViewOut& oz) {
    z_in = iz;
    z_out = oz;
    md = 3;
  }

  void interpolate_scalar();
};

}
#endif
