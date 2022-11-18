#include "lpm_f_interp.hpp"
#include "mesh/lpm_mesh_seed.hpp"

namespace Lpm {

void c_idbvip(int md, int nsrc, double* xin, double* yin, double* zin,
  int ndst, double* xout, double* yout, double* zout, int* iwk, double* rwk) {

  idbvip(&md, &nsrc, xin, yin, zin, &ndst, xout, yout, zout, iwk, rwk);
}

void BivarInterface::interpolate_scalar() {
  c_idbvip(md, nsrc, x_in.data(), y_in.data(), z_in.data(),
    ndst, x_out.data(), y_out.data(), z_out.data(),
    integer_work.data(), real_work.data());
  md = 3;
}

}
