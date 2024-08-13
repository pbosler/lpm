#include "lpm_bivar_interface.hpp"
#include "lpm_bivar_interface_impl.hpp"

#ifdef __cplusplus
extern "C" {
#endif

extern void idbvip(int* md, int*, double*, double*, double*, int*, double*,
                   double*, double*, int*, double*);

#ifdef __cplusplus
}
#endif


namespace Lpm {

void c_idbvip(int md, int nsrc, double* xin, double* yin, double* zin, int ndst,
              double* x_out, double* y_out, double* zout, int* iwk,
              double* rwk) {
  idbvip(&md, &nsrc, xin, yin, zin, &ndst, x_out, y_out, zout, iwk, rwk);
}

}
