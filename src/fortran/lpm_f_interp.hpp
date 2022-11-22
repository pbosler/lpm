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

template <typename SeedType>
struct BivarInterface {
  const GatherMeshData<SeedType>& input;
  const GatherMeshData<SeedType>& output;
  Kokkos::View<Int*, Host> integer_work;
  Kokkos::View<Real*, Host> real_work;
  std::map<std::string, std::string> scalar_in_out_map;
  std::map<std::string, std::string> vector_in_out_map;
  Int md;

  static_assert(std::is_same<typename SeedType::geo, PlaneGeometry>::value,
    "planar geometry required.");

  BivarInterface(const GatherMeshData<SeedType>& in,
    const GatherMeshData<SeedType>& out,
    const std::map<std::string, std::string>& s_in_out,
    const std::map<std::string, std::string>& v_in_out =
      std::map<std::string, std::string>());

  void interpolate();
};

}
#endif
