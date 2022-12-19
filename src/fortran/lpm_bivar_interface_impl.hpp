#ifndef LPM_BIVAR_INTERFACE_IMPL_HPP
#define LPM_BIVAR_INTERFACE_IMPL_HPP

#include "lpm_bivar_interface.hpp"

namespace Lpm {

void c_idbvip(int md, int nsrc, double* xin, double* yin, double* zin,
  int ndst, double* xout, double* yout, double* zout, int* iwk, double* rwk) {

  idbvip(&md, &nsrc, xin, yin, zin, &ndst, xout, yout, zout, iwk, rwk);
}

template <typename SeedType>
BivarInterface<SeedType>::BivarInterface(
  const GatherMeshData<SeedType>& in,
  const GatherMeshData<SeedType>& out,
  const std::map<std::string, std::string>& s_in_out,
  const std::map<std::string, std::string>& v_in_out) :
    input(in),
    output(out),
    md(1),
    integer_work("bivar_integer_work", 31*in.x.extent(0) + out.x.extent(0)),
    real_work("bivar_real_work", 8*in.x.extent(0)),
    scalar_in_out_map(s_in_out),
    vector_in_out_map(v_in_out)
{
  for (const auto& io : s_in_out) {
    const bool input_found = (in.scalar_fields.find(io.first) !=
                              in.scalar_fields.end());

    const bool output_found = (out.scalar_fields.find(io.second) !=
                               out.scalar_fields.end());
    LPM_REQUIRE_MSG( input_found,
      "BivarInterface: input field " + io.first + " not found.");
    LPM_REQUIRE_MSG( output_found,
      "BivarInterface: output field " + io.second + " not found.");
  }
  for (const auto& io : v_in_out) {
    const bool input_found = (in.vector_fields.find(io.first) !=
                              in.vector_fields.end());
    const bool output_found = (out.vector_fields.find(io.second) !=
                               out.vector_fields.end());
    LPM_REQUIRE_MSG( input_found,
      "BivarInterface: input field " + io.first + " not found.");
    LPM_REQUIRE_MSG( output_found,
      "BivarInterface: output field " + io.second + " not found.");
  }
  in.update_host();
  out.update_host();
}

template <typename SeedType>
void BivarInterface<SeedType>::interpolate() {
  const Index nsrc = input.x.extent(0);
  const Index ndst = output.x.extent(0);
  const auto xin = input.h_x;
  const auto yin = input.h_y;
  const auto xout = output.h_x;
  const auto yout = output.h_y;

  for (const auto& io : scalar_in_out_map) {
    const auto zin = input.scalar_fields.at(io.first);
    const auto zout = output.h_scalar_fields.at(io.second);

    c_idbvip(md, nsrc, xin.data(), yin.data(), zin.data(),
      ndst, xout.data(), yout.data(), zout.data(),
      integer_work.data(), real_work.data());
    md = 3;
  }
  if ( !vector_in_out_map.empty() ) {
    Kokkos::View<Real*, Host> vec_comp_in("bivar_vec_component_in", input.n());
    Kokkos::View<Real*, Host> vec_comp_out("bivar_vec_component_out", output.n());
    for (const auto& io : vector_in_out_map) {
      for (int j=0; j<SeedType::geo::ndim; ++j) {
        const auto vci = Kokkos::subview(input.h_vector_fields.at(io.first),
          Kokkos::ALL, j);
        for (Index i=0; i<input.n(); ++i) {
          vec_comp_in(i) = vci(i);
        }
        c_idbvip(md, nsrc, xin.data(), yin.data(), vec_comp_in.data(),
          ndst, xout.data(), yout.data(), vec_comp_out.data(),
          integer_work.data(), real_work.data());
        md = 3;
        const auto vco = Kokkos::subview(output.h_vector_fields.at(io.second),
          Kokkos::ALL, j);
        for (Index i=0; i<output.n(); ++i) {
          vco(i) = vec_comp_out(i);
        }
      }
    }
  }
  output.update_device();
}

} // namespace Lpm

#endif
