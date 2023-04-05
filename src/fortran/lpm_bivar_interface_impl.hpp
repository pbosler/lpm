#ifndef LPM_BIVAR_INTERFACE_IMPL_HPP
#define LPM_BIVAR_INTERFACE_IMPL_HPP

#include "lpm_bivar_interface.hpp"
#include "util/lpm_string_util.hpp"

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

template <typename SeedType>
BivarInterface<SeedType>::BivarInterface(
    const GatherMeshData<SeedType>& in,
    const typename scalar_view_type::HostMirror xo,
    const typename scalar_view_type::HostMirror yo,
    const std::map<std::string, std::string>& s_in_out,
    const std::map<std::string, std::string>& v_in_out)
    : input(in),
      x_out(xo),
      y_out(yo),
      md(1),
      integer_work("bivar_integer_work", 31 * in.x.extent(0) + x_out.extent(0)),
      real_work("bivar_real_work", 8 * in.x.extent(0)),
      scalar_in_out_map(s_in_out),
      vector_in_out_map(v_in_out) {
  LPM_REQUIRE_MSG(
      in.unpacked,
      "BivarInterface error: GatheredData must have unpacked coordinates.");
  for (const auto& io : s_in_out) {
    const bool input_found =
        (in.scalar_fields.find(io.first) != in.scalar_fields.end());
    LPM_REQUIRE_MSG(input_found,
                    "BivarInterface: input field " + io.first + " not found.");
  }
  for (const auto& io : v_in_out) {
    const bool input_found =
        (in.vector_fields.find(io.first) != in.vector_fields.end());
    LPM_REQUIRE_MSG(input_found,
                    "BivarInterface: input field " + io.first + " not found.");
  }
  /// mesh data may have been gathered on device, so we update host here
  /// because bivar only works on host.
  in.update_host();
}

template <typename SeedType>
void BivarInterface<SeedType>::interpolate(const sfield_map& output_scalars,
                                           const vfield_map& output_vectors) {
  const Index nsrc = input.x.extent(0);
  const Index ndst = x_out.extent(0);
  const auto xin = input.h_x;
  const auto yin = input.h_y;

  for (const auto& io : scalar_in_out_map) {
    const auto zin = input.scalar_fields.at(io.first);
    LPM_REQUIRE_MSG(output_scalars.count(io.second) == 1,
                    "BivarInterface::interpolate error: output scalar field " +
                        io.second + " not found.");
    const auto zout = output_scalars.at(io.second);

    c_idbvip(md, nsrc, xin.data(), yin.data(), zin.data(), ndst, x_out.data(),
             y_out.data(), zout.data(), integer_work.data(), real_work.data());
    md = 3;  /// Tell bivar.f90 that the input data will not change
  }
  /// todo: testing required --- is it faster to interpolate one component at a
  /// time, or one pair of components individually, in a loop?
  //// to start, we do one component at a time, which requires an extra deep
  ///copy
  /// because of the component subview may have strided memory access, which
  /// does not translate through the c/fortran bridge.
  if (!vector_in_out_map.empty()) {
    Kokkos::View<Real*, Host> vec_comp_in("bivar_vec_component_in", input.n());
    Kokkos::View<Real*, Host> vec_comp_out("bivar_vec_component_out",
                                           x_out.extent(0));
    for (const auto& io : vector_in_out_map) {
      LPM_REQUIRE_MSG(
          output_vectors.count(io.second) == 1,
          "BivarInterface::interpolate error: output vector field " +
              io.second + " not found.");

      for (int j = 0; j < SeedType::geo::ndim; ++j) {
        /// get the input component subview (generally, it will have strided
        /// access)
        const auto vci =
            Kokkos::subview(input.h_vector_fields.at(io.first), Kokkos::ALL, j);
        /// deep copy the component data into a contiguous array
        for (Index i = 0; i < input.n(); ++i) {
          vec_comp_in(i) = vci(i);
        }
        /// interpolate the single component at all tgt locations
        c_idbvip(md, nsrc, xin.data(), yin.data(), vec_comp_in.data(), ndst,
                 x_out.data(), y_out.data(), vec_comp_out.data(),
                 integer_work.data(), real_work.data());
        /// Tell bivar.f90 that the input data will not change
        md = 3;
        /// get the output component subview (generally, it will have strided
        /// access)
        const auto vco =
            Kokkos::subview(output_vectors.at(io.second), Kokkos::ALL, j);
        /// deep copy the component to the possibly strided output subview
        for (Index i = 0; i < x_out.extent(0); ++i) {
          vco(i) = vec_comp_out(i);
        }
      }
    }
  }
}

template <typename SeedType>
void BivarInterface<SeedType>::interpolate_lag_crds(typename SeedType::geo::crd_view_type::HostMirror lcrds) {
  Kokkos::View<Real*, Host> lag_x_out("bivar_lag_x_out", x_out.extent(0));
  Kokkos::View<Real*, Host> lag_y_out("bivar_lag_x_out", x_out.extent(0));

  c_idbvip(md, input.n(), input.h_x.data(), input.h_y.data(), input.h_lag_x.data(),
    x_out.data(), y_out.data(), lag_x_out.data(), integer_work.data(),
    real_work.data());
  c_idbvip(md, input.n(), input.h_x.data(), input.h_y.data(), input.h_lag_y.data(),
    x_out.data(), y_out.data(), lag_y_out.data(), integer_work.data(),
    real_work.data());

  for (int i=0; i<x_out.extent(0); ++i) {
    lcrds(i, 0) = lag_x_out(i);
    lcrds(i, 1) = lag_y_out(i);
  }
}

template <typename SeedType>
std::string BivarInterface<SeedType>::info_string(const int tab_lev) const {
  auto tabstr = indent_string(tab_lev);
  std::ostringstream ss;
  ss << tabstr << "BivarInterface<" << SeedType::id_string() << "> info:\n";
  tabstr += "\t";
  ss << tabstr << "nsrc pts = " << input.n() << "\n";
  ss << tabstr << "ntgt pts = " << x_out.extent(0) << "\n";
  ss << tabstr << "scalar_in_out_map.size() = " << scalar_in_out_map.size() << "\n";
  ss << tabstr << "vector_in_out_map.size() = " << vector_in_out_map.size() << "\n";
  ss << tabstr << "md = " << md << "(" << md_str(md) << ")\n";
  return ss.str();
}

template <typename SeedType>
std::string BivarInterface<SeedType>::md_str(const int val) const {
  std::string result;
  switch (val) {
    case 1 : {
      result = "first interpolation, or new source data locations";
      break;
    }
    case 2 : {
      result = "second or later interpolation of same source data locations, but new target locations";
      break;
    }
    case 3 : {
      result = "second or later interpolation of same source data locations and same target locations, but new function data";
      break;
    }
    default : {
      break;
    }
  }
  return result;
}

}  // namespace Lpm

#endif
