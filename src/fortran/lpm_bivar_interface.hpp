#ifndef LPM_BIVAR_INTERFACE_HPP
#define LPM_BIVAR_INTERFACE_HPP

#include "LpmConfig.h"
#include "mesh/lpm_gather_mesh_data.hpp"

namespace Lpm {

/**
  Given source data from a planar Polymesh2d, interpolate using the bivar.f90
  module to output locations.
*/
template <typename SeedType>
struct BivarInterface final {
  using sfield_map = std::map<std::string, scalar_view_type>;
  using vfield_map =
      std::map<std::string, typename SeedType::geo::vec_view_type>;
  /**
    every source point is included in the bivar algorithm, so we need
    GatherMeshData to avoid duplicate points from divided panels for input.
  */
  const GatherMeshData<SeedType>& input;
  /// x-coordinates of output pts
  typename scalar_view_type::HostMirror x_out;
  /// y-coordinates of output pts
  typename scalar_view_type::HostMirror y_out;
  /// input scalar field names map to output field names; allows interpolated
  /// output to have different name than the input field
  std::map<std::string, std::string> scalar_in_out_map;
  /// input vector field names map to output field names; allows interpolated
  /// output to have different name than the input field
  std::map<std::string, std::string> vector_in_out_map;

  static_assert(std::is_same<typename SeedType::geo, PlaneGeometry>::value,
                "planar geometry required.");

  /** @brief constructor.

    @param [in] in GatheredMeshData
    @param [in] xo x-coordinates of output pts
    @param [in] yo y-coordinates of output pts
    @param [in] s_in_out mapping pairs input -> output names for scalar data
    @param [in] v_in_out mapping pairs input -> output names for vector data
  */
  BivarInterface(const GatherMeshData<SeedType>& in,
                 const typename scalar_view_type::HostMirror xo,
                 const typename scalar_view_type::HostMirror yo,
                 const std::map<std::string, std::string>& s_in_out,
                 const std::map<std::string, std::string>& v_in_out =
                     std::map<std::string, std::string>());

  void interpolate(const sfield_map& output_scalars,
                   const vfield_map& output_vectors = vfield_map());

 private:
  Kokkos::View<int*, Host> integer_work;
  Kokkos::View<double*, Host> real_work;
  Int md;
};

}  // namespace Lpm

#endif
