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
  using sfield_map = std::map<std::string, typename scalar_view_type::HostMirror>;
  using vfield_map =
      std::map<std::string, typename SeedType::geo::vec_view_type::HostMirror>;

  static_assert(std::is_same<typename SeedType::geo, PlaneGeometry>::value,
                "planar geometry required.");
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
                 const std::map<std::string, std::string>& s_in_out =
                     std::map<std::string, std::string>(),
                 const std::map<std::string, std::string>& v_in_out =
                     std::map<std::string, std::string>());

  void interpolate(const sfield_map& output_scalars,
                   const vfield_map& output_vectors = vfield_map());

  void interpolate(const sfield_map& output_scalars,
                   const vfield_map& output_vectors,
                   const Index start_idx,
                   const Index end_idx);

  void interpolate_vectors(const vfield_map& output_vectors);

  void interpolate_vectors(const vfield_map& output_vectors, const Index start, const Index end);

  void interpolate_lag_crds(typename SeedType::geo::crd_view_type::HostMirror lcrds);

  void interpolate_lag_crds(typename SeedType::geo::crd_view_type::HostMirror lcrds,
    const Index start_idx, const Index end_idx);

  std::string info_string(const int tab_lev=0) const;

  void set_md_new_source() {md = 1;}

  void set_md_same_source_new_target() {md = 2;}

  void set_md_same_source_same_target_new_data() {md = 3;}

 private:
  Kokkos::View<int*, Host> integer_work;
  Kokkos::View<double*, Host> real_work;
  Int md;

  std::string md_str(const int val) const;
};

}  // namespace Lpm

#endif
