#ifndef LPM_SSRFPACK_INTERFACE_HPP
#define LPM_SSRFPACK_INTERFACE_HPP

#include "LpmConfig.h"
#include "mesh/lpm_gather_mesh_data.hpp"

namespace Lpm {

/** Interface to the STRIPACK Fortan module for constructing
  Delaunay triangulations and Voronoi tesselations of points on a sphere.

  Required as input to the SSRFPACK interpolation library.

  All of these class methods run on host.   Call update_host before using
  this class.
*/
class STRIPACKInterface final {
 public:
  Kokkos::View<int*, Host> list;
  Kokkos::View<int*, Host> lptr;
  Kokkos::View<int*, Host> lend;
  int n;

  template <typename SeedType>
  STRIPACKInterface(const GatherMeshData<SeedType>& in);

 private:
  template <typename SeedType>
  void build_triangulation(const GatherMeshData<SeedType>& in);
};

/** Interface to the SSRFPACK Fortan module for interpolation of scattered
  data on the sphere.

  All of these class methods run on host.  Call update_host before using.

*/
template <typename SeedType>
struct SSRFPackInterface final {
  const GatherMeshData<SeedType>& input;
  using sfield_map = std::map<std::string, typename scalar_view_type::HostMirror>;
  using vfield_map =
      std::map<std::string, typename SphereGeometry::vec_view_type::HostMirror>;
  std::map<std::string, std::string> scalar_in_out_map;
  std::map<std::string, std::string> vector_in_out_map;

  static_assert(std::is_same<typename SeedType::geo, SphereGeometry>::value,
                "SSRFPackInterface:: sphere geometry required.");

  SSRFPackInterface(const GatherMeshData<SeedType>& in,
                    const std::map<std::string, std::string>& s_in_out =
                      std::map<std::string, std::string>(),
                    const std::map<std::string, std::string>& v_in_out =
                        std::map<std::string, std::string>());

  template <typename PolyMeshType>
  void interpolate(
      PolyMeshType& pm,
      std::map<std::string, ScalarField<VertexField>> vert_scalar_fields,
      std::map<std::string, ScalarField<FaceField>> face_scalar_fields,
      std::map<std::string, VectorField<SphereGeometry, VertexField>>
          vert_vector_fields =
              std::map<std::string, VectorField<SphereGeometry, VertexField>>(),
      std::map<std::string, VectorField<SphereGeometry, FaceField>>
          face_vector_fields =
              std::map<std::string, VectorField<SphereGeometry, FaceField>>());

  void interpolate(
      const typename Kokkos::View<Real* [3]>::HostMirror output_pts,
      const sfield_map& scalar_fields,
      const vfield_map& vector_fields = vfield_map());

  template <typename PolyMeshType>
  void interpolate_lag_crds(PolyMeshType& mesh_out);

  int sigma_flag;

  double sigma_tol;

  std::string info_string(const int tab_lev=0) const;

 private:
  void set_scalar_source_data(const std::string& field_name);

  void set_vector_source_data(const std::string& field_name);

  void set_lag_crd_source_data();

  std::string sigma_str() const;

  std::string grad_str() const;

  Kokkos::View<double*, HostMemory> comp1;
  Kokkos::View<double*, HostMemory> comp2;
  Kokkos::View<double*, HostMemory> comp3;
  Kokkos::View<double**, Kokkos::LayoutLeft, HostMemory> grad1;
  Kokkos::View<double**, Kokkos::LayoutLeft, HostMemory> grad2;
  Kokkos::View<double**, Kokkos::LayoutLeft, HostMemory> grad3;
  Kokkos::View<double*, HostMemory> sigma1;
  Kokkos::View<double*, HostMemory> sigma2;
  Kokkos::View<double*, HostMemory> sigma3;
  STRIPACKInterface del_tri;
  int grad_flag;
};

}  // namespace Lpm
#endif
