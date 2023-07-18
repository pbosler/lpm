#ifndef LPM_DFS_GRID_HPP
#define LPM_DFS_GRID_HPP

#include "LpmConfig.h"
#include "lpm_constants.hpp"
#include "lpm_coords.hpp"
#include "lpm_coords_impl.hpp"
#include "lpm_geometry.hpp"
#include "lpm_field.hpp"
#ifdef LPM_USE_VTK
#include "vtkStructuredGrid.h"
#include "vtkSmartPointer.h"
#endif

namespace Lpm {
namespace DFS {

using scalar_field_type = ScalarField<VertexField>;
using vector_field_type = VectorField<SphereGeometry, VertexField>;

struct DFSGrid {
  using view_type = typename SphereGeometry::crd_view_type;

  Int nlon;
  Int nlat;

  KOKKOS_INLINE_FUNCTION
  explicit DFSGrid(const Int nl) : nlon(nl), nlat(nl/2 + 1) {}

  KOKKOS_INLINE_FUNCTION
  Real colatitude(const Int i) const {
    // here we subtract 1 from nlat to make sure that both poles are included.
    return constants::PI * i / (nlat-1);
  }

  KOKKOS_INLINE_FUNCTION
  Real longitude(const Int j) const {
    // we don't subtract 1 from nlon because we don't want both 0 and 2*/\pi
    return 2*constants::PI * j / nlon;
  }

  template <typename PtType> KOKKOS_INLINE_FUNCTION
  void sph2xyz(PtType& xyz, const Int i, const Int j) const {
    xyz[0] = sin(colatitude(i)) * cos(longitude(j));
    xyz[1] = sin(colatitude(i)) * sin(longitude(j));
    xyz[2] = cos(colatitude(i));
  }

  view_type packed_view() const;

#ifdef LPM_USE_VTK
  vtkSmartPointer<vtkStructuredGrid> vtk_grid() const;
#endif
};

} // namespace DFS
} // namespace Lpm

#endif
