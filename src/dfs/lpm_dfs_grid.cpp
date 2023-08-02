#include "lpm_dfs_grid.hpp"
#ifdef LPM_USE_VTK
#include "vtkPoints.h"
#endif

namespace Lpm {
namespace DFS {

void DFSGrid::fill_packed_view(view_type& view) const {

//     auto h_view = Kokkos::create_mirror_view(view);
//     for (Int i=0; i<nlat; ++i) {
//       for (Int j=0; j<nlon; ++j) {
//         const Int idx = i * nlon + j;
//         auto mxyz = Kokkos::subview(h_view, idx, Kokkos::ALL);
//         sph2xyz(mxyz, i, j);
//       }
//     }
//     Kokkos::deep_copy(view, h_view);
    const auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{nlat,nlon});
    Kokkos::parallel_for("DfsGridCoords", policy,
      KOKKOS_LAMBDA (const Index i, const Index j) {
      const auto idx = i*nlon +j;
      const auto mxyz = Kokkos::subview(view, idx, Kokkos::ALL);
      sph2xyz(mxyz, i, j);
    });
}

#ifdef LPM_USE_VTK
vtkSmartPointer<vtkStructuredGrid> DFSGrid::vtk_grid() const {
  auto pts = vtkSmartPointer<vtkPoints>::New();
  for (int i=0; i<nlat; ++i) {
    const Real colat = colatitude(i);
    const Real z = cos(colat);
    // vtk needs the 2 \pi point, even though we don't use it
    for (int j=0; j<=nlon; ++j) {
      const Real lon = longitude(j);
      const Real x = sin(colat)*cos(lon);
      const Real y = sin(colat)*sin(lon);
      pts->InsertNextPoint(x,y,z);
    }
  }
  auto grid = vtkSmartPointer<vtkStructuredGrid>::New();
  grid->SetDimensions(nlon+1, nlat, 1);
  grid->SetPoints(pts);
  return grid;
}
#endif

} // namespace DFS
} // namespace Lpm
