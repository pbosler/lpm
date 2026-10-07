#include "lpm_dfs_grid.hpp"
#include "lpm_constants.hpp"
#include "lpm_coords_impl.hpp"
#include "util/lpm_string_util.hpp"

#include "vtkPoints.h"


namespace Lpm {
namespace DFS {

Coords<SphereGeometry> DFSGrid::init_coords() const {
  typename SphereGeometry::crd_view_type crd_view("dfs_grid_coords", nlon*nlat);
  fill_packed_view(crd_view);
  return Coords<SphereGeometry>(crd_view);
}

void DFSGrid::fill_packed_view(view_type& view) const {
    const auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{nlat,nlon});
    Kokkos::parallel_for("DfsGridCoords", policy,
      KOKKOS_LAMBDA (const Index i, const Index j) {
      const auto idx = i*nlon +j;
      const auto mxyz = Kokkos::subview(view, idx, Kokkos::ALL);
      sph2xyz(mxyz, i, j);
    });
}

scalar_view_type DFSGrid::weights_view() const {
  scalar_view_type result("grid_area", size());
  auto policy = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{nlat,nlon});
  Kokkos::parallel_for("set_grid_area_weights", policy,
    KOKKOS_LAMBDA (const Index i, const Index j) {
      const auto idx = i*nlon + j;
      result(idx) = area_weight(i,j);
    });
  return result;
}

std::string DFSGrid::info_string(const int tab_level) const {
  std::ostringstream ss;
  auto tabstr = indent_string(tab_level);
  ss << tabstr << "DFSGrid info:\n";
  tabstr += "\t";
  ss << tabstr << "nlon = " << nlon << "\n";
  ss << tabstr << "nlat = " << nlat << "\n";
  ss << tabstr << "size() = " << size() << "\n";
  ss << tabstr << "dtheta = " << dtheta << "(" << dtheta * 180 / constants::PI << " degrees)\n";
  ss << tabstr << "dlambda = " << dlambda << "(" << dlambda * 180 / constants::PI << " degrees)\n";
  return ss.str();
}

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


} // namespace DFS
} // namespace Lpm
