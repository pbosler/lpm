#ifndef LPM_VTK_IO_HPP
#define LPM_VTK_IO_HPP

#include "LpmConfig.h"

#ifdef LPM_USE_VTK

#include <memory>
#include <vector>

#include "Kokkos_Core.hpp"
#include "lpm_coords.hpp"
#include "lpm_geometry.hpp"
#include "mesh/lpm_edges.hpp"
#include "mesh/lpm_faces.hpp"
#include "mesh/lpm_mesh_seed.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#include "mesh/lpm_vertices.hpp"
#include "vtkCellData.h"
#include "vtkDoubleArray.h"
#include "vtkPointData.h"
#include "vtkPolyData.h"
#include "vtkPolyDataWriter.h"
#include "vtkSmartPointer.h"
#include "vtkXMLPolyDataWriter.h"

namespace Lpm {

template <typename SeedType>
class VtkPolymeshInterface {
 public:
  VtkPolymeshInterface(const PolyMesh2d<SeedType>& pm);

  VtkPolymeshInterface(
      const PolyMesh2d<SeedType>& pm,
      const typename scalar_view_type::HostMirror height_field);

  void write(const std::string& ofilename);

  void update_positions();

  template <typename VT = typename scalar_view_type::HostMirror>
  void add_scalar_point_data(const VT s, const std::string& name = "");

  template <typename VT = typename SeedType::geo::vec_view_type::HostMirror>
  void add_vector_point_data(const VT v, const std::string& name = "");

  template <typename VT = typename scalar_view_type::HostMirror>
  void add_scalar_cell_data(const VT s, const std::string& name = "");

  template <typename VT = typename SeedType::geo::vec_view_type::HostMirror>
  void add_vector_cell_data(const VT v, const std::string& name = "");

  void add_tracers(const std::vector<scalar_view_type>& point_tracers,
                   const std::vector<scalar_view_type>& cell_tracers);

 protected:
  const PolyMesh2d<SeedType>& mesh_;

  vtkSmartPointer<vtkPolyData> polydata_;
  vtkSmartPointer<vtkPointData> pointdata_;
  vtkSmartPointer<vtkCellData> celldata_;

  vtkSmartPointer<vtkXMLPolyDataWriter> writer_;

  vtkSmartPointer<vtkPoints> make_points() const;
  vtkSmartPointer<vtkPoints> make_points(
      const typename scalar_view_type::HostMirror height) const;
  vtkSmartPointer<vtkCellArray> make_cells() const;
  vtkSmartPointer<vtkDoubleArray> make_cell_area() const;
};

template <typename Geo, typename FaceKind>
class VtkInterface {
 public:
  vtkSmartPointer<vtkPolyData> toVtkPolyData(
      const Faces<FaceKind, Geo>& faces, const Edges& edges,
      const Vertices<Coords<Geo>>& verts,
      const vtkSmartPointer<vtkPointData>& ptdata = 0,
      const vtkSmartPointer<vtkCellData>& cdata = 0) const;

  void writePolyData(const std::string& fname,
                     const vtkSmartPointer<vtkPolyData> pd);

  void addScalarToPointData(vtkSmartPointer<vtkPointData>& pd,
                            const typename scalar_view_type::HostMirror sf,
                            const std::string& name, const Index nverts) const;

  void addVectorToPointData(
      vtkSmartPointer<vtkPointData>& pd,
      const typename ko::View<Real * [Geo::ndim], Dev>::HostMirror vf,
      const std::string& name, const Index nverts) const;

  void addScalarToCellData(vtkSmartPointer<vtkCellData>& cd,
                           const typename scalar_view_type::HostMirror sf,
                           const std::string& name,
                           const Faces<FaceKind, Geo>& faces) const;

  void addVectorToCellData(
      vtkSmartPointer<vtkCellData>& cd,
      const typename ko::View<Real * [Geo::ndim], Dev>::HostMirror vf,
      const std::string& name, const Faces<FaceKind, Geo>& faces) const;

 protected:
  vtkSmartPointer<vtkPolyDataWriter> pdwriter;
};

inline std::string vtp_suffix() { return ".vtp"; }

}  // namespace Lpm

#endif  // LPM_USE_VTK

#endif
