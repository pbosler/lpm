#ifndef LPM_POLYMESH2D_VTK_INTERFACE_HPP
#define LPM_POLYMESH2D_VTK_INTERFACE_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmUtilities.hpp"
#include "LpmCoords.hpp"
#include "LpmPolyMesh2d.hpp"
#include "vtkSmartPointer.h"
#include "vtkPointData.h"
#include "vtkCellData.h"
#include "vtkPolyData.h"
#include "vtkXMLPolyDataWriter.h"
#include "vtkDoubleArray.h"

#include "Kokkos_Core.hpp"
#include <memory>

namespace Lpm {

template <typename SeedType> class Polymesh2dVtkInterface {
  public:
    Polymesh2dVtkInterface(const std::shared_ptr<PolyMesh2d<SeedType>>& pm);

    void write(const std::string& ofname);

    void updatePositions();
//     void updateAreas();

    template <typename ViewType=typename scalar_view_type::HostMirror>
    void addScalarPointData(const ViewType& s, const std::string& name="");

    template <typename ViewType=typename SeedType::geo::vec_view_type::HostMirror>
    void addVectorPointData(const ViewType& v, const std::string& name="");

    template <typename ViewType=typename scalar_view_type::HostMirror>
    void addScalarCellData(const ViewType& s, const std::string& name="");

    template <typename ViewType=typename SeedType::geo::vec_view_type::HostMirror>
    void addVectorCellData(const ViewType& s, const std::string& name="");

    void addTracers(const std::vector<scalar_view_type>& vt, const std::vector<scalar_view_type>& ft);

  protected:
    std::shared_ptr<PolyMesh2d<SeedType>> mesh;

    vtkSmartPointer<vtkPolyData> polydata;
    vtkSmartPointer<vtkPointData>  pointdata;
    vtkSmartPointer<vtkCellData> celldata;

    vtkSmartPointer<vtkXMLPolyDataWriter> writer;

    vtkSmartPointer<vtkPoints> make_points() const;
    vtkSmartPointer<vtkCellArray> make_cells() const;
    vtkSmartPointer<vtkDoubleArray> make_cell_area() const;
};
}
#endif
