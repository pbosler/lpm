#ifndef LPM_VTK_IO_HPP
#define LPM_VTK_IO_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmUtilities.hpp"
#include "LpmGeometry.hpp"
#include "LpmCoords.hpp"
#include "LpmEdges.hpp"
#include "LpmFaces.hpp"
#include "LpmMeshSeed.hpp"
#include "Kokkos_Core.hpp"
#include "Kokkos_View.hpp"

#include "vtkSmartPointer.h"
#include "vtkPolyData.h"
#include "vtkPolyDataWriter.h"
#include "vtkPointData.h"
#include "vtkCellData.h"

namespace Lpm {

template <typename Geo, typename FacesType> class VtkInterface {
    public:
        vtkSmartPointer<vtkPolyData> toVtkPolyData(const FacesType& faces, const Edges& edges, 
            const Coords<Geo>& faceCrds, const Coords<Geo>& vertCrds, const vtkSmartPointer<vtkPointData>& ptdata=0,
            const vtkSmartPointer<vtkCellData>& cdata=0) const ;
    
        void writePolyData(const std::string& fname, const vtkSmartPointer<vtkPolyData> pd);
        
        void addScalarToPointData(vtkSmartPointer<vtkPointData>& pd, 
            const typename scalar_view_type::HostMirror sf, const std::string& name, const Index nverts) const;
        
        void addVectorToPointData(vtkSmartPointer<vtkPointData>& pd, 
            const typename ko::View<Real*[Geo::ndim],Dev>::HostMirror vf, const std::string& name, const Index nverts) const;
        
        void addScalarToCellData(vtkSmartPointer<vtkCellData>& cd, 
            const typename scalar_view_type::HostMirror sf, const std::string& name, const FacesType& faces) const;
        
        void addVectorToCellData(vtkSmartPointer<vtkCellData>& cd, 
            const typename ko::View<Real*[Geo::ndim],Dev>::HostMirror vf, const std::string& name, const FacesType& faces) const;
        
    protected:
        vtkSmartPointer<vtkPolyDataWriter> pdwriter;
};

}
#endif
