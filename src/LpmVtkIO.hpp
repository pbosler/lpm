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
#include "vtkPoints.h"
#include "vtkPointData.h"
#include "vtkCellData.h"
#include "vtkPolyData.h"
#include "vtkPolyDataWriter.h"

namespace Lpm {

template <typename Geo, typename FacesType> class VtkInterface {
    public:
        void toVtkPolyData(const FacesType& faces, const Edges& edges, 
            const Coords<Geo>& faceCrds, const Coords<Geo>& vertCrds);
    
        void writePolyData(const std::string& fname);
        
    protected:
        vtkSmartPointer<vtkPolyData> pd;
        vtkSmartPointer<vtkPolyDataWriter> pdwriter;
};

}
#endif
