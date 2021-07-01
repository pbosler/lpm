#ifndef LPM_VTK_IO_HPP
#define LPM_VTK_IO_HPP

#include "LpmConfig.h"

#include "lpm_geometry.hpp"
#include "lpm_coords.hpp"

#include "mesh/lpm_mesh_seed.hpp"
#include "mesh/lpm_vertices.hpp"
#include "mesh/lpm_edges.hpp"
#include "mesh/lpm_faces.hpp"

#include "Kokkos_Core.hpp"
#include "Kokkos_View.hpp"

#include <vector>

#include "vtkSmartPointer.h"
#include "vtkPolyData.h"
#include "vtkPolyDataWriter.h"
#include "vtkPointData.h"
#include "vtkCellData.h"

namespace Lpm {

template <typename Geo, typename FaceKind> class VtkInterface {
    public:
        vtkSmartPointer<vtkPolyData> toVtkPolyData(const Faces<FaceKind,Geo>& faces, const Edges& edges,
            const Vertices<Coords<Geo>>& verts, const vtkSmartPointer<vtkPointData>& ptdata=0,
            const vtkSmartPointer<vtkCellData>& cdata=0) const ;

        void writePolyData(const std::string& fname, const vtkSmartPointer<vtkPolyData> pd);

        void addScalarToPointData(vtkSmartPointer<vtkPointData>& pd,
            const typename scalar_view_type::HostMirror sf, const std::string& name, const Index nverts) const;

        void addVectorToPointData(vtkSmartPointer<vtkPointData>& pd,
            const typename ko::View<Real*[Geo::ndim],Dev>::HostMirror vf, const std::string& name, const Index nverts) const;

        void addScalarToCellData(vtkSmartPointer<vtkCellData>& cd,
            const typename scalar_view_type::HostMirror sf, const std::string& name, const Faces<FaceKind,Geo>& faces) const;

        void addVectorToCellData(vtkSmartPointer<vtkCellData>& cd,
            const typename ko::View<Real*[Geo::ndim],Dev>::HostMirror vf, const std::string& name, const Faces<FaceKind,Geo>& faces) const;

    protected:
        vtkSmartPointer<vtkPolyDataWriter> pdwriter;
};


}
#endif
