#include "LpmVtkIO.hpp"
#include "vtkDoubleArray.h"

namespace Lpm {

template <typename Geo, typename FacesType> 
void VtkInterface<Geo, FacesType>::toVtkPolyData(const FacesType& faces, const Edges& edges, 
    const Coords<Geo>& faceCrds, const Coords<Geo>& vertCrds) {
    pd = vtkSmartPointer<vtkPolyData>::New();
    vtkSmartPointer<vtkPoints> pts = vtkSmartPointer<vtkPoints>::New();
    for (Index i=0; i<vertCrds.nh(); ++i) {
        auto crdvec = vertCrds.crdVecHostConst(i);
        pts->InsertNextPoint(crdvec(0), crdvec(1), (Geo::ndim == 3 ? crdvec(2) : 0.0));
    }
    std::cout << "vtk points done." << std::endl;
    vtkSmartPointer<vtkCellArray> polys = vtkSmartPointer<vtkCellArray>::New();
    for (Index i=0; i<faces.nh(); ++i) {
        if (!faces.hasKidsHost(i)) {
            polys->InsertNextCell(FacesType::nverts);
            auto verts = faces.getVertsHostConst(i);
            for (int j=0; j<FacesType::nverts; ++j) {
                polys->InsertCellPoint(verts(j));
            }
        }
    }
    std::cout << "vtk polys done." << std::endl;
    pd->SetPoints(pts);
    pd->SetPolys(polys);
}

template <typename Geo, typename FacesType>
void VtkInterface<Geo, FacesType>::writePolyData(const std::string& fname) {
    std::cout << "starting to write." << std::endl;
//     if (!pdwriter.GetPointer()) {
        pdwriter = vtkSmartPointer<vtkPolyDataWriter>::New();
        std::cout << "writer initialized." << std::endl;
//     }
    pdwriter->SetInputData(pd);
    pdwriter->SetFileName(fname.c_str());
    std::cout << "writing." << std::endl;
    pdwriter->Write();
}


/// ETI
template class VtkInterface<PlaneGeometry, Faces<TriFace>>;
template class VtkInterface<SphereGeometry, Faces<TriFace>>;
template class VtkInterface<PlaneGeometry, Faces<QuadFace>>;
template class VtkInterface<SphereGeometry, Faces<QuadFace>>;
}

