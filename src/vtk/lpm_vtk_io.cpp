#include "lpm_vtk_io.hpp"
#include "lpm_vtk_io_impl.hpp"

#ifdef LPM_USE_VTK

#include "vtkDoubleArray.h"
#include "vtkIntArray.h"
#include "vtkPoints.h"


namespace Lpm {

template <typename Geo, typename FaceKind>
vtkSmartPointer<vtkPolyData> VtkInterface<Geo, FaceKind>::toVtkPolyData(const Faces<FaceKind, Geo>& faces, const Edges& edges,
    const Vertices<Coords<Geo>>& verts, const vtkSmartPointer<vtkPointData>& ptdata,
            const vtkSmartPointer<vtkCellData>& cdata) const {
    auto result = vtkSmartPointer<vtkPolyData>::New();
    auto pts = vtkSmartPointer<vtkPoints>::New();
    Real crdvec[Geo::ndim];
    for (Index i=0; i<verts.phys_crds->nh(); ++i) {
        for (int j=0; j<Geo::ndim; ++j) {
            crdvec[j] = verts.phys_crds->get_crd_component_host(i,j);
        }
        pts->InsertNextPoint(crdvec[0], crdvec[1], (Geo::ndim == 3 ? crdvec[2] : 0.0));
    }
//     std::cout << "vtk points done." << std::endl;
    auto polys = vtkSmartPointer<vtkCellArray>::New();
    for (Index i=0; i<faces.nh(); ++i) {
        if (!faces.has_kids_host(i)) {
            polys->InsertNextCell(FaceKind::nverts);
            for (int j=0; j<FaceKind::nverts; ++j) {
                polys->InsertCellPoint(faces.vert_host(i,j));
            }
        }
    }
//     std::cout << "vtk polys done." << std::endl;
    result->SetPoints(pts);
    result->SetPolys(polys);
    if (ptdata) {
        const Index nFields = ptdata->GetNumberOfArrays();
        for (Int i=0; i<nFields; ++i) {
            result->GetPointData()->AddArray(ptdata->GetAbstractArray(i));
        }
    }
    if (cdata) {
        const Index nFields = cdata->GetNumberOfArrays();
        for (Int i=0; i<nFields; ++i) {
            result->GetCellData()->AddArray(cdata->GetAbstractArray(i));
        }
    }
    return result;
}

template <typename Geo, typename FaceKind>
void VtkInterface<Geo, FaceKind>::writePolyData(const std::string& fname, const vtkSmartPointer<vtkPolyData> pd) {
//     std::cout << "starting to write." << std::endl;
//     if (!pdwriter.GetPointer()) {
        pdwriter = vtkSmartPointer<vtkPolyDataWriter>::New();
//         std::cout << "writer initialized." << std::endl;
//     }
    pdwriter->SetInputData(pd);
    pdwriter->SetFileName(fname.c_str());
//     std::cout << "writing." << std::endl;
    pdwriter->Write();
}

template <typename Geo, typename FaceKind>
void VtkInterface<Geo,FaceKind>::addScalarToPointData(vtkSmartPointer<vtkPointData>& pd,
    const typename scalar_view_type::HostMirror sf, const std::string& name, const Index nverts) const {
    auto data = vtkSmartPointer<vtkDoubleArray>::New();
    data->SetName(name.c_str());
    data->SetNumberOfComponents(1);
    data->SetNumberOfTuples(nverts);
    for (Index i=0; i<nverts; ++i) {
        data->InsertTuple1(i, sf(i));
    }
    pd->AddArray(data);
}

template <typename Geo, typename FaceKind>
void VtkInterface<Geo,FaceKind>::addVectorToPointData(vtkSmartPointer<vtkPointData>& pd,
    const typename ko::View<Real*[Geo::ndim],Dev>::HostMirror vf, const std::string& name, const Index nverts) const {
    auto data = vtkSmartPointer<vtkDoubleArray>::New();
    data->SetName(name.c_str());
    data->SetNumberOfComponents(Geo::ndim);
    data->SetNumberOfTuples(nverts);
    switch (Geo::ndim) {
        case (2) : {
            for (Index i=0; i<nverts; ++i) {
                data->InsertTuple2(i, vf(i,0), vf(i,1));
            }
            break;
        }
        case (3) : {
            for (Index i=0; i<nverts; ++i) {
                data->InsertTuple3(i, vf(i,0), vf(i,1), vf(i,2));
            }
            break;
        }
    }
    pd->AddArray(data);
}

template <typename Geo, typename FaceKind>
void VtkInterface<Geo,FaceKind>::addScalarToCellData(vtkSmartPointer<vtkCellData>& cd,
    const typename scalar_view_type::HostMirror sf, const std::string& name, const Faces<FaceKind, Geo>& faces) const {
    auto data = vtkSmartPointer<vtkDoubleArray>::New();
    data->SetName(name.c_str());
    data->SetNumberOfComponents(1);
    data->SetNumberOfTuples(faces.n_leaves_host());
    Index ctr=0;
    for (Index i=0; i<faces.nh(); ++i) {
        if (!faces.has_kids_host(i)) {
            data->InsertTuple1(ctr++, sf(i));
        }
    }
    cd->AddArray(data);
}

template <typename Geo, typename FaceKind>
void VtkInterface<Geo,FaceKind>::addVectorToCellData(vtkSmartPointer<vtkCellData>& cd,
    const typename ko::View<Real*[Geo::ndim],Dev>::HostMirror vf, const std::string& name, const Faces<FaceKind, Geo>& faces) const {
    auto data = vtkSmartPointer<vtkDoubleArray>::New();
    data->SetName(name.c_str());
    data->SetNumberOfComponents(Geo::ndim);
    data->SetNumberOfTuples(faces.n_leaves_host());
    Index ctr=0;
    switch (Geo::ndim) {
        case (2) : {
            for (Index i=0; i<faces.nh(); ++i) {
                if (!faces.has_kids_host(i)) {
                    data->InsertTuple2(ctr++, vf(i,0), vf(i,1));
                }
             }
            break;
        }
        case (3) : {
            for (Index i=0; i<faces.nh(); ++i) {
                if (!faces.has_kids_host(i)) {
                    data->InsertTuple3(ctr++, vf(i,0), vf(i,1), vf(i,2));
                }
            }
            break;
        }
    }
    cd->AddArray(data);
}




/// ETI
template class VtkInterface<PlaneGeometry, TriFace>;
template class VtkInterface<SphereGeometry,TriFace>;
template class VtkInterface<PlaneGeometry, QuadFace>;
template class VtkInterface<SphereGeometry, QuadFace>;
template class VtkPolymeshInterface<CubedSphereSeed>;
template class VtkPolymeshInterface<IcosTriSphereSeed>;
template class VtkPolymeshInterface<QuadRectSeed>;
template class VtkPolymeshInterface<TriHexSeed>;
}

#endif
