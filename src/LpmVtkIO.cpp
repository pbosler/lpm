#include "LpmVtkIO.hpp"
#include "vtkDoubleArray.h"
#include "vtkIntArray.h"
#include "vtkPoints.h"


namespace Lpm {

template <typename Geo, typename FacesType>
vtkSmartPointer<vtkPolyData> VtkInterface<Geo, FacesType>::toVtkPolyData(const FacesType& faces, const Edges& edges,
    const Coords<Geo>& faceCrds, const Coords<Geo>& vertCrds, const vtkSmartPointer<vtkPointData>& ptdata,
            const vtkSmartPointer<vtkCellData>& cdata) const {
    auto result = vtkSmartPointer<vtkPolyData>::New();
    vtkSmartPointer<vtkPoints> pts = vtkSmartPointer<vtkPoints>::New();
    Real crdvec[Geo::ndim];
    for (Index i=0; i<vertCrds.nh(); ++i) {
        for (int j=0; j<Geo::ndim; ++j) {
            crdvec[j] = vertCrds.getCrdComponentHost(i,j);
        }
        pts->InsertNextPoint(crdvec[0], crdvec[1], (Geo::ndim == 3 ? crdvec[2] : 0.0));
    }
//     std::cout << "vtk points done." << std::endl;
    vtkSmartPointer<vtkCellArray> polys = vtkSmartPointer<vtkCellArray>::New();
    for (Index i=0; i<faces.nh(); ++i) {
        if (!faces.hasKidsHost(i)) {
            polys->InsertNextCell(FacesType::nverts);
            for (int j=0; j<FacesType::nverts; ++j) {
                polys->InsertCellPoint(faces.getVertHost(i,j));
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

namespace Voronoi {
template <typename SeedType>
vtkSmartPointer<vtkPolyData> VtkInterface<SeedType>::toVtkPolyData(const Voronoi::VoronoiMesh<SeedType>& vmesh,
    const vtkSmartPointer<vtkPointData>& ptdata, const vtkSmartPointer<vtkCellData>& cdata) const {

    auto result = vtkSmartPointer<vtkPolyData>::New();

    auto pts = vtkSmartPointer<vtkPoints>::New();
    Real xyz[3];
    for (Index i=0; i<vmesh.vertices.size(); ++i) {
        for (Short j=0; j<3; ++j) {
            xyz[j] = vmesh.vertices[i].xyz[j];
        }
        pts->InsertNextPoint(xyz[0], xyz[1], xyz[2]);
    }

    auto polys = vtkSmartPointer<vtkCellArray>::New();
    std::vector<Index> einds;
    std::vector<Index> vinds;
    for (Index i=0; i<vmesh.cells.size(); ++i) {
        vmesh.getEdgesAndVerticesInCell(einds, vinds, i);
        polys->InsertNextCell(vinds.size());
        for (Short j=0; j<vinds.size(); ++j) {
            polys->InsertCellPoint(vinds[j]);
        }
    }
    result->SetPoints(pts);
    result->SetPolys(polys);

    auto ccr = vtkSmartPointer<vtkDoubleArray>::New();
    ccr->SetName("circumradius");
    ccr->SetNumberOfComponents(1);
    ccr->SetNumberOfTuples(vmesh.vertices.size());
    for (Index i=0; i<vmesh.vertices.size(); ++i) {
        ccr->InsertTuple1(i, vmesh.vertices[i].circumradius);
    }

    auto vid = vtkSmartPointer<vtkIntArray>::New();
    vid->SetName("vertex_id");
    vid->SetNumberOfComponents(1);
    vid->SetNumberOfTuples(vmesh.vertices.size());
    for (Index i=0; i<vmesh.vertices.size(); ++i) {
        vid->InsertTuple1(i, vmesh.vertices[i].id);
    }

    auto ca = vtkSmartPointer<vtkDoubleArray>::New();
    ca->SetName("cell_area");
    ca->SetNumberOfComponents(1);
    ca->SetNumberOfTuples(vmesh.cells.size());
    for (Index i=0; i<vmesh.cells.size(); ++i) {
        ca->InsertTuple1(i, vmesh.cells[i].area);
    }

    auto cid = vtkSmartPointer<vtkIntArray>::New();
    cid->SetName("cell_id");
    cid->SetNumberOfComponents(1);
    cid->SetNumberOfTuples(vmesh.cells.size());
    for (Index i=0; i<vmesh.cells.size(); ++i) {
        cid->InsertTuple1(i, vmesh.cells[i].id);
    }

    vtkSmartPointer<vtkPointData> pdata;
    if (!ptdata) {
        pdata = vtkSmartPointer<vtkPointData>::New();
    }
    else {
        pdata = ptdata;
    }
    pdata->AddArray(ccr);
    pdata->AddArray(vid);
    for (Short i=0; i<pdata->GetNumberOfArrays(); ++i) {
        result->GetPointData()->AddArray(pdata->GetAbstractArray(i));
    }

    vtkSmartPointer<vtkCellData> celldata;
    if (!cdata) {
        celldata = vtkSmartPointer<vtkCellData>::New();
    }
    else {
        celldata = cdata;
    }
    celldata->AddArray(ca);
    celldata->AddArray(cid);
    for (Short i=0; i<celldata->GetNumberOfArrays(); ++i) {
        result->GetCellData()->AddArray(celldata->GetAbstractArray(i));
    }


    return result;
}

template <typename SeedType>
void VtkInterface<SeedType>::writePolyData(const std::string& fname,
    const vtkSmartPointer<vtkPolyData> pd) {
    pdwriter = vtkSmartPointer<vtkPolyDataWriter>::New();
    pdwriter->SetInputData(pd);
    pdwriter->SetFileName(fname.c_str());
    pdwriter->Write();
}
} //namespace Voronoi

template <typename Geo, typename FacesType>
void VtkInterface<Geo, FacesType>::writePolyData(const std::string& fname, const vtkSmartPointer<vtkPolyData> pd) {
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

template <typename Geo, typename FacesType>
void VtkInterface<Geo,FacesType>::addScalarToPointData(vtkSmartPointer<vtkPointData>& pd,
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

template <typename Geo, typename FacesType>
void VtkInterface<Geo,FacesType>::addVectorToPointData(vtkSmartPointer<vtkPointData>& pd,
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

template <typename Geo, typename FacesType>
void VtkInterface<Geo,FacesType>::addScalarToCellData(vtkSmartPointer<vtkCellData>& cd,
    const typename scalar_view_type::HostMirror sf, const std::string& name, const FacesType& faces) const {
    auto data = vtkSmartPointer<vtkDoubleArray>::New();
    data->SetName(name.c_str());
    data->SetNumberOfComponents(1);
    data->SetNumberOfTuples(faces.nLeavesHost());
    Index ctr=0;
    for (Index i=0; i<faces.nh(); ++i) {
        if (!faces.hasKidsHost(i)) {
            data->InsertTuple1(ctr++, sf(i));
        }
    }
    cd->AddArray(data);
}

template <typename Geo, typename FacesType>
void VtkInterface<Geo,FacesType>::addVectorToCellData(vtkSmartPointer<vtkCellData>& cd,
    const typename ko::View<Real*[Geo::ndim],Dev>::HostMirror vf, const std::string& name, const FacesType& faces) const {
    auto data = vtkSmartPointer<vtkDoubleArray>::New();
    data->SetName(name.c_str());
    data->SetNumberOfComponents(Geo::ndim);
    data->SetNumberOfTuples(faces.nLeavesHost());
    Index ctr=0;
    switch (Geo::ndim) {
        case (2) : {
            for (Index i=0; i<faces.nh(); ++i) {
                if (!faces.hasKidsHost(i)) {
                    data->InsertTuple2(ctr++, vf(i,0), vf(i,1));
                }
             }
            break;
        }
        case (3) : {
            for (Index i=0; i<faces.nh(); ++i) {
                if (!faces.hasKidsHost(i)) {
                    data->InsertTuple3(ctr++, vf(i,0), vf(i,1), vf(i,2));
                }
            }
            break;
        }
    }
    cd->AddArray(data);
}




/// ETI
template class VtkInterface<PlaneGeometry, Faces<TriFace>>;
template class VtkInterface<SphereGeometry, Faces<TriFace>>;
template class VtkInterface<PlaneGeometry, Faces<QuadFace>>;
template class VtkInterface<SphereGeometry, Faces<QuadFace>>;
template class VtkInterface<CircularPlaneGeometry, Faces<QuadFace>>;
template class Voronoi::VtkInterface<IcosTriDualSeed>;
}

