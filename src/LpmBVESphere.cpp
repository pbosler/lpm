#include "LpmBVESphere.hpp"
#include "LpmVtkIO.hpp"
#include "LpmEdges.hpp"
#include "LpmFaces.hpp"
#include "LpmCoords.hpp"
#include "LpmPolyMesh2d.hpp"

namespace Lpm {

template <typename FaceType> 
void BVESphere<FaceType>::updateDevice() const {
    PolyMesh2d<SphereGeometry,FaceType>::updateDevice();
    ko::deep_copy(relVortVerts, _hostRelVortVerts);
    ko::deep_copy(absVortVerts, _hostAbsVortVerts);
    ko::deep_copy(streamFnVerts, _hostStreamFnVerts);
    ko::deep_copy(velocityVerts, _hostVelocityVerts);
    ko::deep_copy(relVortFaces, _hostRelVortFaces);
    ko::deep_copy(absVortFaces, _hostAbsVortFaces);
    ko::deep_copy(streamFnFaces, _hostStreamFnFaces);
    ko::deep_copy(velocityFaces, _hostVelocityFaces);    
}

template <typename FaceType>
void BVESphere<FaceType>::updateHost() const {
    PolyMesh2d<SphereGeometry,FaceType>::updateHost();
    ko::deep_copy(_hostRelVortVerts, relVortVerts);
    ko::deep_copy(_hostAbsVortVerts, absVortVerts);
    ko::deep_copy(_hostStreamFnVerts, streamFnVerts);
    ko::deep_copy(_hostVelocityVerts, velocityVerts);
    ko::deep_copy(_hostRelVortFaces, relVortFaces);
    ko::deep_copy(_hostAbsVortFaces, absVortFaces);
    ko::deep_copy(_hostStreamFnFaces, streamFnFaces);
    ko::deep_copy(_hostVelocityFaces, velocityFaces);    
}

template <typename FaceType>
void BVESphere<FaceType>::outputVtk(const std::string& fname) const {
    VtkInterface<SphereGeometry,Faces<FaceType>> vtk;
    auto ptdata = vtkSmartPointer<vtkPointData>::New();
    vtk.addScalarToPointData(ptdata, _hostRelVortVerts, "relVort", this->physVerts.nh());
    vtk.addScalarToPointData(ptdata, _hostAbsVortVerts, "absVort", this->physVerts.nh());
    vtk.addScalarToPointData(ptdata, _hostStreamFnVerts, "streamFn", this->physVerts.nh());
    vtk.addVectorToPointData(ptdata, _hostVelocityVerts, "velocity", this->physVerts.nh());
    
    auto celldata = vtkSmartPointer<vtkCellData>::New();
    vtk.addScalarToCellData(celldata, this->faces.getAreaHost(), "area", this->faces);
    vtk.addScalarToCellData(celldata, _hostRelVortFaces, "relVort", this->faces);
    vtk.addScalarToCellData(celldata, _hostAbsVortFaces, "absVort", this->faces);
    vtk.addScalarToCellData(celldata, _hostStreamFnFaces, "streamFn", this->faces);
    vtk.addVectorToCellData(celldata, _hostVelocityFaces, "velocity", this->faces);
    
    auto pd = vtk.toVtkPolyData(this->faces, this->edges, this->physFaces, this->physVerts, ptdata, celldata);
    vtk.writePolyData(fname, pd);
}


/// ETI
template class BVESphere<TriFace>;
template class BVESphere<QuadFace>;

}
