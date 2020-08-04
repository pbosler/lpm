#include "LpmPolyMesh2d.hpp"
#include "LpmKokkosUtil.hpp"
#include "LpmVtkIO.hpp"
#ifdef LPM_HAVE_NETCDF
#include "LpmNetCDF.hpp"
#endif

namespace Lpm {

template <typename SeedType>
void PolyMesh2d<SeedType>::seedInit(const MeshSeed<SeedType>& seed) {
    physVerts.initBoundaryCrdsFromSeed(seed);
    lagVerts.initBoundaryCrdsFromSeed(seed);
    edges.initFromSeed(seed);
    faces.initFromSeed(seed);
    physFaces.initInteriorCrdsFromSeed(seed);
    lagFaces.initInteriorCrdsFromSeed(seed);
}

#ifdef LPM_HAVE_NETCDF
template <typename SeedType>
PolyMesh2d<SeedType>::PolyMesh2d(const PolyMeshReader& reader) :
  physVerts(reader.getVertPhysCrdView()), lagVerts(reader.getVertLagCrdView()),
  edges(reader), faces(reader), physFaces(reader.getFacePhysCrdView()),
  lagFaces(reader.getFaceLagCrdView()) {updateDevice();}
#endif

template <typename SeedType>
void PolyMesh2d<SeedType>::treeInit(const Int initDepth, const MeshSeed<SeedType>& seed) {
    seedInit(seed);
    baseTreeDepth=initDepth;
    for (int i=0; i<initDepth; ++i) {
        Index startInd = 0;
        Index stopInd = faces.nh();
        for (Index j=startInd; j<stopInd; ++j) {
            if (!faces.hasKidsHost(j)) {
                divider::divide(j, physVerts, lagVerts, edges, faces, physFaces, lagFaces);
            }
        }
    }
    updateDevice();
}

template <typename SeedType>
void PolyMesh2d<SeedType>::outputVtk(const std::string& fname) const {
    VtkInterface<Geo,Faces<FaceType>> vtk;
    auto cd = vtkSmartPointer<vtkCellData>::New();
    vtk.addScalarToCellData(cd, faces.getAreaHost(), "area", faces);
    vtkSmartPointer<vtkPolyData> pd = vtk.toVtkPolyData(faces, edges, physFaces, physVerts, NULL, cd);
    vtk.writePolyData(fname, pd);
}

template <typename SeedType>
void PolyMesh2d<SeedType>::updateDevice() const {
    physVerts.updateDevice();
    lagVerts.updateDevice();
    edges.updateDevice();
    faces.updateDevice();
    physFaces.updateDevice();
    lagFaces.updateDevice();
}

template <typename SeedType>
void PolyMesh2d<SeedType>::updateHost() const {
    physVerts.updateHost();
    lagVerts.updateHost();
    faces.updateHost();
    physFaces.updateHost();
    lagFaces.updateHost();
}

template <typename SeedType>
std::string PolyMesh2d<SeedType>::infoString(const std::string& label, const int& tab_level, const bool& dump_all) const {
  std::ostringstream ss;
  ss << "PolyMesh2d " << label << " info:\n";
  ss << physVerts.infoString(label, tab_level+1, dump_all);
  ss << edges.infoString(label, tab_level+1, dump_all);
  ss << faces.infoString(label, tab_level+1, dump_all);
  ss << physFaces.infoString(label, tab_level+1, dump_all);
  return ss.str();
}

/// ETI
template class PolyMesh2d<TriHexSeed>;
template class PolyMesh2d<QuadRectSeed>;
template class PolyMesh2d<IcosTriSphereSeed>;
template class PolyMesh2d<CubedSphereSeed>;
}
