#include "LpmBVESphere.hpp"
#include "LpmVtkIO.hpp"
#include "LpmEdges.hpp"
#include "LpmFaces.hpp"
#include "LpmCoords.hpp"
#include "LpmPolyMesh2d.hpp"

namespace Lpm {

template <typename SeedType>
BVESphere<SeedType>::BVESphere(const Index nmaxverts, const Index nmaxedges, const Index nmaxfaces, const Int nq) :
  PolyMesh2d<SeedType>(nmaxverts, nmaxedges, nmaxfaces),
  relVortVerts("relVortVerts", nmaxverts),
  absVortVerts("absVortVerts",nmaxverts),
  streamFnVerts("streamFnVerts", nmaxverts),
  velocityVerts("velocityVerts", nmaxverts),
  relVortFaces("relVortFaces", nmaxfaces),
  absVortFaces("absVortFaces", nmaxfaces),
  streamFnFaces("streamFnFaces", nmaxfaces),
  velocityFaces("velocityFaces", nmaxfaces),
  ntracers("ntracers"),
  tracer_verts(nq), _hostTracerVerts(nq),
  tracer_faces(nq), _hostTracerFaces(nq),
  Omega(2*PI),
  omg_set(false)
  {
    ntracers() = nq;
    _hostntracers = ko::create_mirror_view(ntracers);
    _hostntracers() = nq;
    _hostRelVortVerts = ko::create_mirror_view(relVortVerts);
    _hostAbsVortVerts = ko::create_mirror_view(absVortVerts);
    _hostStreamFnVerts = ko::create_mirror_view(streamFnVerts);
    _hostVelocityVerts = ko::create_mirror_view(velocityVerts);
    _hostRelVortFaces = ko::create_mirror_view(relVortFaces);
    _hostAbsVortFaces = ko::create_mirror_view(absVortFaces);
    _hostStreamFnFaces = ko::create_mirror_view(streamFnFaces);
    _hostVelocityFaces = ko::create_mirror_view(velocityFaces);

    std::ostringstream ss;
    for (int k=0; k<nq; ++k) {
      ss << "tracer" << k;
      tracer_verts[k] = scalar_field(ss.str(), nmaxverts);
      tracer_faces[k] = scalar_field(ss.str(), nmaxfaces);
      ss.str("");
      _hostTracerVerts[k] = ko::create_mirror_view(tracer_verts[k]);
      _hostTracerFaces[k] = ko::create_mirror_view(tracer_faces[k]);
    }
  }

template <typename SeedType>
Short BVESphere<SeedType>::create_tracer(const std::string& name) {
  const Short tracer_ind = tracer_verts.size();
  tracer_verts.push_back(scalar_field(name, relVortVerts.extent(0)));
  _hostTracerVerts.push_back(ko::create_mirror_view(tracer_verts[tracer_ind]));
  tracer_faces.push_back(scalar_field(name, relVortFaces.extent(0)));
  _hostTracerFaces.push_back(ko::create_mirror_view(tracer_faces[tracer_ind]));
  return tracer_ind;
}

template <typename SeedType>
void BVESphere<SeedType>::updateDevice() const {
    PolyMesh2d<SeedType>::updateDevice();
    ko::deep_copy(relVortVerts, _hostRelVortVerts);
    ko::deep_copy(absVortVerts, _hostAbsVortVerts);
    ko::deep_copy(streamFnVerts, _hostStreamFnVerts);
    ko::deep_copy(velocityVerts, _hostVelocityVerts);
    ko::deep_copy(relVortFaces, _hostRelVortFaces);
    ko::deep_copy(absVortFaces, _hostAbsVortFaces);
    ko::deep_copy(streamFnFaces, _hostStreamFnFaces);
    ko::deep_copy(velocityFaces, _hostVelocityFaces);
    for (Int k=0; k<_hostntracers(); ++k) {
      ko::deep_copy(tracer_verts[k], _hostTracerVerts[k]);
      ko::deep_copy(tracer_faces[k], _hostTracerFaces[k]);
    }
}

template <typename SeedType>
void BVESphere<SeedType>::updateHost() const {
    PolyMesh2d<SeedType>::updateHost();
    ko::deep_copy(_hostRelVortVerts, relVortVerts);
    ko::deep_copy(_hostAbsVortVerts, absVortVerts);
    ko::deep_copy(_hostStreamFnVerts, streamFnVerts);
    ko::deep_copy(_hostVelocityVerts, velocityVerts);
    ko::deep_copy(_hostRelVortFaces, relVortFaces);
    ko::deep_copy(_hostAbsVortFaces, absVortFaces);
    ko::deep_copy(_hostStreamFnFaces, streamFnFaces);
    ko::deep_copy(_hostVelocityFaces, velocityFaces);
}

template <typename SeedType>
void BVESphere<SeedType>::outputVtk(const std::string& fname) const {
    VtkInterface<SphereGeometry,Faces<typename SeedType::faceKind>> vtk;
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

    for (Short i=0; i<tracer_verts.size(); ++i) {
      vtk.addScalarToPointData(ptdata, _hostTracerVerts[i], _hostTracerVerts[i].label(), this->physVerts.nh());
      vtk.addScalarToCellData(celldata, _hostTracerFaces[i], _hostTracerFaces[i].label(), this->faces);
    }

    auto pd = vtk.toVtkPolyData(this->faces, this->edges, this->physFaces, this->physVerts, ptdata, celldata);
    vtk.writePolyData(fname, pd);
}

template <typename SeedType>
void BVESphere<SeedType>::set_omega(const Real& omg) {
  if (omg_set) {
    std::cout << "BVESphere::set_omega warning: omega = " << Omega << " already set.\n";
  }
  else {
    Omega = omg;
    omg_set = true;
  }
}

template <typename SeedType>
void BVESphere<SeedType>::init_vorticity(const VorticityInitialCondition::ptr relvort) {
  const auto hvertx = this->physVerts.getHostCrdView();
  # pragma omp parallel for
  for (Index i=0; i<this->nvertsHost(); ++i) {
    const auto mxyz = ko::subview(hvertx, i, ko::ALL());
    const Real zeta = relvort->eval(mxyz(0), mxyz(1), mxyz(2));
    _hostRelVortVerts(i) = zeta;
    _hostAbsVortVerts(i) = zeta + 2*Omega*mxyz(2);
  }
  ko::deep_copy(relVortVerts, _hostRelVortVerts);
  ko::deep_copy(absVortVerts, _hostAbsVortVerts);

  const auto hfacex = this->physFaces.getHostCrdView();
  # pragma omp parallel for
  for (Index i=0; i<this->nfacesHost(); ++i) {
    const auto mxyz = ko::subview(hfacex, i, ko::ALL());
    const Real zeta = relvort->eval(mxyz(0), mxyz(1), mxyz(2));
    _hostRelVortFaces(i) = zeta;
    _hostAbsVortFaces(i) = zeta + 2*Omega*mxyz(2);
  }

  ko::deep_copy(relVortFaces, _hostRelVortFaces);
  ko::deep_copy(absVortFaces, _hostAbsVortFaces);

  ko::TeamPolicy<> vertex_policy(this->nvertsHost(), ko::AUTO());
  ko::TeamPolicy<> face_policy(this->nfacesHost(), ko::AUTO());

  ko::parallel_for("init_vorticity: stream fn verts", vertex_policy,
    BVEVertexStreamFn(streamFnVerts, this->physVerts.crds, this->physFaces.crds, this->relVortFaces, this->faces.area,
      this->faces.mask, this->faces.nh()));

  ko::parallel_for("init_vorticity: stream fn faces", face_policy,
    BVEFaceStreamFn(streamFnFaces, this->physFaces.crds, this->relVortFaces, this->faces.area,
      this->faces.mask, this->faces.nh()));

  ko::parallel_for("init_vorticity: velocity verts", vertex_policy,
    BVEVertexVelocity(velocityVerts, this->physVerts.crds, this->physFaces.crds, this->relVortFaces, this->faces.area,
      this->faces.mask, this->faces.nh()));

  ko::parallel_for("init_vorticity: velocity faces", face_policy,
    BVEFaceVelocity(velocityFaces, this->physFaces.crds, this->relVortFaces, this->faces.area, this->faces.mask,
      this->faces.nh()));
}


/// ETI
template class BVESphere<IcosTriSphereSeed>;
template class BVESphere<CubedSphereSeed>;

}
