#include "lpm_bve_sphere.hpp"
#include "lpm_constants.hpp"
#include "lpm_bve_sphere_kernels.hpp"
#include "lpm_logger.hpp"
#include "mesh/lpm_vertices.hpp"
#include "mesh/lpm_edges.hpp"
#include "mesh/lpm_faces.hpp"
#include "Kokkos_Core.hpp"
#include <memory>
#include <iostream>
namespace Lpm {

template <typename SeedType>
BVESphere<SeedType>::BVESphere(const Index nmaxverts, const Index nmaxedges, const Index nmaxfaces, const Int nq) :
  PolyMesh2d<SeedType>(nmaxverts, nmaxedges, nmaxfaces),
  rel_vort_verts("rel_vort_verts", nmaxverts),
  abs_vort_verts("abs_vort_verts",nmaxverts),
  stream_fn_verts("stream_fn_verts", nmaxverts),
  velocity_verts("velocity_verts", nmaxverts),
  rel_vort_faces("rel_vort_faces", nmaxfaces),
  abs_vort_faces("abs_vort_faces", nmaxfaces),
  stream_fn_faces("stream_fn_faces", nmaxfaces),
  velocity_faces("velocity_faces", nmaxfaces),
  ntracers("ntracers"),
  tracer_verts(nq), _host_tracer_verts(nq),
  tracer_faces(nq), _host_tracer_faces(nq),
  Omega(2*constants::PI),
  omg_set(false),
  t(0)
  {
    ntracers() = nq;
    _host_ntracers = ko::create_mirror_view(ntracers);
    _host_ntracers() = nq;
    _host_rel_vort_verts = ko::create_mirror_view(rel_vort_verts);
    _host_abs_vort_verts = ko::create_mirror_view(abs_vort_verts);
    _host_stream_fn_verts = ko::create_mirror_view(stream_fn_verts);
    _host_velocity_verts = ko::create_mirror_view(velocity_verts);
    _host_rel_vort_faces = ko::create_mirror_view(rel_vort_faces);
    _host_abs_vort_faces = ko::create_mirror_view(abs_vort_faces);
    _host_stream_fn_faces = ko::create_mirror_view(stream_fn_faces);
    _host_velocity_faces = ko::create_mirror_view(velocity_faces);

    std::ostringstream ss;
    for (int k=0; k<nq; ++k) {
      ss << "tracer" << k;
      tracer_verts[k] = scalar_field(ss.str(), nmaxverts);
      tracer_faces[k] = scalar_field(ss.str(), nmaxfaces);
      ss.str("");
      _host_tracer_verts[k] = ko::create_mirror_view(tracer_verts[k]);
      _host_tracer_faces[k] = ko::create_mirror_view(tracer_faces[k]);
    }
  }

template <typename SeedType>
BVESphere<SeedType>::BVESphere(const Index nmaxverts, const Index nmaxedges, const Index nmaxfaces, const std::vector<std::string>& tracers) :
  PolyMesh2d<SeedType>(nmaxverts, nmaxedges, nmaxfaces),
  rel_vort_verts("rel_vort_verts", nmaxverts),
  abs_vort_verts("abs_vort_verts",nmaxverts),
  stream_fn_verts("stream_fn_verts", nmaxverts),
  velocity_verts("velocity_verts", nmaxverts),
  rel_vort_faces("rel_vort_faces", nmaxfaces),
  abs_vort_faces("abs_vort_faces", nmaxfaces),
  stream_fn_faces("stream_fn_faces", nmaxfaces),
  velocity_faces("velocity_faces", nmaxfaces),
  ntracers("ntracers"),
  tracer_verts(tracers.size()), _host_tracer_verts(tracers.size()),
  tracer_faces(tracers.size()), _host_tracer_faces(tracers.size()),
  Omega(2*constants::PI),
  omg_set(false),
  t(0)
  {
    const int nq = tracers.size();
    _host_ntracers = ko::create_mirror_view(ntracers);
    _host_ntracers() = nq;
    _host_rel_vort_verts = ko::create_mirror_view(rel_vort_verts);
    _host_abs_vort_verts = ko::create_mirror_view(abs_vort_verts);
    _host_stream_fn_verts = ko::create_mirror_view(stream_fn_verts);
    _host_velocity_verts = ko::create_mirror_view(velocity_verts);
    _host_rel_vort_faces = ko::create_mirror_view(rel_vort_faces);
    _host_abs_vort_faces = ko::create_mirror_view(abs_vort_faces);
    _host_stream_fn_faces = ko::create_mirror_view(stream_fn_faces);
    _host_velocity_faces = ko::create_mirror_view(velocity_faces);

    for (int k=0; k<tracers.size(); ++k) {
      tracer_verts[k] = scalar_field(tracers[k], nmaxverts);
      tracer_faces[k] = scalar_field(tracers[k], nmaxfaces);
      _host_tracer_verts[k] = ko::create_mirror_view(tracer_verts[k]);
      _host_tracer_faces[k] = ko::create_mirror_view(tracer_faces[k]);
    }
  }


template <typename SeedType>
void BVESphere<SeedType>::update_device() const {
    PolyMesh2d<SeedType>::update_device();
    ko::deep_copy(rel_vort_verts, _host_rel_vort_verts);
    ko::deep_copy(abs_vort_verts, _host_abs_vort_verts);
    ko::deep_copy(stream_fn_verts, _host_stream_fn_verts);
    ko::deep_copy(velocity_verts, _host_velocity_verts);
    ko::deep_copy(rel_vort_faces, _host_rel_vort_faces);
    ko::deep_copy(abs_vort_faces, _host_abs_vort_faces);
    ko::deep_copy(stream_fn_faces, _host_stream_fn_faces);
    ko::deep_copy(velocity_faces, _host_velocity_faces);
    ko::deep_copy(ntracers, _host_ntracers);
    for (Int k=0; k<_host_ntracers(); ++k) {
      ko::deep_copy(tracer_verts[k], _host_tracer_verts[k]);
      ko::deep_copy(tracer_faces[k], _host_tracer_faces[k]);
    }
}

template <typename SeedType>
void BVESphere<SeedType>::update_host() const {
    PolyMesh2d<SeedType>::update_host();
    ko::deep_copy(_host_rel_vort_verts, rel_vort_verts);
    ko::deep_copy(_host_abs_vort_verts, abs_vort_verts);
    ko::deep_copy(_host_stream_fn_verts, stream_fn_verts);
    ko::deep_copy(_host_velocity_verts, velocity_verts);
    ko::deep_copy(_host_rel_vort_faces, rel_vort_faces);
    ko::deep_copy(_host_abs_vort_faces, abs_vort_faces);
    ko::deep_copy(_host_stream_fn_faces, stream_fn_faces);
    ko::deep_copy(_host_velocity_faces, velocity_faces);
}

#ifdef LPM_USE_VTK
template <typename SeedType>
VtkPolymeshInterface<SeedType> vtk_interface(const std::shared_ptr<BVESphere<SeedType>> bve) {
    std::shared_ptr<PolyMesh2d<SeedType>> base_ptr(bve);
    VtkPolymeshInterface<SeedType> vtk(base_ptr);
    vtk.add_scalar_point_data(bve->rel_vort_verts, "rel_vort");
    vtk.add_scalar_point_data(bve->abs_vort_verts, "abs_vort");
    vtk.add_scalar_point_data(bve->stream_fn_verts, "stream_fn");
    vtk.add_vector_point_data(bve->velocity_verts, "velocity");

    vtk.add_scalar_cell_data(bve->rel_vort_faces, "rel_vort");
    vtk.add_scalar_cell_data(bve->abs_vort_faces, "abs_vort");
    vtk.add_scalar_cell_data(bve->stream_fn_faces, "stream_fn");
    vtk.add_vector_cell_data(bve->velocity_faces, "velocity");

    for (Short i=0; i<bve->tracer_verts.size(); ++i) {
      vtk.add_scalar_point_data(bve->tracer_verts[i], bve->tracer_verts[i].label());
      vtk.add_scalar_cell_data(bve->tracer_faces[i], bve->tracer_faces[i].label());
    }
    return vtk;
}
#endif

template <typename SeedType>
void BVESphere<SeedType>::set_omega(const Real& omg) {
  if (omg_set) {
    Log::warn("BVESphere::set_omega warning: omega = {} already set.", Omega);
  }
  else {
    Omega = omg;
    omg_set = true;
  }
}

template <typename SeedType>
Real BVESphere<SeedType>::avg_mesh_size_radians() const {
  return this->faces.appx_mesh_size();
}

template <typename SeedType>
Real BVESphere<SeedType>::avg_mesh_size_degrees() const {
  return constants::RAD2DEG * avg_mesh_size_radians();
}

template <typename SeedType> template <typename VorticityInitialCondition>
void BVESphere<SeedType>::init_vorticity(const VorticityInitialCondition& vorticity_fn) {

  auto zeta_verts = this->rel_vort_verts;
  auto omega_verts = this->abs_vort_verts;
  auto vert_crds = this->vertices.phys_crds->crds;
  Real Omg = this->Omega;
  Kokkos::parallel_for(this->n_vertices_host(), KOKKOS_LAMBDA (const Index i) {
    const auto mxyz = Kokkos::subview(vert_crds, i, Kokkos::ALL);
    const Real zeta = vorticity_fn(mxyz(0), mxyz(1), mxyz(2));
    zeta_verts(i) = zeta;
    omega_verts(i) = zeta + 2*Omg*mxyz(2);
  });

  auto zeta_faces = this->rel_vort_faces;
  auto omega_faces = this->abs_vort_faces;
  auto face_crds = this->faces.phys_crds->crds;
  Kokkos::parallel_for(this->n_faces_host(), KOKKOS_LAMBDA (const Index i) {
    const auto mxyz = Kokkos::subview(face_crds, i, Kokkos::ALL);
    const Real zeta = vorticity_fn(mxyz(0), mxyz(1), mxyz(2));
    zeta_faces(i) = zeta;
    omega_faces(i) = zeta + 2*Omg*mxyz(2);
  });

  Kokkos::deep_copy(_host_rel_vort_verts, rel_vort_verts);
  Kokkos::deep_copy(_host_rel_vort_faces, rel_vort_faces);
  Kokkos::deep_copy(_host_abs_vort_verts, abs_vort_verts);
  Kokkos::deep_copy(_host_abs_vort_faces, abs_vort_faces);
}

template <typename SeedType>
void BVESphere<SeedType>::init_velocity() {
  ko::TeamPolicy<> vertex_policy(this->n_vertices_host(), ko::AUTO());
  ko::TeamPolicy<> face_policy(this->n_faces_host(), ko::AUTO());

  Kokkos::Profiling::pushRegion("BVESphere::init_velocity");

  Kokkos::parallel_for("BVESphere::init_velocity_vertices", vertex_policy,
    BVEVertexVelocity(velocity_verts, this->vertices.phys_crds->crds,
      this->faces.phys_crds->crds, rel_vort_faces, this->faces.area, this->faces.mask,
      this->n_faces_host()));

  Kokkos::parallel_for("BVESphere::init_velocity_faces", face_policy,
    BVEFaceVelocity(velocity_faces, this->faces.phys_crds->crds, rel_vort_faces,
    this->faces.area, this->faces.mask, this->n_faces_host()));

  Kokkos::Profiling::popRegion();
}

template <typename SeedType>
void BVESphere<SeedType>::init_stream_fn() {
  ko::TeamPolicy<> vertex_policy(this->n_vertices_host(), ko::AUTO());
  ko::TeamPolicy<> face_policy(this->n_faces_host(), ko::AUTO());

  Kokkos::Profiling::pushRegion("BVESphere::init_stream_fn");

  Kokkos::parallel_for("BVESphere::init_stream_fn_vertices", vertex_policy,
    BVEVertexStreamFn(stream_fn_verts, this->vertices.phys_crds->crds,
      this->faces.phys_crds->crds, rel_vort_faces, this->faces.area,
      this->faces.mask, this->n_faces_host()));

  Kokkos::parallel_for("BVESphere::init_stream_fn_faces", face_policy,
    BVEFaceStreamFn(stream_fn_faces, this->faces.phys_crds->crds,
      rel_vort_faces, this->faces.area, this->faces.mask, this->n_faces_host()));

  Kokkos::Profiling::popRegion();
}


}
