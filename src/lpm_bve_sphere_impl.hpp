#include <iostream>
#include <memory>

#include "Kokkos_Core.hpp"
#include "lpm_bve_sphere.hpp"
#include "lpm_bve_sphere_kernels.hpp"
#include "lpm_constants.hpp"
#include "lpm_logger.hpp"
#include "mesh/lpm_edges.hpp"
#include "mesh/lpm_faces.hpp"
#include "mesh/lpm_vertices.hpp"
namespace Lpm {

template <typename SeedType>
BVESphere<SeedType>::BVESphere(const Index nmaxverts, const Index nmaxedges,
                               const Index nmaxfaces, const Int nq)
    : PolyMesh2d<SeedType>(nmaxverts, nmaxedges, nmaxfaces),
      rel_vort_verts("rel_vort_verts", nmaxverts),
      abs_vort_verts("abs_vort_verts", nmaxverts),
      stream_fn_verts("stream_fn_verts", nmaxverts),
      velocity_verts("velocity_verts", nmaxverts),
      rel_vort_faces("rel_vort_faces", nmaxfaces),
      abs_vort_faces("abs_vort_faces", nmaxfaces),
      stream_fn_faces("stream_fn_faces", nmaxfaces),
      velocity_faces("velocity_faces", nmaxfaces),
      ntracers("ntracers"),
      Omega(2 * constants::PI),
      omg_set(false),
      t(0) {
  ntracers() = nq;
  _host_ntracers = ko::create_mirror_view(ntracers);
  _host_ntracers() = nq;

  std::ostringstream ss;
  for (int k = 0; k < nq; ++k) {
    ss << "tracer" << k;
    tracer_verts.push_back(ScalarField<VertexField>(ss.str(), nmaxverts));
    tracer_faces.push_back(ScalarField<FaceField>(ss.str(), nmaxfaces));
    ss.str("");
  }
}

template <typename SeedType>
BVESphere<SeedType>::BVESphere(const Index nmaxverts, const Index nmaxedges,
                               const Index nmaxfaces,
                               const std::vector<std::string>& tracers)
    : PolyMesh2d<SeedType>(nmaxverts, nmaxedges, nmaxfaces),
      rel_vort_verts("rel_vort_verts", nmaxverts),
      abs_vort_verts("abs_vort_verts", nmaxverts),
      stream_fn_verts("stream_fn_verts", nmaxverts),
      velocity_verts("velocity_verts", nmaxverts),
      rel_vort_faces("rel_vort_faces", nmaxfaces),
      abs_vort_faces("abs_vort_faces", nmaxfaces),
      stream_fn_faces("stream_fn_faces", nmaxfaces),
      velocity_faces("velocity_faces", nmaxfaces),
      ntracers("ntracers"),
      Omega(2 * constants::PI),
      omg_set(false),
      t(0) {
  const int nq = tracers.size();
  _host_ntracers = ko::create_mirror_view(ntracers);
  _host_ntracers() = nq;

  for (int k = 0; k < tracers.size(); ++k) {
    tracer_verts.push_back(ScalarField<VertexField>(tracers[k], nmaxverts));
    tracer_faces.push_back(ScalarField<FaceField>(tracers[k], nmaxfaces));
  }
}

template <typename SeedType>
void BVESphere<SeedType>::update_device() const {
  PolyMesh2d<SeedType>::update_device();
  rel_vort_verts.update_device();
  abs_vort_verts.update_device();
  stream_fn_verts.update_device();
  velocity_verts.update_device();
  rel_vort_faces.update_device();
  abs_vort_faces.update_device();
  stream_fn_faces.update_device();
  velocity_faces.update_device();
  for (Int k = 0; k < _host_ntracers(); ++k) {
    tracer_verts[k].update_device();
    tracer_faces[k].update_device();
  }
}

template <typename SeedType>
void BVESphere<SeedType>::update_host() const {
  PolyMesh2d<SeedType>::update_host();
  rel_vort_verts.update_host();
  abs_vort_verts.update_host();
  stream_fn_verts.update_host();
  velocity_verts.update_host();
  rel_vort_faces.update_host();
  abs_vort_faces.update_host();
  stream_fn_faces.update_host();
  velocity_faces.update_host();
  for (Int k = 0; k < _host_ntracers(); ++k) {
    tracer_verts[k].update_host();
    tracer_faces[k].update_host();
  }
}

#ifdef LPM_USE_VTK
template <typename SeedType>
VtkPolymeshInterface<SeedType> vtk_interface(
    const BVESphere<SeedType>& bve) {
  VtkPolymeshInterface<SeedType> vtk(bve);
  vtk.add_scalar_point_data(bve.rel_vort_verts.view, "rel_vort");
  vtk.add_scalar_point_data(bve.abs_vort_verts.view, "abs_vort");
  vtk.add_scalar_point_data(bve.stream_fn_verts.view, "stream_fn");
  vtk.add_vector_point_data(bve.velocity_verts.view, "velocity");

  vtk.add_scalar_cell_data(bve.rel_vort_faces.view, "rel_vort");
  vtk.add_scalar_cell_data(bve.abs_vort_faces.view, "abs_vort");
  vtk.add_scalar_cell_data(bve.stream_fn_faces.view, "stream_fn");
  vtk.add_vector_cell_data(bve.velocity_faces.view, "velocity");

  for (Short i = 0; i < bve.tracer_verts.size(); ++i) {
    vtk.add_scalar_point_data(bve.tracer_verts[i].view,
                              bve.tracer_verts[i].view.label());
    vtk.add_scalar_cell_data(bve.tracer_faces[i].view,
                             bve.tracer_faces[i].view.label());
  }
  return vtk;
}
#endif

template <typename SeedType>
void BVESphere<SeedType>::set_omega(const Real& omg) {
  if (omg_set) {
    Log::warn("BVESphere::set_omega warning: omega = {} already set.", Omega);
  } else {
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

template <typename SeedType>
template <typename VorticityInitialCondition>
void BVESphere<SeedType>::init_vorticity(
    const VorticityInitialCondition& vorticity_fn) {
  auto zeta_verts = this->rel_vort_verts.view;
  auto omega_verts = this->abs_vort_verts.view;
  auto vert_crds = this->vertices.phys_crds.view;
  Real Omg = this->Omega;
  Kokkos::parallel_for(
      this->n_vertices_host(), KOKKOS_LAMBDA(const Index i) {
        const auto mxyz = Kokkos::subview(vert_crds, i, Kokkos::ALL);
        const Real zeta = vorticity_fn(mxyz(0), mxyz(1), mxyz(2));
        zeta_verts(i) = zeta;
        omega_verts(i) = zeta + 2 * Omg * mxyz(2);
      });

  auto zeta_faces = this->rel_vort_faces.view;
  auto omega_faces = this->abs_vort_faces.view;
  auto face_crds = this->faces.phys_crds.view;
  Kokkos::parallel_for(
      this->n_faces_host(), KOKKOS_LAMBDA(const Index i) {
        const auto mxyz = Kokkos::subview(face_crds, i, Kokkos::ALL);
        const Real zeta = vorticity_fn(mxyz(0), mxyz(1), mxyz(2));
        zeta_faces(i) = zeta;
        omega_faces(i) = zeta + 2 * Omg * mxyz(2);
      });

  this->rel_vort_verts.update_host();
  this->abs_vort_verts.update_host();
  this->rel_vort_faces.update_host();
  this->abs_vort_faces.update_host();
}

template <typename SeedType>
void BVESphere<SeedType>::init_velocity() {
  ko::TeamPolicy<> vertex_policy(this->n_vertices_host(), ko::AUTO());
  ko::TeamPolicy<> face_policy(this->n_faces_host(), ko::AUTO());

  Kokkos::Profiling::pushRegion("BVESphere::init_velocity");

  Kokkos::parallel_for(
      "BVESphere::init_velocity_vertices", vertex_policy,
      BVEVertexVelocity(velocity_verts.view, this->vertices.phys_crds.view,
                        this->faces.phys_crds.view, rel_vort_faces.view,
                        this->faces.area, this->faces.mask,
                        this->n_faces_host()));

  Kokkos::parallel_for(
      "BVESphere::init_velocity_faces", face_policy,
      BVEFaceVelocity(velocity_faces.view, this->faces.phys_crds.view,
                      rel_vort_faces.view, this->faces.area, this->faces.mask,
                      this->n_faces_host()));

  this->velocity_verts.update_host();
  this->velocity_faces.update_host();

  Kokkos::Profiling::popRegion();
}

template <typename SeedType>
void BVESphere<SeedType>::init_stream_fn() {
  ko::TeamPolicy<> vertex_policy(this->n_vertices_host(), ko::AUTO());
  ko::TeamPolicy<> face_policy(this->n_faces_host(), ko::AUTO());

  Kokkos::Profiling::pushRegion("BVESphere::init_stream_fn");

  Kokkos::parallel_for(
      "BVESphere::init_stream_fn_vertices", vertex_policy,
      BVEVertexStreamFn(stream_fn_verts.view, this->vertices.phys_crds.view,
                        this->faces.phys_crds.view, rel_vort_faces.view,
                        this->faces.area, this->faces.mask,
                        this->n_faces_host()));

  Kokkos::parallel_for(
      "BVESphere::init_stream_fn_faces", face_policy,
      BVEFaceStreamFn(stream_fn_faces.view, this->faces.phys_crds.view,
                      rel_vort_faces.view, this->faces.area, this->faces.mask,
                      this->n_faces_host()));

  this->stream_fn_verts.update_host();
  this->stream_fn_faces.update_host();

  Kokkos::Profiling::popRegion();
}

}  // namespace Lpm
