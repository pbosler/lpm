#ifndef LPM_DFS_BVE_IMPL_HPP
#define LPM_DFS_BVE_IMPL_HPP
#include "lpm_dfs_bve.hpp"
#include "lpm_constants.hpp"
#ifdef LPM_USE_VTK
#include <vtkStructuredGrid.h>
#include <vtkSmartPointer.h>
#include <vtkDoubleArray.h>
#include <vtkNew.h>
#include <vtkXMLStructuredGridWriter.h>
#include <vtkPointData.h>
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"
#endif

namespace Lpm {
namespace DFS {

template <typename SeedType>
DFSBVE<SeedType>::DFSBVE(const PolyMeshParameters<SeedType>& mesh_params,
                         const Int nlon,
                         const Int n_tracers,
                         const gmls::Params& interp_params) :
  rel_vort_passive("relative_vorticity", mesh_params.nmaxverts),
  rel_vort_active("relative_vorticity", mesh_params.nmaxfaces),
  rel_vort_grid("relative_vorticity", nlon*(nlon/2 + 1)),
  abs_vort_passive("absolute_vorticity", mesh_params.nmaxverts),
  abs_vort_active("absolute_vorticity", mesh_params.nmaxfaces),
  abs_vort_grid("absolute_vorticity", nlon*(nlon/2 + 1)),
  stream_fn_passive("stream_function", mesh_params.nmaxverts),
  stream_fn_active("stream_function", mesh_params.nmaxfaces),
  stream_fn_grid("stream_function", nlon*(nlon/2 + 1)),
  velocity_passive("velocity", mesh_params.nmaxverts),
  velocity_active("velocity", mesh_params.nmaxfaces),
  velocity_grid("velocity", nlon*(nlon/2 + 1)),
  grid_crds("dfs_grid_crds", nlon*(nlon/2 + 1)),
  mesh(mesh_params),
  grid(nlon),
  ntracers(n_tracers),
  Omega(2*constants::PI),
  t(0.0),
  gmls_params(interp_params)
{
  for (int k=0; k<n_tracers; ++k) {
    tracer_passive.push_back(ScalarField<VertexField>("tracer" + std::to_string(k), mesh_params.nmaxverts));
    tracer_active.push_back(ScalarField<FaceField>("tracer" + std::to_string(k), mesh_params.nmaxfaces));
  }
  grid.fill_packed_view(grid_crds);
  h_grid_crds = Kokkos::create_mirror_view(grid_crds);
  Kokkos::deep_copy(h_grid_crds, grid_crds);
}

template <typename SeedType> template <typename VorticityInitialCondition>
void DFSBVE<SeedType>::init_vorticity(const VorticityInitialCondition& vorticity_fn) {
  static_assert(std::is_same<typename VorticityInitialCondition::geo,
    SphereGeometry>::value, "Spherical vorticity function required.");

  auto zeta_verts = rel_vort_passive.view;
  auto omega_verts = abs_vort_passive.view;
  auto vert_crds = mesh.vertices.phys_crds.view;
  Real Omg = this->Omega;
  Kokkos::parallel_for("initialize vorticity (passive)",
    mesh.n_vertices_host(), KOKKOS_LAMBDA (const Index i) {
      const auto mxyz = Kokkos::subview(vert_crds, i, Kokkos::ALL);
      const Real zeta = vorticity_fn(mxyz[0], mxyz[1], mxyz[2]);
      zeta_verts(i) = zeta;
      omega_verts(i) = zeta + 2*Omg*mxyz[2];
    });
  auto zeta_faces = rel_vort_active.view;
  auto omega_faces = abs_vort_active.view;
  auto face_crds = mesh.faces.phys_crds.view;
  Kokkos::parallel_for("initialize vorticity (active)",
    mesh.n_faces_host(), KOKKOS_LAMBDA (const Index i) {
      const auto mxyz = Kokkos::subview(face_crds, i, Kokkos::ALL);
      const Real zeta = vorticity_fn(mxyz[0], mxyz[1], mxyz[2]);
      zeta_faces(i) = zeta;
      omega_faces(i) = zeta + 2*Omg*mxyz[2];
    });
  auto zeta_grid = rel_vort_grid.view;
  auto omega_grid = abs_vort_grid.view;
  auto gridcrds = grid_crds;
  Kokkos::parallel_for("initialize vorticity (grid)",
    grid.size(), KOKKOS_LAMBDA (const Index i) {
    const auto mxyz = Kokkos::subview(gridcrds, i, Kokkos::ALL);
    const Real zeta = vorticity_fn(mxyz[0], mxyz[1], mxyz[2]);
    zeta_grid(i) = zeta;
    omega_grid(i) = zeta + 2*Omg*mxyz[2];
  });
};

template <typename SeedType>
void DFSBVE<SeedType>::write_vtk(const std::string mesh_fname, const std::string grid_fname) const {
  auto vtk_mesh = VtkPolymeshInterface<SeedType>(mesh);
  vtk_mesh.add_scalar_point_data(rel_vort_passive.view, "relative_vorticity");
  vtk_mesh.add_scalar_point_data(abs_vort_passive.view, "absolute_vorticity");
  vtk_mesh.add_scalar_point_data(stream_fn_passive.view, "stream_function");
  vtk_mesh.add_vector_point_data(velocity_passive.view, "velocity");
  vtk_mesh.add_scalar_cell_data(rel_vort_active.view, "relative_vorticity");
  vtk_mesh.add_scalar_cell_data(abs_vort_active.view, "absolute_vorticity");
  vtk_mesh.add_scalar_cell_data(stream_fn_active.view, "stream_function");
  vtk_mesh.add_vector_cell_data(velocity_active.view, "velocity");
  for (Short i = 0; i < tracer_passive.size(); ++i) {
    vtk_mesh.add_scalar_point_data(tracer_passive[i].view,
                              tracer_passive[i].view.label());
    vtk_mesh.add_scalar_cell_data(tracer_active[i].view,
                             tracer_active[i].view.label());
  }
  vtk_mesh.write(mesh_fname);

  vtkSmartPointer<vtkStructuredGrid> vtk_grid = grid.vtk_grid();
  auto grid_relvort = vtkSmartPointer<vtkDoubleArray>::New();
  auto grid_absvort = vtkSmartPointer<vtkDoubleArray>::New();
  auto grid_stream = vtkSmartPointer<vtkDoubleArray>::New();
  auto grid_vel = vtkSmartPointer<vtkDoubleArray>::New();

  /// vtkStructuredGrid needs the longitude periodicity repeated, so we have to add
  /// an extra point for each row of the grid
  grid_relvort->SetName("relative_vorticity");
  grid_relvort->SetNumberOfComponents(1);
  grid_relvort->SetNumberOfTuples(grid.size()+grid.nlat);

  grid_absvort->SetName("absolute_vorticity");
  grid_absvort->SetNumberOfComponents(1);
  grid_absvort->SetNumberOfTuples(grid.size()+grid.nlat);

  grid_stream->SetName("stream_function");
  grid_stream->SetNumberOfComponents(1);
  grid_stream->SetNumberOfTuples(grid.size()+grid.nlat);

  grid_vel->SetName("velocity");
  grid_vel->SetNumberOfComponents(3);
  grid_vel->SetNumberOfTuples(grid.size()+grid.nlat);

  rel_vort_grid.update_host();
  abs_vort_grid.update_host();
  stream_fn_grid.update_host();
  velocity_grid.update_host();

  Index vtk_idx = 0;
  for (Index i=0; i<grid.nlat; ++i) {
    for (Index j=0; j<grid.nlon; ++j) {
      grid_relvort->InsertTuple1(vtk_idx, rel_vort_grid.hview(i*grid.nlon + j));
      grid_absvort->InsertTuple1(vtk_idx, abs_vort_grid.hview(i*grid.nlon + j));
      grid_stream->InsertTuple1(vtk_idx, stream_fn_grid.hview(i*grid.nlon + j));
      const auto vel = Kokkos::subview(velocity_grid.hview, i*grid.nlon + j, Kokkos::ALL);
      grid_vel->InsertTuple3(vtk_idx, vel[0], vel[1], vel[2]);
      ++vtk_idx;
    }
    grid_relvort->InsertTuple1(vtk_idx, rel_vort_grid.hview(i*grid.nlon));
    grid_absvort->InsertTuple1(vtk_idx, abs_vort_grid.hview(i*grid.nlon));
    grid_stream->InsertTuple1(vtk_idx, stream_fn_grid.hview(i*grid.nlon));
    const auto vel = Kokkos::subview(velocity_grid.hview, i*grid.nlon, Kokkos::ALL);
    grid_vel->InsertTuple3(vtk_idx, vel[0], vel[1], vel[2]);
    ++vtk_idx;
  }

  auto grid_data = vtk_grid->GetPointData();
  grid_data->AddArray(grid_relvort);
  grid_data->AddArray(grid_absvort);
  grid_data->AddArray(grid_stream);
  grid_data->AddArray(grid_vel);

  vtkNew<vtkXMLStructuredGridWriter> grid_writer;
  grid_writer->SetInputData(vtk_grid);
  grid_writer->SetFileName(grid_fname.c_str());
  grid_writer->Write();
}

} // namespace DFS
} // namespace Lpm
#endif
