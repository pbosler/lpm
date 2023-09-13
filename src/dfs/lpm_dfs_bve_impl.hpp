#ifndef LPM_DFS_BVE_IMPL_HPP
#define LPM_DFS_BVE_IMPL_HPP

#include "lpm_dfs_bve.hpp"
#include "mesh/lpm_gather_mesh_data_impl.hpp"
#include "mesh/lpm_scatter_mesh_data_impl.hpp"
#include "util/lpm_string_util.hpp"
#include "lpm_velocity_gallery.hpp"

#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"
#endif

namespace Lpm {
namespace DFS {

template <typename SeedType>
DFSBVE<SeedType>::DFSBVE(const PolyMeshParameters<SeedType>& mesh_params,
                         const Int nlon,
                         const Int n_tracers,
                         const gmls::Params& interp_params,
                         const Real Omg) :
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
  mesh(mesh_params),
  grid(nlon),
  ntracers(n_tracers),
  Omega(Omg),
  t(0.0),
  gmls_params(interp_params)
{
  for (int k=0; k<n_tracers; ++k) {
    tracer_passive.push_back(ScalarField<VertexField>("tracer" + std::to_string(k), mesh_params.nmaxverts));
    tracer_active.push_back(ScalarField<FaceField>("tracer" + std::to_string(k), mesh_params.nmaxfaces));
  }

  grid_crds = grid.init_coords();
  grid_crds.update_host();
  grid_area = grid.weights_view();

  gathered_mesh = std::make_unique<GatherMeshData<SeedType>>(mesh);
  scatter_mesh = std::make_unique<ScatterMeshData<SeedType>>(*gathered_mesh, mesh);

  passive_scalar_fields.emplace("relative_vorticity", rel_vort_passive);
  active_scalar_fields.emplace("relative_vorticity", rel_vort_active);
  passive_vector_fields.emplace("velocity", velocity_passive);
  active_vector_fields.emplace("velocity", velocity_active);

  gathered_mesh->init_scalar_fields(passive_scalar_fields, active_scalar_fields);
  gathered_mesh->init_vector_fields(passive_vector_fields, active_vector_fields);

  mesh_to_grid_neighborhoods = gmls::Neighborhoods(gathered_mesh->h_phys_crds, grid_crds.get_host_crd_view(), gmls_params);
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
  auto gridcrds = grid_crds.view;
  Kokkos::parallel_for("initialize vorticity (grid)",
    grid.size(), KOKKOS_LAMBDA (const Index i) {
    const auto mxyz = Kokkos::subview(gridcrds, i, Kokkos::ALL);
    const Real zeta = vorticity_fn(mxyz[0], mxyz[1], mxyz[2]);
    zeta_grid(i) = zeta;
    omega_grid(i) = zeta + 2*Omg*mxyz[2];
  });

  gathered_mesh->gather_scalar_fields(passive_scalar_fields, active_scalar_fields);
};

template <typename SeedType> template <typename VelocityType>
void DFSBVE<SeedType>::init_velocity() {
  auto u_verts = velocity_passive.view;
  auto vert_crds = mesh.vertices.phys_crds.view;
  Kokkos::parallel_for("initialize velocity (passive)",
    mesh.n_vertices_host(),
    VelocityKernel<VelocityType>(u_verts, vert_crds, 0));
  auto u_faces = velocity_active.view;
  auto face_crds = mesh.faces.phys_crds.view;
  Kokkos::parallel_for("initialize velocity (active)",
    mesh.n_faces_host(),
    VelocityKernel<VelocityType>(u_faces, face_crds, 0));
  auto u_grid = velocity_grid.view;
  auto gridcrds = grid_crds.view;
  Kokkos::parallel_for("initialize velocity (grid)",
    grid.size(),
    VelocityKernel<VelocityType>(u_grid, gridcrds, 0));
}

#ifdef LPM_USE_VTK
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
#endif

template <typename SeedType>
std::string DFSBVE<SeedType>::info_string(const int tab_level) const {
  std::ostringstream ss;
  auto tabstr = indent_string(tab_level);
  ss << tabstr << "DSFBVE info:\n";
  tabstr += "\t";
  ss << mesh.info_string("DFSBVE",tab_level+1);
  ss << grid.info_string(tab_level+1);
  ss << gmls_params.info_string(tab_level+1);
  ss << mesh_to_grid_neighborhoods.info_string(tab_level+1);
  ss << grid_crds.info_string("DFSBVE grid_crds", tab_level+1);
  return ss.str();
}

template <typename SeedType>
void DFSBVE<SeedType>::interpolate_vorticity_from_mesh_to_grid() {
  return interpolate_vorticity_from_mesh_to_grid(rel_vort_grid);
}

template <typename SeedType>
void DFSBVE<SeedType>::interpolate_vorticity_from_mesh_to_grid(ScalarField<VertexField>& target) {
  const auto gmls_ops = std::vector<Compadre::TargetOperation>({Compadre::ScalarPointEvaluation});

  auto rel_vort_gmls = gmls::sphere_scalar_gmls(gathered_mesh->phys_crds,
    grid_crds.view, mesh_to_grid_neighborhoods, gmls_params, gmls_ops);

  Compadre::Evaluator rel_vort_eval(&rel_vort_gmls);
  target.view = rel_vort_eval.applyAlphasToDataAllComponentsAllTargetSites<Real*,DevMemory>(
    gathered_mesh->scalar_fields.at("relative_vorticity"),
    Compadre::ScalarPointEvaluation,
    Compadre::PointSample);
}

template <typename SeedType>
void DFSBVE<SeedType>::interpolate_velocity_from_grid_to_mesh() {
  // pb->mc:
  // 1. add your code to the src/dfs/ folder
  //            and update lpm/src/CMakeLists.txt to build your code.
  // 2. Then make sure this function works by adding it to lpm/tests/lpm_dfs_bve_test.cpp
  // 3. Compute the error on the particles: Add a scalar field to the test
  //      for each set of particles (active and passive).   Look at src/lpm_error.hpp;
  //      Just give it the exact velocity and the interpolated velocity, and it will do the rest.
  // 4. Output the velocity to vtk with lpm/tests/dfs_bve_test.cpp
  const auto rel_vort_dfs = rel_vort_grid.view;
  auto velocity_out = gathered_mesh->vector_fields.at("velocity");
  dfs_vort_2_velocity(gathered_mesh->phys_crds, rel_vort_dfs, velocity_out);
  scatter_mesh->scatter_fields(passive_scalar_fields, active_scalar_fields,
    passive_vector_fields, active_vector_fields);
}

template <typename SeedType>
void DFSBVE<SeedType>::init_velocity_from_vorticity() {
  const auto rel_vort_dfs = rel_vort_grid.view;
  auto velocity_out = gathered_mesh->vector_fields.at("velocity");
  dfs_vort_2_velocity(gathered_mesh->phys_crds, rel_vort_dfs, velocity_out);
  scatter_mesh->scatter_fields(passive_scalar_fields, active_scalar_fields,
    passive_vector_fields, active_vector_fields);
}

template <typename SeedType> template <typename SolverType>
void DFSBVE<SeedType>::advance_timestep(SolverType& solver) {
  solver.advance_timestep();
  scatter_mesh->scatter_fields(passive_scalar_fields, active_scalar_fields,
    passive_vector_fields, active_vector_fields);
  scatter_mesh->scatter_phys_crds();
  gathered_mesh->update_host();
#ifndef NDEBUG
  constexpr bool verbose_output = true;
#else
  constexpr bool verbose_output = false;
#endif
  mesh_to_grid_neighborhoods = gmls::Neighborhoods(gathered_mesh->h_phys_crds,
    grid_crds.get_host_crd_view(), gmls_params);
  t = solver.t_idx * solver.dt;
}

#ifdef LPM_USE_VTK
template <typename SeedType>
  VtkPolymeshInterface<SeedType> vtk_mesh_interface(const DFSBVE<SeedType>& dfs_bve) {
    VtkPolymeshInterface<SeedType> vtk(dfs_bve.mesh);
    vtk.add_scalar_point_data(dfs_bve.rel_vort_passive.view, "relative_vorticity");
    vtk.add_scalar_point_data(dfs_bve.abs_vort_passive.view, "absolute_vorticity");
    vtk.add_scalar_point_data(dfs_bve.stream_fn_passive.view, "stream_function");
    vtk.add_vector_point_data(dfs_bve.velocity_passive.view, "velocity");
    vtk.add_scalar_cell_data(dfs_bve.rel_vort_active.view, "relative_vorticity");
    vtk.add_scalar_cell_data(dfs_bve.abs_vort_active.view, "absolute_vorticity");
    vtk.add_scalar_cell_data(dfs_bve.stream_fn_active.view, "stream_function");
    vtk.add_vector_cell_data(dfs_bve.velocity_active.view, "velocity");
    for (short i=0; i<dfs_bve.tracer_passive.size(); ++i) {
      vtk.add_scalar_point_data(dfs_bve.tracer_passive[i].view,
        dfs_bve.tracer_passive[i].view.label());
      vtk.add_scalar_cell_data(dfs_bve.tracer_active[i].view,
        dfs_bve.tracer_active[i].view.label());
    }
    return vtk;
  }



  template <typename SeedType>
  VtkGridInterface vtk_grid_interface(const DFSBVE<SeedType>& dfs_bve) {
    VtkGridInterface vtk(dfs_bve.grid);
    vtk.add_scalar_point_data(dfs_bve.rel_vort_grid.view, "relative_vorticity");
    vtk.add_scalar_point_data(dfs_bve.abs_vort_grid.view, "absolute_vorticity");
    vtk.add_scalar_point_data(dfs_bve.stream_fn_grid.view, "stream_function");
    vtk.add_vector_point_data(dfs_bve.velocity_grid.view, "velocity");
    return vtk;
  }
#endif

} // namespace DFS
} // namespace Lpm
#endif
