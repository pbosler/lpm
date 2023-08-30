#include "dfs/lpm_dfs_bve.hpp"
#include "dfs/lpm_dfs_bve_impl.hpp"
#include "dfs/lpm_dfs_bve_solver.hpp"
#include "dfs/lpm_dfs_bve_solver_impl.hpp"
#include "dfs/lpm_dfs_grid.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
#include "lpm_compadre.hpp"
#include "lpm_vorticity_gallery.hpp"
#include "lpm_error.hpp"
#include "lpm_error_impl.hpp"
#include <catch2/catch_test_macros.hpp>

using namespace Lpm;

TEST_CASE("dfs_bve_unit_tests", "[dfs]") {
  Comm comm;
  Logger<> logger("dfs_bve_tests", Log::level::debug, comm);

  const Int nlon = 10;
  const Int ntracers = 0;

  typedef CubedSphereSeed seed_type;
  typedef RossbyWave54Velocity velocity_type;
  RossbyHaurwitz54 vorticity_fn;

  const Int mesh_depth = 3;
  PolyMeshParameters<seed_type> mesh_params(mesh_depth);
  const Int gmls_order = 4;
  gmls::Params gmls_params(gmls_order);

  DFS::DFSBVE<seed_type> dfs_bve(mesh_params, nlon, ntracers, gmls_params);
  dfs_bve.init_vorticity(vorticity_fn);
  dfs_bve.template init_velocity<velocity_type>();

  // mesh to grid
  ScalarField<VertexField> rel_vort_grid_gmls("relative_vorticity_gmls",
    dfs_bve.vtk_grid_size());
  dfs_bve.interpolate_vorticity_from_mesh_to_grid(rel_vort_grid_gmls);

  logger.info(dfs_bve.info_string());

  scalar_view_type rel_vort_error("relative_vorticity_gmls_error", dfs_bve.grid_size());
  ErrNorms gmls_err(rel_vort_error, rel_vort_grid_gmls.view, dfs_bve.rel_vort_grid.view, dfs_bve.grid_area);
  logger.info(gmls_err.info_string());

  // grid to mesh
	dfs_bve.interpolate_velocity_from_grid_to_mesh();

  // finish-up: write output
  const std::string mesh_vtk_file = "dfs_bve_particles.vtp";
  const std::string grid_vtk_file = "dfs_bve_grid.vts";

  auto vtk_mesh = vtk_mesh_interface(dfs_bve);
  auto vtk_grid = vtk_grid_interface(dfs_bve);
  vtk_grid.add_scalar_point_data(rel_vort_grid_gmls.view, "relative_vorticity_gmls");
  vtk_grid.add_scalar_point_data(rel_vort_error, "relative_vorticity_gmls_error");

  vtk_mesh.write(mesh_vtk_file);
  vtk_grid.write(grid_vtk_file);

}

struct RelVortError {
  scalar_view_type rel_vort_error;
  scalar_view_type abs_vort;
  scalar_view_type rel_vort;
  typename SphereGeometry::crd_view_type phys_crds_view;
  Real Omega;

  RelVortError(scalar_view_type err, const scalar_view_type omega,
    const scalar_view_type zeta, const typename SphereGeometry::crd_view_type pcrds,
    const Real Omg) :
    rel_vort_error(err),
    abs_vort(omega),
    rel_vort(zeta),
    phys_crds_view(pcrds),
    Omega(Omg) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index i) const {
    rel_vort_error(i) = abs_vort(i) - 2*Omega*phys_crds_view(i,2) - rel_vort(i);
  }
};

template <typename SeedType>
void compute_vorticity_error(scalar_view_type vert_err, scalar_view_type face_err,
  const DFS::DFSBVE<SeedType>& sph) {
  Kokkos::parallel_for("relative vorticity error (vertices)",
    sph.mesh.n_vertices_host(),
    RelVortError(vert_err,
      sph.abs_vort_passive.view, sph.rel_vort_passive.view,
      sph.mesh.vertices.phys_crds.view, sph.Omega));
  Kokkos::parallel_for("relative vorticity error (faces)",
    sph.mesh.n_faces_host(),
    RelVortError(face_err,
      sph.abs_vort_active.view, sph.rel_vort_active.view,
      sph.mesh.faces.phys_crds.view, sph.Omega));
}

TEST_CASE("dfs_bve_timestep_test", "[dfs]") {
  Comm comm;
  Logger<> logger("dfs_bve_timestep", Log::level::debug, comm);
  const Int nlon = 40;
  const Int ntracers = 0;

  typedef CubedSphereSeed seed_type;
  typedef RossbyWave54Velocity velocity_type;
  RossbyHaurwitz54 vorticity_fn;

  const Int mesh_depth = 5;
  PolyMeshParameters<seed_type> mesh_params(mesh_depth);
  const Int gmls_order = 4;
  gmls::Params gmls_params(gmls_order);

  const Real dt = 0.01;
  const Real tfinal = 2;
  const int nsteps = int(tfinal/dt);
  DFS::DFSBVE<seed_type> sphere(mesh_params, nlon, ntracers, gmls_params);
  sphere.init_vorticity(vorticity_fn);
  sphere.template init_velocity<velocity_type>();

  ScalarField<VertexField> vert_rel_vort_error("relative_vorticity_error", sphere.mesh.n_vertices_host());
  ScalarField<FaceField> face_rel_vort_error("relative_vorticity_error", sphere.mesh.n_faces_host());
  compute_vorticity_error(vert_rel_vort_error.view, face_rel_vort_error.view, sphere);

  int output_ctr = 0;
  const std::string fname_root = "dfs_bve" + seed_type::id_string() + "_nlon" +
    std::to_string(nlon) + "_dt" + std::to_string(dt);
  {
    const std::string mesh_vtk_file = fname_root + "_" + zero_fill_str(output_ctr) + ".vtp";
    const std::string grid_vtk_file = fname_root + "_" + zero_fill_str(output_ctr) + ".vts";
    auto vtk_mesh = vtk_mesh_interface(sphere);
    auto vtk_grid = vtk_grid_interface(sphere);
    vtk_mesh.add_scalar_point_data(vert_rel_vort_error.view);
    vtk_mesh.add_scalar_cell_data(face_rel_vort_error.view);
    vtk_mesh.write(mesh_vtk_file);
    vtk_grid.write(grid_vtk_file);
    ++output_ctr;
  }

  DFS::DFSRK2<seed_type> rk2_solver(dt, sphere);

  for (int time_idx=0; time_idx<nsteps; ++time_idx) {
    sphere.advance_timestep(rk2_solver);
    compute_vorticity_error(vert_rel_vort_error.view, face_rel_vort_error.view, sphere);
    {
      const std::string mesh_vtk_file = fname_root + "_" + zero_fill_str(output_ctr) + ".vtp";
      const std::string grid_vtk_file = fname_root + "_" + zero_fill_str(output_ctr) + ".vts";
      auto vtk_mesh = vtk_mesh_interface(sphere);
      auto vtk_grid = vtk_grid_interface(sphere);
      vtk_mesh.add_scalar_point_data(vert_rel_vort_error.view);
      vtk_mesh.add_scalar_cell_data(face_rel_vort_error.view);
      vtk_mesh.write(mesh_vtk_file);
      vtk_grid.write(grid_vtk_file);
      ++output_ctr;
    }
    logger.debug("t = {}", time_idx*dt);
  }
}
