#ifndef LPM_DFS_BVE_HPP
#define LPM_DFS_BVE_HPP

#include "LpmConfig.h"
#include "lpm_field.hpp"
#include "LpmConfig.h"
#include "lpm_field.hpp"
#include "lpm_geometry.hpp"
#include "lpm_compadre.hpp"
#include "dfs/lpm_dfs_grid.hpp"
#include "mesh/lpm_mesh_seed.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#include "mesh/lpm_gather_mesh_data.hpp"
#include "mesh/lpm_gather_mesh_data_impl.hpp"
#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"
#endif

namespace Lpm {
namespace DFS {

/**  Particle/mesh solver for the barotropic vorticity equation (BVE).

  Advection and vorticity are computed on Lagrangian particles.
  Velocity is solved on a uniform colatitude-longitude grid with
  double Fourier series (DFS) methods.
*/
template <typename SeedType>
class DFSBVE {
  static_assert(std::is_same<typename SeedType::geo, SphereGeometry>::value,
                "Spherical mesh seed required.");

  /// Relative vorticity at passive particles
  ScalarField<VertexField> rel_vort_passive;
  /// Relative vorticity at active particles
  ScalarField<FaceField> rel_vort_active;
  /// Relative vorticity on the grid
  ScalarField<VertexField> rel_vort_grid;
  /// Absolute vorticity at passive particles
  ScalarField<VertexField> abs_vort_passive;
  /// Absolute vorticity at active particles
  ScalarField<FaceField> abs_vort_active;
  /// Absolute vorticity on the grid
  ScalarField<VertexField> abs_vort_grid;
  /// Stream function at passive particles
  ScalarField<VertexField> stream_fn_passive;
  /// Stream function at active particles
  ScalarField<FaceField> stream_fn_active;
  /// Stream function on the grid
  ScalarField<VertexField> stream_fn_grid;
  /// Velocity at passive particles
  VectorField<SphereGeometry,VertexField> velocity_passive;
  /// Velocity at active particles
  VectorField<SphereGeometry,FaceField> velocity_active;
  /// Velocity on the grid
  VectorField<SphereGeometry,VertexField> velocity_grid;
  /// Lagrangian particle/panel mesh
  PolyMesh2d<SeedType> mesh;
  /// Uniform colatitude-longitude grid for double Fourier series
  DFSGrid grid;
  /// number of passive tracers
  Int ntracers;
  /// background rotation rate of sphere about positive z-axis
  Real Omega;
  /// time
  Real t;
  /// passive tracers at passive particles
  std::vector<ScalarField<VertexField>> tracer_passive;
  /// passive tracers at active particles
  std::vector<ScalarField<FaceField>> tracer_active;
  /// DFS grid coordinates
  typename SphereGeometry::crd_view_type grid_crds;
  typename SphereGeometry::crd_view_type::HostMirror h_grid_crds;
  /// GMLS interpolation parameters
  gmls::Params gmls_params;
  /// GMLS neighborhoods
  gmls::Neighborhoods neighborhoods;


  public:
    /** Constructor.

      @param [in] mesh_params initialization parameters for the Lagrangian particle/panel mesh
      @param [in] nlon number of longitude points in the DFS grid
      @param [in] n_tracers number of scalar tracer fields carried by the particles
    */
    DFSBVE(const PolyMeshParameters<SeedType>& mesh_params,
           const Int nlon,
           const Int n_tracers,
           const gmls::Params& interp_params);

    template <typename VorticityInitialCondition>
    void init_vorticity(const VorticityInitialCondition& vorticity_fn);

#ifdef LPM_USE_VTK
  void write_vtk(const std::string mesh_fname, const std::string grid_fname) const;
#endif

};

} // namespace DFS
} // namespace Lpm

#endif
