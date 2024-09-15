#ifndef LPM_DFS_BVE_HPP
#define LPM_DFS_BVE_HPP

#include "LpmConfig.h"
#include "lpm_field.hpp"
#include "lpm_geometry.hpp"
#include "lpm_compadre.hpp"
#include "lpm_constants.hpp"
#include "lpm_coriolis.hpp"
#include "dfs/lpm_dfs_grid.hpp"
#include "mesh/lpm_compadre_remesh.hpp"
#include "mesh/lpm_mesh_seed.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#include "mesh/lpm_gather_mesh_data.hpp"
#include "mesh/lpm_scatter_mesh_data.hpp"
#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#include "dfs_vort2velocity.hpp"
#endif

using namespace SpherePoisson;
namespace Lpm {
namespace DFS {

template <typename SeedType> class DFSRK2; // fwd decl

/**  Particle/mesh solver for the barotropic vorticity equation (BVE).

  Advection and vorticity are computed on Lagrangian particles.
  Velocity is solved on a uniform colatitude-longitude grid with
  double Fourier series (DFS) methods.
*/
template <typename SeedType>
class DFSBVE {
  static_assert(std::is_same<typename SeedType::geo, SphereGeometry>::value,
                "Spherical mesh seed required.");
  using geo = SphereGeometry;
  using Coriolis = CoriolisSphere;
  using crd_view = typename SphereGeometry::crd_view_type;
  using vec_view = typename SphereGeometry::vec_view_type;

  friend class DFSRK2<SeedType>;

  public:
    /// Coriolis
    Coriolis coriolis;
    /// reference coordinates, for FTLE computations
    Coords<geo> ref_crds_passive;
    Coords<geo> ref_crds_active;
    // FTLE
    ScalarField<FaceField> ftle;
    ScalarField<VertexField> ftle_grid;
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
    /// time
    Real t;
    /// reference time for ftle
    Real t_ref;
    /// passive tracers at passive particles
    std::vector<ScalarField<VertexField>> tracer_passive;
    /// passive tracers at active particles
    std::vector<ScalarField<FaceField>> tracer_active;
    /// area weights for grid points
    scalar_view_type grid_area;
    /// DFS grid coordinates
    Coords<SphereGeometry> grid_crds;

  private:
    /** gathered mesh data collects only the particles that are at
    /// the lowest level of the quadtree --- it excludes all panels that have
    /// been divided and only keeps their lowest-level children.
    */
    std::unique_ptr<GatherMeshData<SeedType>> gathered_mesh;
    /** scattered mesh sends received data from the gathered mesh back to the full
      set of panels, divided and undivided.
    */
    std::unique_ptr<ScatterMeshData<SeedType>> scatter_mesh;
    /// Lists of field names to interpolate from mesh to grid or grid to mesh
    std::map<std::string, ScalarField<VertexField>> passive_scalar_fields;
    std::map<std::string, ScalarField<FaceField>> active_scalar_fields;
    std::map<std::string, VectorField<SphereGeometry,VertexField>> passive_vector_fields;
    std::map<std::string, VectorField<SphereGeometry,FaceField>> active_vector_fields;
    /// GMLS interpolation parameters
    gmls::Params gmls_params;
    /// GMLS neighborhoods
    gmls::Neighborhoods mesh_to_grid_neighborhoods;

    void update_mesh_to_grid_neighborhoods();



  public:
    /** Constructor.

      @param [in] mesh_params initialization parameters for the Lagrangian particle/panel mesh
      @param [in] nlon number of longitude points in the DFS grid
      @param [in] n_tracers number of scalar tracer fields carried by the particles
    */
    DFSBVE(const PolyMeshParameters<SeedType>& mesh_params,
           const Int nlon,
           const Int n_tracers,
           const gmls::Params& interp_params,
           const Real Omg = 2*constants::PI);

    template <typename VorticityInitialCondition>
    void init_vorticity(const VorticityInitialCondition& vorticity_fn);

    template <typename VelocityType>
    void init_velocity(const VelocityType& vel_fn);

    void init_velocity_from_vorticity();

    Index grid_size() const {return grid.size();}

    void interpolate_vorticity_from_mesh_to_grid();
    void interpolate_vorticity_from_mesh_to_grid(ScalarField<VertexField>& target);
    void interpolate_velocity_from_grid_to_mesh();

    void update_grid_absolute_vorticity();

    Real total_vorticity() const;
    Real total_kinetic_energy() const;
    Real total_enstrophy() const;
    Real ftle_max() const;

    std::string info_string(const int tab_level=0) const;

    template <typename SolverType>
    void advance_timestep(SolverType& solver);

#ifdef LPM_USE_VTK
  void write_vtk(const std::string mesh_fname, const std::string grid_fname) const;

  inline Index vtk_grid_size() {return grid.vtk_size(); }
#endif
};

#ifdef LPM_USE_VTK
  /** Return a vtk interface for the DFSBVE's Lagrangian particle/panel mesh
  */
  template <typename SeedType>
  VtkPolymeshInterface<SeedType> vtk_mesh_interface(const DFSBVE<SeedType>& dfs_bve);

  /** Return a VTK interface for the DFSBVE uniform grid.
  */
  template <typename SeedType>
  VtkGridInterface vtk_grid_interface(const DFSBVE<SeedType>& dfs_bve);
#endif

template <typename SeedType>
CompadreRemesh<SeedType> compadre_remesh(DFSBVE<SeedType>& new_dfs_bve, const DFSBVE<SeedType>& old_dfs_bve, const gmls::Params& gmls_params);

} // namespace DFS
} // namespace Lpm

#endif
