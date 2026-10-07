#ifndef LPM_DFS_POLAR_VORTEX_SOLVER_HPP
#define LPM_DFS_POLAR_VORTEX_SOLVER_HPP

#include "LpmConfig.h"
#include "lpm_input.hpp"
#include "dfs/lpm_dfs_bve.hpp"
#include "lpm_vorticity_gallery.hpp"

namespace Lpm {
namespace DFS {

struct TimestepParams {
  Real dt;
  Real tfinal;
  Int nsteps;

  explicit TimestepParams(const user::Input& input);

  Real courant_number(const Real& dx, const Real& max_vel) const;

  std::string filename_piece() const;
};

template <typename SeedType>
class DFSPolarVortexRK4 {
  static_assert(std::is_same<typename SeedType::geo, SphereGeometry>::value,
    "sphere geometry required.");
  public:
    typedef SphereGeometry::crd_view_type crd_view;
    typedef SphereGeometry::vec_view_type vec_view;

    /** passive particles and active particles that are leaves in the face tree

      (no active particles from divided panels here)

      use GatherMeshData.
    */
    crd_view xyz_particles;
    scalar_view_type rel_vort_particles;
    vec_view velocity_particles;

    crd_view xyz_grid;
    scalar_view_type rel_vort_grid;

    Real dt;
    Real Omega;
    Real t;
    Int t_idx;

    JM86Forcing forcing;

    DFSBVE<SeedType>& sphere;

    DFSPolarVortexRK4(const Real timestep, DFSBVE<SeedType>& sph, const Int t_idx, const PolarVortexParams& pv_params) :
      dt(timestep),
      Omega(sph.Omega()),
      t(t_idx * timestep),
      t_idx(t_idx),
      xyz_particles(sph.gathered_mesh->phys_crds),
      rel_vort_particles(sph.gathered_mesh->scalar_fields.at("relative_vorticity")),
      velocity_particles(sph.gathered_mesh->vector_fields.at("velocity")),
      xyz_grid(sph.grid_crds.view),
      rel_vort_grid(sph.rel_vort_grid.view),
      xyz_particles1("xyz_particles_stage1", sph.gathered_mesh->n()),
      xyz_particles2("xyz_particles_stage2", sph.gathered_mesh->n()),
      xyz_particles3("xyz_particles_stage3", sph.gathered_mesh->n()),
      xyz_particles4("xyz_particles_stage4", sph.gathered_mesh->n()),
      xyz_particles_work("xyz_particles_work", sph.gathered_mesh->n()),
      rel_vort_particles1("rel_vort_particles_stage1", sph.gathered_mesh->n()),
      rel_vort_particles2("rel_vort_particels_stage2", sph.gathered_mesh->n()),
      rel_vort_particles3("rel_vort_particles_stage3", sph.gathered_mesh->n()),
      rel_vort_particles4("rel_vort_particels_stage4", sph.gathered_mesh->n()),
      rel_vort_particles_work("rel_vort_particles_work", sph.gathered_mesh->n()),
      vel_particles("velocity_particles_stage2", sph.gathered_mesh->n()),
      sphere(sph)
//       forcing(pv_params)
      {}

    void advance_timestep();

    /** Invariant potential vorticity formulation (as in the Fortran thesis code).

      q = zeta + 2*Omega*z + F(x,t) is materially conserved. When enabled, zeta at
      every RK stage and after every step is set to q - 2*Omega*z - F(x,t) instead of
      being time-integrated with dzeta/dt = -(2*Omega*w + DF/Dt).
    */
    bool use_invariant_vorticity = false;

    /// Store q from the current particle state; call once before the first step.
    void record_invariant();

    /// max_i |zeta_i + 2*Omega*z_i + F(x_i,t) - q_i| (needs record_invariant()).
    Real max_invariant_drift() const;

    /// max_i |velocity_i|
    Real max_velocity() const;

  protected:
    scalar_view_type pv_invariant;

    crd_view xyz_particles1;
    crd_view xyz_particles2;
    crd_view xyz_particles3;
    crd_view xyz_particles4;
    crd_view xyz_particles_work;

    scalar_view_type rel_vort_particles1;
    scalar_view_type rel_vort_particles2;
    scalar_view_type rel_vort_particles3;
    scalar_view_type rel_vort_particles4;
    scalar_view_type rel_vort_particles_work;

    vec_view vel_particles;

    void interpolate_vorticity_from_mesh_to_grid(scalar_view_type& rel_vort_grid, const crd_view& xyz_mesh, const crd_view& xyz_grid, const scalar_view_type& rel_vort_mesh)const;

    void normalize_coordinates(crd_view xyz);

}; /// DFSPolarVortexRK4

} // namespace DFS
} // namespace Lpm
#endif
