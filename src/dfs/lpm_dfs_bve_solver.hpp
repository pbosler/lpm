#ifndef LPM_DFSBVE_SOLVER_HPP
#define LPM_DFSBVE_SOLVER_HPP

#include "LpmConfig.h"

namespace Lpm {
namespace DFS {

// Second order Runge-Kutta
template <typename SeedType>
class DFSRK2 {
  public:
    typedef SphereGeometry::crd_view_type crd_view;
  typedef SphereGeometry::vec_view_type vec_view;

  crd_view xyz_particles;
  scalar_view_type rel_vort_particles;
  vec_view velocity_particles;

  crd_view xyz_grid;
  scalar_view_type rel_vort_grid;

  Real dt;
  Real Omega;
  DFSBVE<SeedType>& sphere;
  Int t_idx;

  DFSRK2(const Real timestep, DFSBVE<SeedType>& sph, const Int tidx = 0) :
    dt(timestep),
    Omega(sph.Omega()),
    xyz_particles(sph.gathered_mesh->phys_crds),
    rel_vort_particles(sph.gathered_mesh->scalar_fields.at("relative_vorticity")),
    velocity_particles(sph.gathered_mesh->vector_fields.at("velocity")),
    xyz_grid(sph.grid_crds.view),
    rel_vort_grid(sph.rel_vort_grid.view),
    xyz_particles1("xyz_particles_stage1", sph.gathered_mesh->n()),
    xyz_particles2("xyz_particles_stage2", sph.gathered_mesh->n()),
    xyz_particles_work("xyz_particles_work", sph.gathered_mesh->n()),
    rel_vort_particles1("rel_vort_particles_stage1", sph.gathered_mesh->n()),
    rel_vort_particles2("rel_vort_particels_stage2", sph.gathered_mesh->n()),
    rel_vort_particles_work("rel_vort_particles_work", sph.gathered_mesh->n()),
    vel_particles("velocity_particles_stage2", sph.gathered_mesh->n()),
    sphere(sph),
    t_idx(tidx)
    {}

  void advance_timestep() ;
  protected:


    crd_view xyz_particles1;
    crd_view xyz_particles2;
    crd_view xyz_particles_work;

    scalar_view_type rel_vort_particles1;
    scalar_view_type rel_vort_particles2;
    scalar_view_type rel_vort_particles_work;

    vec_view vel_particles;

    void interpolate_vorticity_from_mesh_to_grid(scalar_view_type& rel_vort_grid, const crd_view& xyz_mesh, const crd_view& xyz_grid, const scalar_view_type& rel_vort_mesh)const;

};

// Fourth order Runge-Kutta method
template <typename SeedType>
class DFSRK4 {
  public:
    typedef SphereGeometry::crd_view_type crd_view;
  typedef SphereGeometry::vec_view_type vec_view;

  crd_view xyz_particles;
  scalar_view_type rel_vort_particles;
  vec_view velocity_particles;

  crd_view xyz_grid;
  scalar_view_type rel_vort_grid;

  Real dt;
  Real Omega;
  DFSBVE<SeedType>& sphere;
  Int t_idx;

  DFSRK4(const Real timestep, DFSBVE<SeedType>& sph, const Int t_idx = 0) :
    dt(timestep),
    Omega(sph.Omega()),
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
    {}

  void advance_timestep();

  protected:


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
    void standardize_particles(crd_view xyz_particles);
};


// Third order Runge-Kutta (RK3)
template <typename SeedType>
class DFSRK3 {
  public:
    typedef SphereGeometry::crd_view_type crd_view;
  typedef SphereGeometry::vec_view_type vec_view;

  crd_view xyz_particles;
  scalar_view_type rel_vort_particles;
  vec_view velocity_particles;

  crd_view xyz_grid;
  scalar_view_type rel_vort_grid;

  Real dt;
  Real Omega;
  DFSBVE<SeedType>& sphere;
  Int t_idx;

  DFSRK3(const Real timestep, DFSBVE<SeedType>& sph, const Int t_idx = 0) :
    dt(timestep),
    Omega(sph.Omega()),
    t_idx(t_idx),
    xyz_particles(sph.gathered_mesh->phys_crds),
    rel_vort_particles(sph.gathered_mesh->scalar_fields.at("relative_vorticity")),
    velocity_particles(sph.gathered_mesh->vector_fields.at("velocity")),
    xyz_grid(sph.grid_crds.view),
    rel_vort_grid(sph.rel_vort_grid.view),
    xyz_particles1("xyz_particles_stage1", sph.gathered_mesh->n()),
    xyz_particles2("xyz_particles_stage2", sph.gathered_mesh->n()),
    xyz_particles3("xyz_particles_stage3", sph.gathered_mesh->n()),
    xyz_particles_work("xyz_particles_work", sph.gathered_mesh->n()),
    rel_vort_particles1("rel_vort_particles_stage1", sph.gathered_mesh->n()),
    rel_vort_particles2("rel_vort_particels_stage2", sph.gathered_mesh->n()),
    rel_vort_particles3("rel_vort_particles_stage3", sph.gathered_mesh->n()),
    rel_vort_particles_work("rel_vort_particles_work", sph.gathered_mesh->n()),
    vel_particles("velocity_particles_stage2", sph.gathered_mesh->n()),
    sphere(sph)
    {}

  void advance_timestep();

  protected:


    crd_view xyz_particles1;
    crd_view xyz_particles2;
    crd_view xyz_particles3;
    crd_view xyz_particles_work;

    scalar_view_type rel_vort_particles1;
    scalar_view_type rel_vort_particles2;
    scalar_view_type rel_vort_particles3;
    scalar_view_type rel_vort_particles_work;

    vec_view vel_particles;

    void interpolate_vorticity_from_mesh_to_grid(scalar_view_type& rel_vort_grid, const crd_view& xyz_mesh, const crd_view& xyz_grid, const scalar_view_type& rel_vort_mesh)const;
    void std_crds(const crd_view& xyz_grid);

};



} // namespace DFS
} // namespace Lpm

#endif
