#ifndef LPM_SWE_HPP
#define LPM_SWE_HPP

#include "LpmConfig.h"
#include "lpm_coriolis.hpp"
#include "lpm_field.hpp"
#include "lpm_coords.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#endif

namespace Lpm {

template <typename InitialCondition> struct SWEInitializeProblem;

template <typename SeedType>
class SWE {
  public: // members
    typedef typename SeedType::geo geo;
    using coords_type = Coords<geo>;
    using crd_view = typename coords_type::view_type;
    using vec_view = typename geo::vec_view_type;

    /// relative vorticity
    ScalarField<VertexField> rel_vort_passive;
    ScalarField<FaceField> rel_vort_active;
    /// potential vorticity
    ScalarField<VertexField> pot_vort_passive;
    ScalarField<FaceField> pot_vort_active;
    /// divergence
    ScalarField<VertexField> div_passive;
    ScalarField<FaceField> div_active;
    /// surface height
    ScalarField<VertexField> surf_passive;
    ScalarField<FaceField> surf_active;
    // bottom topography
    ScalarField<VertexField> bottom_passive;
    ScalarField<FaceField> bottom_active;
    /// surface laplacian
    ScalarField<VertexField> surf_lap_passive;
    ScalarField<FaceField> surf_lap_active;
    /// fluid depth
    ScalarField<VertexField> depth_passive;
    ScalarField<FaceField> depth_active;
    /// double dot product
    ScalarField<VertexField> double_dot_passive;
    ScalarField<FaceField> double_dot_active;
    /// velocity
    VectorField<geo,VertexField> velocity_passive;
    VectorField<geo, FaceField> velocity_active;
    /// mass
    ScalarField<FaceField> mass_active;
    /// Lagrangian particle/panel mesh
    PolyMesh2d<SeedType> mesh;

    /// for planar problems, Coriolis parameter f = f0 + beta*y
    Real f0;
    Real beta;
    /// for spherical problems, Coriolis parameter f = 2*Omega*z
    Real Omega;
    /// gravity
    Real g;
    /// time
    Real t;

  public: // functions
    void update_host();
    void update_device();

    /**  @brief Sets bottom topography values on all particles.

      Should be called at each time increment to update bottom
      topography values to new particle locations.
    */
    template <typename TopoType>
    void set_bottom_topography(const TopoType& topo);

    /** @brief Sets initial (t=0) velocity on all particles.

        Velocity function should be consistent with the initial
        vorticity and divergence profiles.
    */
    template <typename VelocityType>
    void init_velocity_from_function();

    /** @brief Sets initial vorticity on all particles.

      If depth_set = true, also sets potential voricity.
    */
    template <typename VorticityType>
    void init_vorticity(const VorticityType& vort_fn, const bool depth_set=false);


    /** @brief Sets initial divergence values for all particles.
    */
    template <typename DivergenceType>
    void init_divergence(const DivergenceType& div_fn);

    /* @brief Sets all initial data on all particles (WIP).

      TODO: InitialConditions type is not a fixed definition yet.
    */
    template <typename InitialConditions>
    void init_swe_problem(const InitialConditions& ic);

    /* @brief Sets initial surface height and depth at all particles.
    */
    template <typename BottomType, typename SurfaceType>
    void init_surface(const BottomType& topo, const SurfaceType& surf);

    /* @brief Sets initial velocity from initial vorticity and divergence
      values (when a function is not available.)

      @warning Initial vorticity and divergence must be set prior to
      calling this function.
    */
    void init_velocity();

    Real total_mass() const;
    Real total_energy() const;
    Real kinetic_energy() const;
    Real potential_energy() const;
    Real total_enstrophy() const;

    std::string info_string(const int tab_level, const bool verbose=false) const;

    template <typename SolverType>
    void advance_timestep(SolverType& solver);

    /// constructor for spherical problems
    SWE(const PolyMeshParameters<SeedType>& mesh_params, const Real Omg);

    /// constructor for planar problems
    SWE(const PolyMeshParameters<SeedType>& mesh_params, const Real f, const Real b);

};



#ifdef LPM_USE_VTK
  template <typename SeedType>
  VtkPolymeshInterface<SeedType> vtk_mesh_interface(const SWE<SeedType>& swe);
#endif

} // namespace Lpm

#endif
