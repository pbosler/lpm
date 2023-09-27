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
  public:
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
    /// fluid depth
    ScalarField<VertexField> depth_passive;
    ScalarField<FaceField> depth_active;
    /// velocity
    VectorField<geo,VertexField> velocity_passive;
    VectorField<geo, FaceField> velocity_active;
    /// Lagrangian particle/panel mesh
    PolyMesh2d<SeedType> mesh;

    /// for planar problems, Coriolis parameter f = f0 + beta*y
    Real f0;
    Real beta;
    /// for spherical problems, Coriolis parameter f = 2*Omega*z
    Real Omega;
    /// time
    Real t;

    void update_host();
    void update_device();

    template <typename TopoType>
    void set_bottom_topography(const TopoType& topo);

    template <typename VelocityType>
    void init_velocity_from_function();

    template <typename VorticityType>
    void init_vorticity(const VorticityType& vort_fn);

    template <typename DivergenceType>
    void init_divergence(const DivergenceType& div_fn);

    template <typename InitialConditions>
    void init_swe_problem(const InitialConditions& ic);

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
