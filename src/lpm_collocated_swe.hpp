#ifndef LPM_COLLOCATED_SWE_HPP
#define LPM_COLLOCATED_SWE_HPP

#include "LpmConfig.h"
#include "lpm_compadre.hpp"
#include "lpm_coriolis.hpp"
#include "lpm_field.hpp"
#include "lpm_coords.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#include "vtk/lpm_vtk_io.hpp"

namespace Lpm {

namespace colloc {
/**  Collocated method for use with high order PSE kernels.  All fields are
  FaceFields; passive particles carry no data. They are only kept for use with
  adaptive methods and FTLE.
*/
template <typename SeedType>
class CollocatedSWE {

  public: // members
    using geometry_t = typename SeedType::geo;
    using coords_t = Coords<geometry_t>;
    using crd_view = typename coords_t::view_type;
    using vec_view = typename geometry_t::vec_view_type;
    using Coriolis = typename std::conditional<
      std::is_same<geometry_t, PlaneGeometry>::value,
      CoriolisBetaPlane, CoriolisSphere>::type;

    /// relative vorticity
    ScalarField<FaceField> rel_vort;
    /// potential vorticity
    ScalarField<FaceField> pot_vort;
    /// divergence
    ScalarField<FaceField> divergence;
    /// surface height
    ScalarField<FaceField> surface;
    // bottom topography
    ScalarField<FaceField> bottom;
    /// surface laplacian
    ScalarField<FaceField> surface_lap;
    /// fluid depth
    ScalarField<FaceField> depth;
    /// double dot product
    ScalarField<FaceField> double_dot;
    /// finite-time Lyaponov exponents
    ScalarField<FaceField> ftle;

    /// velocity derivatives
    // TODO: remove these after double dot is verified
    ScalarField<FaceField> du1dx1;
    ScalarField<FaceField> du1dx2;
    ScalarField<FaceField> du2dx1;
    ScalarField<FaceField> du2dx2;

    /// velocity
    VectorField<geometry_t, FaceField> velocity_active;
    VectorField<geometry_t, VertexField> velocity_passive;
    /// mass
    ScalarField<FaceField> mass;
    /// Lagrangian particle/panel mesh
    PolyMesh2d<SeedType> mesh;
    /// reference coordinates, for FTLE computations
    Coords<geometry_t> ref_crds_passive;
    Coords<geometry_t> ref_crds_active;
    /// Coriolis parameter and derivatives
    Coriolis coriolis;
    /// passive tracers
    std::map<std::string, ScalarField<FaceField>> tracers;
    /// gravity
    Real g;
    /// time
    Real t;
    /// kernel smoothing parameter
    Real eps;

  public: // functions

    void update_host();
    void update_device();

    void allocate_scalar_tracer(const std::string& name);

    /**  @brief Sets bottom topography values on all particles.

      Should be called at each time increment to update bottom
      topography values to new particle locations.
    */
    template <typename TopoType>
    void set_bottom_topography(const TopoType& topo);

    // constructor
    CollocatedSWE(const PolyMeshParameters<SeedType>& mesh_params, const Coriolis& coriolis);

    void set_kernel_width_from_power(const Real power);

    void set_kernel_width_from_multiplier(const Real multiplier);

    Real total_mass() const;
    Real total_vorticity() const;
    Real total_divergence() const;
    Real total_enstrophy() const;
    Real total_potential_energy() const;
    Real total_kinetic_energy() const;
    Real total_energy() const;

    template <typename KernelType>
    void compute_surface_laplacian_pse(const KernelType& kernels);

    template <typename KernelType>
    void compute_velocity_direct_sum(const KernelType& kernels);

    void compute_surface_laplacian_gmls(const gmls::Params& gmls_params);

    template <typename TopographyType, typename SurfaceType>
    void init_surface_and_depth(const TopographyType& b, const SurfaceType& s);

    template <typename VorticityType, typename DivergenceType>
    void init_vorticity_and_divergence(const VorticityType& zeta, const DivergenceType& delta);

    template <typename TopographyType, typename SurfaceType, typename VorticityType, typename DivergenceType, typename KernelType>
    void init_swe_problem(const TopographyType& topo, const SurfaceType& sfc, const VorticityType& zeta, const DivergenceType& delta, const KernelType& kernels);

    private:
      bool depth_set;
};

template <typename SeedType>
VtkPolymeshInterface<SeedType> vtk_mesh_interface(const CollocatedSWE<SeedType>& swe);

} // namespace colloc
} // namespace Lpm
#endif
