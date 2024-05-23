#ifndef LPM_STAGGERED_SWE_HPP
#define LPM_STAGGERED_SWE_HPP

#include "LpmConfig.h"
#include "lpm_compadre.hpp"
#include "lpm_coriolis.hpp"
#include "lpm_coords.hpp"
#include "lpm_field.hpp"
#include "lpm_logger.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#endif

namespace Lpm {

template <typename SeedType, typename TopoType>
class StaggeredSWE {
  public: // members
    using geo = typename SeedType::geo;
    using coords_type = Coords<geo>;
    using crd_view = typename coords_type::view_type;
    using vec_view = typename geo::vec_view_type;
    using Coriolis = typename std::conditional<
      std::is_same<geo, PlaneGeometry>::value,
      CoriolisBetaPlane, CoriolisSphere>::type;

    // face fields
    ScalarField<FaceField> relative_vorticity;
    ScalarField<FaceField> potential_vorticity;
    ScalarField<FaceField> divergence;
    ScalarField<FaceField> mass;
    ScalarField<FaceField> double_dot_avg;
    ScalarField<FaceField> grad_f_cross_u_avg;
    ScalarField<FaceField> depth;
    ScalarField<FaceField> bottom_height;
    ScalarField<FaceField> surface_height;
    ScalarField<FaceField> surface_laplacian;
    VectorField<geo, FaceField> velocity_avg;
    std::map<std::string, ScalarField<FaceField>> tracers;

    // vertex fields
    VectorField<geo, VertexField> velocity;
    ScalarField<VertexField> double_dot;
    ScalarField<VertexField> grad_f_cross_u;
    std::map<std::string, ScalarField<VertexField>> diags;

    /// Lagrangian particle/panel mesh
    PolyMesh2d<SeedType> mesh;

    /// Coriolis
    Coriolis coriolis;

    /// gravity
    Real g;
    /// time
    Real t;
    /// velocity kernel smoothing parameter
    Real eps;

    /// bottom topography
    TopoType topography;

  public: // functions

    StaggeredSWE(const PolyMeshParameters<SeedType>& mesh_params, const Coriolis& coriolis, const TopoType& topography, const std::shared_ptr<spdlog::logger> logger_in=nullptr);

    template <typename SurfaceType, typename VorticityType, typename DivergenceType>
    void init_fields(const SurfaceType& sfc, const VorticityType& vorticity,
      const DivergenceType& div, const gmls::Params gmls_paramsejckcbidtnbkvillrbjhc);

    template <typename SolverType>
    void advance_timestep(SolverType& solver);

    void update_host();
    void update_device();

    void allocate_scalar_tracer(const std::string& name);
    void allocate_scalar_diag(const std::string& name);

    std::string info_string(const int tab_level=0, const bool verbose=false) const;

    void gmls_surface_laplacian(const crd_view& face_x, const gmls::Params& gmls_params);

  private: // functions

    void gmls_surface_laplacian(const gmls::Params& gmls_params);

    /** @brief initializes fields that must be computed, i.e.,
      velocity (optionally)
      the double dot product, and the surfaace Laplacian,
      using direct summation.
    */
    void init_direct_sums(const bool do_velocity = true);

    /**  @brief Sets bottom topography values on all particles.

      Should be called at each time increment to update bottom
      topography values to new particle locations.
    */
    void set_bottom_topography();

    std::shared_ptr<spdlog::logger> logger;
};

#ifdef LPM_USE_VTK
  template <typename SeedType, typename TopoType>
  VtkPolymeshInterface<SeedType> vtk_mesh_interface(const StaggeredSWE<SeedType,TopoType>& swe);
#endif

} // namespace Lpm

#endif
