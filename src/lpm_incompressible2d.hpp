#ifndef LPM_INCOMPRESSIBLE2D_HPP
#define LPM_INCOMPRESSIBLE2D_HPP

#include "LpmConfig.h"
#include "lpm_coriolis.hpp"
#include "lpm_field.hpp"
#include "mesh/lpm_bivar_remesh.hpp"
#include "mesh/lpm_polymesh2d.hpp"

#include <map>

namespace Lpm {

template <typename SeedType>
class Incompressible2D {
  public: // member variables
    using geo = typename SeedType::geo;
    using Coriolis = typename std::conditional<
      std::is_same<geo, PlaneGeometry>::value,
      CoriolisBetaPlane, CoriolisSphere>::type;

    /// relative vorticity
    ScalarField<VertexField> rel_vort_passive;
    ScalarField<FaceField> rel_vort_active;
    /// potential vorticity
    ScalarField<VertexField> abs_vort_passive;
    ScalarField<FaceField> abs_vort_active;
    /// stream function
    ScalarField<VertexField> stream_fn_passive;
    ScalarField<FaceField> stream_fn_active;
    /// velocity
    VectorField<geo,VertexField> velocity_passive;
    VectorField<geo, FaceField> velocity_active;
    /// passive tracers at passive particles
    std::map<std::string, ScalarField<VertexField>> tracer_passive;
    /// passive tracers at active particles
    std::map<std::string, ScalarField<FaceField>> tracer_active;

    /// Lagrangian particle/panel mesh
    PolyMesh2d<SeedType> mesh;

    Coriolis coriolis;

    /// time
    Real t;
    /// velocity kernel smoothing parameter
    Real eps;

  public:  // functions

    Incompressible2D(const PolyMeshParameters<SeedType>& mesh_params, const Coriolis& coriolis, const Real velocity_eps); //, const std::vector<std::string>& tracers = std::vector<std::string>());

    void update_host();
    void update_device();

    template <typename VorticityType>
    void init_vorticity(const VorticityType& vorticity);

    template <typename TracerType>
    void init_tracer(const TracerType& tracer, const std::string& tname = std::string());

    void init_direct_sums();

    template <typename VelocityType>
    void init_velocity();

    template <typename SolverType>
    void advance_timestep(SolverType& solver);

    Int n_tracers() const;

    std::string info_string(const int tab_level=0) const;

  private:
};

#ifdef LPM_USE_VTK
  template <typename SeedType>
  VtkPolymeshInterface<SeedType> vtk_mesh_interface(const Incompressible2D<SeedType>& ic2d);
#endif

template <typename SeedType>
BivarRemesh<SeedType> bivar_remesh(Incompressible2D<SeedType>& new_ic2d, const Incompressible2D<SeedType>& old_ic2d);

} // namespace Lpm

#endif
