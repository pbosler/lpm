#ifndef LPM_2DTRANSPORT_MESH_HPP
#define LPM_2DTRANSPORT_MESH_HPP

#include "Kokkos_Core.hpp"
#include "LpmConfig.h"
#include "lpm_field.hpp"
#include "lpm_logger.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#endif

#include <map>
#include <memory>

namespace Lpm {

template <typename SeedType>
class TransportMesh2d {
 public:
  typedef SeedType seed_type;
  typedef typename SeedType::geo Geo;
  typedef typename SeedType::faceKind FaceType;
  typedef Coords<Geo> coords_type;

  PolyMesh2d<SeedType> mesh;

  std::map<std::string, ScalarField<VertexField>>
      tracer_verts;  /// passive tracers at passive particles
  std::map<std::string, ScalarField<FaceField>>
      tracer_faces;  /// passive tracers at active particles

  VectorField<typename SeedType::geo, VertexField> velocity_verts;
  VectorField<typename SeedType::geo, FaceField> velocity_faces;

  Real t;
  Int t_idx;

  TransportMesh2d(const PolyMeshParameters<SeedType>& params)
      : mesh(params),
        velocity_verts("velocity", params.nmaxverts),
        velocity_faces("velocity", params.nmaxfaces),
        t(0),
        t_idx(0) {}

  template <typename ICType>
  void initialize_tracer(const ICType& tracer_ic,
                         const std::string& = std::string());

  template <typename ICType>
  void set_tracer_from_lag_crds(const ICType& tracer_ic,
                                const Index vert_start_idx = 0,
                                const Index face_start_idx = 0,
                                const std::string& name    = "");

  void allocate_scalar_tracer(const std::string name);

  template <typename VelocityType>
  void initialize_velocity();

  template <typename VelocityType>
  void set_velocity(const Real t, const Index vert_start_idx = 0,
                    const Index face_start_idx = 0);

  inline Int ntracers() const { return tracer_verts.size(); }

  std::string info_string(const int tab_lev = 0) const;

  void update_device() const;

  void update_host() const;

 protected:
};

#ifdef LPM_USE_VTK
template <typename SeedType>
VtkPolymeshInterface<SeedType> vtk_interface(
    const TransportMesh2d<SeedType>& tm);
#endif

}  // namespace Lpm
#endif
