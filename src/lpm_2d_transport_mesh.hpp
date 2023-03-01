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

template <typename VelocityFtor>
struct VelocityKernel {
  Kokkos::View<Real**> velocity;
  Kokkos::View<Real**> xcrds;
  Real t;
  VelocityFtor velfn;

  VelocityKernel(Kokkos::View<Real**> u, const Kokkos::View<Real**> x,
                 const Real tt)
      : velocity(u), xcrds(x), t(tt) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index i) const {
    const auto myx = Kokkos::subview(xcrds, i, Kokkos::ALL);
    auto myu = Kokkos::subview(velocity, i, Kokkos::ALL);
    Kokkos::Tuple<Real, VelocityFtor::ndim> u = velfn(myx, t);
    for (int j = 0; j < VelocityFtor::ndim; ++j) {
      myu(j) = u[j];
    }
  }
};

template <typename SeedType>
class TransportMesh2d : public PolyMesh2d<SeedType> {
 public:
  typedef SeedType seed_type;
  typedef typename SeedType::geo Geo;
  typedef typename SeedType::faceKind FaceType;
  typedef Coords<Geo> coords_type;
  typedef std::shared_ptr<Coords<Geo>> coords_ptr;

  std::map<std::string, ScalarField<VertexField>>
      tracer_verts;  /// passive tracers at passive particles
  std::map<std::string, ScalarField<FaceField>>
      tracer_faces;  /// passive tracers at active particles

  VectorField<typename SeedType::geo, VertexField> velocity_verts;
  VectorField<typename SeedType::geo, FaceField> velocity_faces;

  Real t;
  Int t_idx;

  TransportMesh2d(const PolyMeshParameters<SeedType>& params)
      : PolyMesh2d<SeedType>(params),
        velocity_verts("velocity", params.nmaxverts),
        velocity_faces("velocity", params.nmaxfaces),
        t(0),
        t_idx(0) {}

  template <typename ICType>
  void initialize_tracer(const ICType& tracer_ic);

  void initialize_scalar_tracer(const std::string name);

  template <typename VelocityType>
  void initialize_velocity();

  inline Int ntracers() const { return tracer_verts.size(); }

  std::string info_string(const int tab_lev = 0) const /*override*/;

  void update_device() const override;

  void update_host() const override;

 protected:
};

#ifdef LPM_USE_VTK
template <typename SeedType>
VtkPolymeshInterface<SeedType> vtk_interface(
    const std::shared_ptr<TransportMesh2d<SeedType>> tm);
#endif

}  // namespace Lpm
#endif
