#ifndef LPM_MONATOMIC_GAS2D_HPP
#define LPM_MONATOMIC_GAS2D_HPP

#include "LpmConfig.h"
#include "lpm_assert.hpp"
#include "lpm_coords.hpp"
#include "lpm_field.hpp"
#include "lpm_geometry.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#include "util/lpm_floating_point.hpp"

namespace Lpm {

/** Computes the square of the local speed of sound

  @f$ a^2 = (\gamma - 1) * h @f$

  @param [in] gamma specific heat ratio
  @param [in] h enthalpy
  @return a_square
*/
KOKKOS_INLINE_FUNCTION
Real a_square(const Real gamma, const Real h) {
  LPM_KERNEL_ASSERT( !FloatingPoint<Real>::zero(h) );
  return (gamma - 1)/h;
}

/** A class for solving 2D monatomic gas problems using the methods in
  Eldredge, Colonius, and Leonard, "A vortex particle method
  for two-dimensional compressible flow," JCP, 2002.
*/
template <typename SeedType>
class MonatomicGas2D {
  using geo = typename SeedType::geo;
  static_assert(std::is_same<typename SeedType::geo, PlaneGeometry>::value,
    "planar geometry required.");

  /// Reference coordinates for FTLE
  Coords<geo> ref_crds_vertices;
  Coords<geo> ref_crds_faces;

  /// vorticity (@f$ \omega @f$)
  ScalarField<VertexField> vorticity_verts;
  ScalarField<FaceField> vorticity_faces;
  /// dilatation (@f$ \theta @f$)
  ScalarField<VertexField> dilatation_verts;
  ScalarField<FaceField> dilatation_faces;
  /// velocity (@f$ \vec{u} @f$)
  VectorField<geo, VertexField> velocity_verts;
  VectorField<geo, FaceField> velocity_faces;
  /// density (@f$ \rho @f$)
  ScalarField<VertexField> density_verts;
  ScalarField<FaceField> density_faces;
  /// entropy (@f$ s @f$)
  ScalarField<VertexField> entropy_verts;
  ScalarFIeld<FaceField> entropy_faces;
  /// enthalpy (@f$ h @f$)
  ScalarField<VertexField> enthalpy_verts;
  ScalarField<FaceField> enthalpy_faces;
  /// enthalpy Laplacian (@f$ \nabla^2 h @f$)
  ScalarField<VertexField> enthalpy_laplacian_verts;
  ScalarField<FaceField> enthalpy_laplacian_faces;
  /// viscous dissipation (@f$ \Phi @f$)
  ScalarField<VertexField> visc_verts;
  ScalarField<FaceField> visc_faces;

  /// boundary zone flags
  /// in_boundary_verts(i) = true if particle i is in the boundary zone;
  /// similarly for faces.
  mask_view_type in_boundary_verts;
  mask_view_type in_boundary_faces;

  /// specific heat ratio, c_p / c_v
  Real gamma;

  Real t;

};

}
#endif
