#ifndef LPM_SWE_HIGH_ORDER_HPP
#define LPM_SWE_HIGH_ORDER_HPP

#include "LpmConfig.h"
#include "lpm_coriolis.hpp"
#include "lpm_field.hpp"
#include "lpm_coords.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#include "vtk/lpm_vtk_io.hpp"

template <typename SeedType, typename BoundaryType>
struct SWEHighOrder {
  public: // members
    using geo = typename SeedType::geo;
    using coords_type = Coords<geo>;
    using crd_view = typename coords_type::view_type;
    using vec_view = typename geo::vec_view_type;
    using Coriolis = typename std::conditional<
      std::is_same<geo, PlaneGeometry>::value,
      CoriolisBetaPlane, CoriolisSphere>::type;

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

};

#endif
