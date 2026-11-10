#ifndef LPM_DFS_GRID_HPP
#define LPM_DFS_GRID_HPP

#include "LpmConfig.h"
#include "lpm_constants.hpp"
#include "lpm_coords.hpp"
#include "lpm_geometry.hpp"
#include "lpm_field.hpp"
#include "lpm_assert.hpp"
#ifdef LPM_USE_VTK
#include "vtkStructuredGrid.h"
#include "vtkSmartPointer.h"
#endif

namespace Lpm {
namespace DFS {

using scalar_field_type = ScalarField<VertexField>;
using vector_field_type = VectorField<SphereGeometry, VertexField>;

/** Creates a uniform colatitude-longitude discretization of the sphere.

  Both poles are included.
*/
struct DFSGrid {
  using view_type = typename SphereGeometry::crd_view_type;

  Int nlon; /// Number of longitude points
  Int nlat; /// Number of colatitude points
  Real dtheta; /// increment of arc length in colatitude
  Real dlambda; /// increment of arc length in longitude direction

  KOKKOS_INLINE_FUNCTION
  Index size() const {return nlon*nlat;}

  Index vtk_size() const {return size() + nlat;}

  /** Constructor.   Given a desired number of longitude points,
    computes nlat = nlon/2 + 1.

    @param [in] nl number of longitude points
  */
  KOKKOS_INLINE_FUNCTION
  explicit DFSGrid(const Int nl) :
    nlon(nl),
    nlat(nl/2 + 1),
    dtheta(2*constants::PI/nl),
    dlambda(2*constants::PI/nl) {
    LPM_REQUIRE_MSG( nl%2 == 0, "use an even number of longitude points for DFSGrid.");
    }

  /**  Return the colatitude of a point in row i of the nlat x nlon grid

    @param [in] i index of row
    @return colatitude
  */
  KOKKOS_INLINE_FUNCTION
  Real colatitude(const Int i) const {
    return i * dtheta;
  }

  /** Return the longitude of a point in column j of the nlat x nlon grid

    @param [in] j index of column
    @return longitude
  */
  KOKKOS_INLINE_FUNCTION
  Real longitude(const Int j) const {
    return j * dlambda;
  }

  /** Return the area represented by the point at idx (i,j)
    @param [in] i colatitude index
    @param [in] j longitude index
    @return area of cell centered at point (i,j)
  */
  KOKKOS_INLINE_FUNCTION
  Real area_weight(const Int i, const Int j) const {
    LPM_KERNEL_ASSERT( (i>=0 and i<=nlat-1) );
    LPM_KERNEL_ASSERT( (j>=0 and j<nlon));
    Real ai;
    if (i==0 or i == nlat-1) {
      ai = 1-cos(0.5*dtheta);
    }
    else {
      ai = 2*sin(colatitude(i))*sin(0.5*dtheta);
    }
    return ai * dlambda;
  }

  /** Return the quadrature weight of the point at idx (i,j).

    Clenshaw-Curtis in z = cos(colatitude) (the colatitude rows are
    Chebyshev-Lobatto points) times the trapezoid rule in longitude, so
    integrals of smooth fields converge spectrally; area_weight() is
    second-order.  The weights sum to 4 pi.

    @param [in] i colatitude index
    @param [in] j longitude index
    @return quadrature weight of point (i,j)
  */
  KOKKOS_INLINE_FUNCTION
  Real quadrature_weight(const Int i, const Int j) const {
    LPM_KERNEL_ASSERT( (i>=0 and i<=nlat-1) );
    LPM_KERNEL_ASSERT( (j>=0 and j<nlon));
    const Int n = nlat - 1;
    Real wi;
    if (i == 0 or i == n) {
      wi = (n%2 == 0 ? 1.0 / (n*n - 1) : 1.0 / (n*n));
    }
    else {
      const Real theta = colatitude(i);
      Real v = 1;
      for (Int k=1; k<=(n-1)/2; ++k) {
        v -= 2*cos(2*k*theta) / (4*k*k - 1);
      }
      if (n%2 == 0) {
        v -= cos(n*theta) / (n*n - 1);
      }
      wi = 2*v/n;
    }
    return wi * dlambda;
  }

  /** Given the i, j (row, column) indices of a point, return its Cartesian coordinates.

    @param [out] xyz Cartesian coordinates (will be overwritten)
    @param [in] i row index, colatitude
    @param [in] j column index, longitude
  */
  template <typename PtType> KOKKOS_INLINE_FUNCTION
  void sph2xyz(PtType& xyz, const Int i, const Int j) const {
    xyz[0] = sin(colatitude(i)) * cos(longitude(j));
    xyz[1] = sin(colatitude(i)) * sin(longitude(j));
    xyz[2] = cos(colatitude(i));
  }

  /** Return the colatitude and azimuthal coordinates of a point, given
    its Cartesian coordinates.
  */
  template <typename PtType> KOKKOS_INLINE_FUNCTION
  void xyz2sph(Real& theta, Real& lambda, const PtType& xyz) const {
    theta = SphereGeometry::colatitude(xyz);
    lambda = SphereGeometry::azimuth(xyz);
  }

  Coords<SphereGeometry> init_coords() const;

  scalar_view_type weights_view() const;
  scalar_view_type quadrature_weights_view() const;

  std::string info_string(const int tab_level=0) const;

  vtkSmartPointer<vtkStructuredGrid> vtk_grid() const;

  private:
  /**  Pack all Cartesian coordinates into an nlat*nlon x 3 view.

    @return coordinate view with point (i,j)'s coordinates at view(i*nlat + j, :)
  */
  void fill_packed_view(view_type& view) const;
};

} // namespace DFS
} // namespace Lpm

#endif
