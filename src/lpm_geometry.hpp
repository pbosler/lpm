#ifndef LPM_GEOMETRY_HPP
#define LPM_GEOMETRY_HPP

#include <cassert>

#include "LpmConfig.h"
#include "lpm_assert.hpp"
#include "lpm_constants.hpp"
#include "lpm_kokkos_defs.hpp"
#include "util/lpm_floating_point.hpp"
#include "util/lpm_math.hpp"
#include "util/lpm_tuple.hpp"

namespace Lpm {

/**
  Required members:
    Int ndim : number of dimensions in Euclidean space
    distance(a,b) : distance between two vectors in this space
    tri_area(a,b,c) : area of the triangle formed by vertices a, b, c
    barycenter( vs, n) : barycenter of the n vectors in array vs
    polygonArea(ctr, vs, n): area of the polygon with ctr interior coordinate
  and vertices vs

  Required typedefs:
    crd_view_type : View type associated with vectors in Euclidean space
*/
struct PlaneGeometry {
  static std::string id_string() { return "PlaneGeometry"; }
  static constexpr Int ndim = 2;
  typedef ko::View<Real* [ndim], Dev> crd_view_type;
  typedef ko::View<Real* [ndim], Dev> vec_view_type;

  template <typename V>
  KOKKOS_INLINE_FUNCTION static void set_zero(V v) {
    v[0] = 0.0;
    v[1] = 0.0;
  }

  template <typename V>
  KOKKOS_INLINE_FUNCTION static void scale(const Real& a, V v) {
    v[0] *= a;
    v[1] *= a;
  }

  template <typename CV>
  KOKKOS_INLINE_FUNCTION static Real dot(const CV a, const CV b) {
    return a[0] * b[0] + a[1] * b[1];
  }

  template <typename CV>
  KOKKOS_INLINE_FUNCTION static Real norm2(const CV v) {
    return dot(v, v);
  }

  template <typename CV>
  KOKKOS_INLINE_FUNCTION static Real mag(const CV v) {
    return std::sqrt(norm2(v));
  }

  template <typename CV, typename CV2>
  KOKKOS_INLINE_FUNCTION static Real distance(const CV a, const CV2 b) {
    Real bma[2];
    bma[0] = b[0] - a[0];
    bma[1] = b[1] - a[1];
    return mag(bma);
  }

  template <typename CV, typename CV2, typename CV3>
  KOKKOS_INLINE_FUNCTION static Real tri_area(const CV& va, const CV2& vb,
                                              const CV3& vc) {
    Real bma[2], cma[2];
    bma[0] = vb[0] - va[0];
    bma[1] = vb[1] - va[1];
    cma[0] = vc[0] - va[0];
    cma[1] = vc[1] - va[1];
    const Real ar = bma[0] * cma[1] - bma[1] * cma[0];
    return 0.5 * std::abs(ar);
  }

  template <typename CV, typename CV2, typename CV3>
  KOKKOS_INLINE_FUNCTION static Real cartesian_tri_area(const CV& a,
                                                        const CV2& b,
                                                        const CV3& c) {
    return tri_area(a, b, c);
  }

  template <typename V>
  KOKKOS_INLINE_FUNCTION static void normalize(V v) {
    scale(1.0 / mag(v), v);
  }

  template <typename V, typename CV>
  KOKKOS_INLINE_FUNCTION static void barycenter(V v, const CV cv, const Int n) {
    set_zero(v);
    for (int i = 0; i < n; ++i) {
      v[0] += cv(i, 0);
      v[1] += cv(i, 1);
    }
    scale(1.0 / n, v);
  }

  template <typename V, typename CV, typename Poly>
  KOKKOS_INLINE_FUNCTION static void barycenter(V v, const CV pts,
                                                const Poly& poly, const Int n) {
    set_zero(v);
    for (int i = 0; i < n; ++i) {
      v[0] += pts(poly[i], 0);
      v[1] += pts(poly[i], 1);
    }
    scale(1.0 / n, v);
  }

  template <typename V>
  KOKKOS_INLINE_FUNCTION static void negate(V& v) {
    scale(-1, v);
  }

  template <typename CV>
  KOKKOS_INLINE_FUNCTION static Real polygon_area(const CV& v, const Int n) {
    Real ar = 0;
    for (int i = 0; i < n; ++i) {
      ar += tri_area(ko::subview(v, 0, ko::ALL), ko::subview(v, i, ko::ALL),
                     ko::subview(v, (i + 1) % n, ko::ALL));
    }
    return ar;
  }

  template <typename CV1, typename CV2>
  KOKKOS_INLINE_FUNCTION static Real polygon_area(const CV1& ctr,
                                                  const CV2& verts,
                                                  const Int nverts) {
    Real ar = 0.0;
    for (int i = 0; i < nverts; ++i) {
      ar += tri_area(ctr, ko::subview(verts, i, ko::ALL),
                     ko::subview(verts, (i + 1) % nverts, ko::ALL));
    }
    return ar;
  }

  template <typename V, typename CV>
  KOKKOS_INLINE_FUNCTION static void midpoint(V v, const CV& a, const CV& b) {
    v[0] = 0.5 * (a[0] + b[0]);
    v[1] = 0.5 * (a[1] + b[1]);
  }

  template <typename V, typename CV>
  KOKKOS_INLINE_FUNCTION static void copy(V d, const CV s) {
    d[0] = s[0];
    d[1] = s[1];
  }
};

struct CircularPlaneGeometry : public PlaneGeometry {
  static std::string id_string() { return "CircularPlaneGeometry"; }

  template <typename V, typename CV>
  KOKKOS_INLINE_FUNCTION static void radial_midpoint(V v, const CV& a,
                                                     const CV& b) {
    const Real ra = mag(a);
    const Real rb = mag(b);

    const Real rmid = 0.5 * (ra + rb);
    midpoint(v, a, b);
    normalize(v);
    scale(rmid, v);
  }

  template <typename CV>
  KOKKOS_INLINE_FUNCTION static Real theta(const CV& v) {
    return std::atan2(v[1], v[0]);
  }

  /** @brief Computes the interior angle between two points.
   */
  template <typename V1, typename V2>
  KOKKOS_INLINE_FUNCTION static Real dtheta(const V1& a, const V2& b) {
    const Real dp = dot(a, b);
    const Real alen = mag(a);
    const Real blen = mag(b);
    return std::acos(dp / (alen * blen));
  }

  template <typename V, typename CV>
  KOKKOS_INLINE_FUNCTION static void barycenter(V v, const CV cv,
                                                const Int n = 4) {
    const auto v0 = ko::subview(cv, 0, ko::ALL());
    const auto v2 = ko::subview(cv, 2, ko::ALL());
    const Real router = mag(v0);
    const Real rinner = mag(v2);
    const Real rmid = 0.5 * (router + rinner);
    const Real theta0 = theta(v0);
    const Real dth = 0.5 * dtheta(v0, v2);
    v[0] = rmid * std::cos(theta0 - dth);
    v[1] = rmid * std::sin(theta0 - dth);
  }

  /** @brief Computes the area of the circular sector defined by a polar
rectangle.

@f$ A = \int_{\theta_0}^{\theta_1}\int_{r_0}^{r_1} r\,dr\,d\theta =
\frac{\Delta\theta}{2}(r_1^2 - r_0^2) @f$

    @param outer_ccw the outer, counter-clockwise-most point of the polar
rectangle.
    @param inner_cw the inner, clockwise-most point of the polar rectangle.
  */
  template <typename V1, typename V2>
  KOKKOS_INLINE_FUNCTION static Real quad_sector_area(const V1& outer_ccw,
                                                      const V2& inner_cw) {
    Real result = 0.0;
    const Real r0 = mag(inner_cw);
    const Real r1 = mag(outer_ccw);
    //     assert(r1 > r0);

    const Real dth = dtheta(outer_ccw, inner_cw);
    result = 0.5 * dth * (square(r1) - square(r0));
    return result;
  }

  template <typename CV1, typename CV2>
  KOKKOS_INLINE_FUNCTION static Real polygon_area(const CV1& ctr,
                                                  const CV2& verts,
                                                  const Int nverts) {
    Real result;
    if (mag(ctr) <= 10 * constants::ZERO_TOL) {
      result = constants::PI * norm2(ko::subview(verts, 0, ko::ALL()));
    } else {
      result = quad_sector_area(ko::subview(verts, 0, ko::ALL()),
                                ko::subview(verts, 2, ko::ALL()));
    }
    return result;
  }
};

/**
  \brief Class to handle computations related to spherical geometry.  Stateless.

  Spherical geometry is expressed in 3D Cartesian coordinates
  [x0,x1,x2] with \f$ x_0^2 + x_1^2 + x_2^2 = 1 \f$.
*/
struct SphereGeometry {
  static std::string id_string() { return "SphereGeometry"; }
  static constexpr Int ndim =
      3;  ///<  number of components in a position vector

  typedef ko::View<Real* [ndim], Dev>
      crd_view_type;  ///< vector array type for, e.g., position and velocity
  typedef ko::View<Real* [ndim], Dev> vec_view_type;

  /** \brief Returns the latitude of a point represented in Cartesian
    coordinates.

    Same as azimuth except that the range is [0,2*pi] instead of [-pi,pi].

    \param v view of a position vector v = [v0,v1,v2]
  */
  template <typename CV>
  KOKKOS_INLINE_FUNCTION static Real latitude(const CV v) {
    return std::atan2(v[2], std::sqrt(v[0] * v[0] + v[1] * v[1]));
  }

  /** \brief Returns the latitude of a point represented in Cartesian
    coordinates

    \param v view of a position vector v = [v0,v1,v2]
  */
  template <typename CV>
  KOKKOS_INLINE_FUNCTION static Real longitude(const CV v) {
    return atan4(v[1], v[0]);
  }

  /** \brief Returns the colatitude of a point represented in Cartesian
    coordinates

    \param v view of a position vector v = [v0,v1,v2]
  */
  template <typename CV>
  KOKKOS_INLINE_FUNCTION static Real colatitude(const CV v) {
    return std::atan2(std::sqrt(v[0] * v[0] + v[1] * v[1]), v[2]);
  }

  /** \brief Returns the azimuth of a point represented in Cartesian
    coordinates.

    Same as longitude, except that the range is [-pi,pi] instead of [0, 2*pi];

    \param v view of a position vector v = [v0,v1,v2]
  */
  template <typename CV>
  KOKKOS_INLINE_FUNCTION static Real azimuth(const CV v) {
    return std::atan2(v[1], v[0]);
  }

  /** \brief Returns Cartesian coordiantes of a point with given latitude and longitude

    \param xyz [out] Cartesian coordinates
    \param lon [in] longitude
    \param lat [in] latitude
  */
  template <typename VT>
  KOKKOS_INLINE_FUNCTION static void xyz_from_lon_lat(VT& xyz, const Real lon, const Real lat) {
    xyz[0] = cos(lat)*cos(lon);
    xyz[1] = cos(lat)*sin(lon);
    xyz[2] = sin(lat);
  }

  /** \brief Returns Cartesian coordiantes of a point with given colatitude and azimuth

    \param xyz [out] Cartesian coordinates
    \param az [in] azimuth
    \param colat [in] colatitude
  */
  template <typename VT>
  KOKKOS_INLINE_FUNCTION static void xyz_from_azimuth_colat(VT& xyz, const Real az, const Real colat) {
    xyz[0] = sin(colat)*cos(az);
    xyz[1] = sin(colat)*sin(az);
    xyz[2] = cos(colat);
  }

  /** \brief Sets all components of vector to zero.

  \param v view of a position vector v = [v0,v1,v2]
  */
  template <typename V>
  KOKKOS_INLINE_FUNCTION static void set_zero(V v) {
    v[0] = 0.0;
    v[1] = 0.0;
    v[2] = 0.0;
  }

  /** \brief  Multiplies all components of vector by the same scalar.

  \param a scalar multiplier
  \param v view of a position vector v = [v0,v1,v2]
  */
  template <typename V>
  KOKKOS_INLINE_FUNCTION static void scale(const Real& a, V v) {
    v[0] *= a;
    v[1] *= a;
    v[2] *= a;
  }

  /** \brief Computes the dot product of two vectors

    \param a view of a position vector a = [a0,a1,a2]
    \param b view of a position vector b = [b0,b1,b2]
    \return \f$ a\cdot \f$
  */
  template <typename CV, typename CV2>
  KOKKOS_INLINE_FUNCTION static Real dot(const CV a, const CV2 b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }

  /** \brief Computes the dot product of two vectors

    \param c output view for cross product
    \param a view of a position vector a = [a0,a1,a2]
    \param b view of a position vector b = [b0,b1,b2]
    \return \f$c = a \times b \f$
  */
  template <typename V, typename CV, typename CV2>
  KOKKOS_INLINE_FUNCTION static void cross(V c, const CV a, const CV2 b) {
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
  }

  /** sum y = a*x + y
   */
  template <typename CV, typename V>
  KOKKOS_INLINE_FUNCTION static void axpy(const Real& a, const CV x, V y) {
    y[0] += a * x[0];
    y[1] += a * x[1];
    y[2] += a * x[2];
  }

  /** \brief Computes the dot product of two vectors

    \param a view of a position vector a = [a0,a1,a2]
    \param b view of a position vector b = [b0,b1,b2]
    \return \f$a \times b \f$
  */
  template <typename CV>
  KOKKOS_INLINE_FUNCTION static ko::Tuple<Real, 3> cross(const CV a,
                                                         const CV b) {
    ko::Tuple<Real, 3> c;
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
    return c;
  }

  /** \brief Computes the circumcenter of a triangle using its vertices a, b, c.

    \param cc output view for circumenter
    \param a view of a position vector a = [a0,a1,a2]
    \param b view of a position vector b = [b0,b1,b2]
    \param c view of a position vector c = [bc,c1,c2]
  */
  template <typename V, typename CV, typename CV1 = CV>
  KOKKOS_INLINE_FUNCTION static void circumcenter(V& cc, const CV a,
                                                  const CV1 b, const CV1 c) {
    Real cpab[3], cpbc[3], cpca[3];
    cross(cpab, a, b);
    cross(cpbc, b, c);
    cross(cpca, c, a);
    Real norm = 0.0;
    for (Short i = 0; i < 3; ++i) {
      norm += square(cpab[i] + cpbc[i] + cpca[i]);
    }
    norm = std::sqrt(norm);
    for (Short i = 0; i < 3; ++i) {
      cc[i] = (cpab[i] + cpbc[i] + cpca[i]) / norm;
    }
  }

  /** \brief Computes the squared Euclidean norm of a vector

    \param v view of a position vector v = [v0,v1,v2]
  */
  template <typename CV>
  KOKKOS_INLINE_FUNCTION static Real norm2(const CV v) {
    return dot(v, v);
  }

  /** \brief Computes the Euclidean norm of a vector

    \param v view of a position vector v = [v0,v1,v2]
  */
  template <typename CV>
  KOKKOS_INLINE_FUNCTION static Real mag(const CV v) {
    return std::sqrt(norm2(v));
  }

  /** \brief Normalizes a vector so that its magnitude = 1

    \param v view of a position vector v = [v0,v1,v2]
  */
  template <typename V>
  KOKKOS_INLINE_FUNCTION static void normalize(V v) {
    scale(1.0 / mag(v), v);
  }

  /** \brief Computes the great circle distance between two points on the sphere

    \param a view of a position vector a = [a0,a1,a2]
    \param b view of a position vector b = [b0,b1,b2]
  */
  template <typename CV, typename CV2>
  KOKKOS_INLINE_FUNCTION static Real distance(const CV a, const CV2 b) {
    Real cp[3];
    cross<Real*, CV>(cp, a, b);
    const Real dp = dot(a, b);
    return std::atan2(mag<Real*>(cp), dp);
  }

  /**\brief  Computes the squared Euclidean distance between two points on the
    sphere

    \param a view of a position vector a = [a0,a1,a2]
    \param b view of a position vector b = [b0,b1,b2]
  */
  template <typename CV, typename CV2>
  KOKKOS_INLINE_FUNCTION static Real square_euclidean_distance(const CV a,
                                                               const CV2 b) {
    Real result = 0.0;
    for (int i = 0; i < 3; ++i) {
      result += square(b[i] - a[i]);
    }
    return result;
  }

  /** \brief Constructs a single row of P(x), the projector to the tangent
    plane of the sphere at x.
  */
  template <typename RowType, typename XType>
  KOKKOS_INLINE_FUNCTION static void proj_row(RowType& r, const XType& x, const int row) {
    for (Short j=0; j<3; ++j) {
      r[j] = -x[row]*x[j];
    }
    r[row] += 1;
  }


  /** \brief copies the content of one vector view to another

    \param d destination vector
    \param s source vector
  */
  template <typename V, typename CV>
  KOKKOS_INLINE_FUNCTION static void copy(V d, const CV& s) {
    d[0] = s[0];
    d[1] = s[1];
    d[2] = s[2];
  }

  /** \brief Computes the spherical barycenter defined by n vertices on the
    sphere.

    \param v output view, contains coordinates of barycenter on the sphere
    \param cv input view of vertex vectors
    \param n number of vertices
  */
  template <typename V, typename CV>
  KOKKOS_INLINE_FUNCTION static void barycenter(V v, const CV& cv,
                                                const Int n) {
    set_zero(v);
    for (int i = 0; i < n; ++i) {
      v[0] += cv(i, 0);
      v[1] += cv(i, 1);
      v[2] += cv(i, 2);
    }
    scale(1.0 / n, v);
    normalize(v);
  }

  template <typename V, typename CV, typename Poly>
  KOKKOS_INLINE_FUNCTION static void barycenter(V v, const CV pts,
                                                const Poly& poly, const Int n) {
    set_zero(v);
    for (int i = 0; i < n; ++i) {
      v[0] += pts(poly[i], 0);
      v[1] += pts(poly[i], 1);
      v[2] += pts(poly[i], 2);
    }
    scale(1.0 / n, v);
    normalize(v);
  }

  /** \brief Computes the spherical midpoint between two vectors on the sphere

    \param v output vector, contains the coordinates of the midpoint
    \param a vertex a = [a0,a1,a2]
    \param b vertex b = [b0,b1,b2]
  */
  template <typename V, typename CV, typename CV2>
  KOKKOS_INLINE_FUNCTION static void midpoint(V v, const CV& a, const CV2& b) {
    v[0] = 0.5 * (a[0] + b[0]);
    v[1] = 0.5 * (a[1] + b[1]);
    v[2] = 0.5 * (a[2] + b[2]);
    normalize(v);
  }

  /** \brief  Computes the area of the spherical triangle whose vertices a
    defined (in ccw order) by a, b, c.

    \param a vertex a = [a0,a1,a2]
    \param b vertex b = [b0,b1,b2]
    \param c vertex c = [c0,c1,c2]
  */
  template <typename CV, typename CV2>
  KOKKOS_INLINE_FUNCTION static Real tri_area(const CV& a, const CV2& b,
                                              const CV2& c) {
    const Real s1 = distance(a, b);
    const Real s2 = distance(b, c);
    const Real s3 = distance(c, a);
    const Real half_perim = 0.5 * (s1 + s2 + s3);
    Real zz = std::tan(0.5 * half_perim) * std::tan(0.5 * (half_perim - s1)) *
              std::tan(0.5 * (half_perim - s2)) *
              std::tan(0.5 * (half_perim - s3));
    if (FloatingPoint<Real>::zero(zz)) {
      // guard against (0 - epsilon)
      zz = 0;
    }
    return 4 * atan(sqrt(zz));
  }

  template <typename CV, typename CV2, typename CV3>
  KOKKOS_INLINE_FUNCTION static Real cartesian_tri_area(const CV& a,
                                                        const CV2& b,
                                                        const CV3& c) {
    Real ab[3];
    Real ac[3];
    for (Int i = 0; i < 3; ++i) {
      ab[i] = b[i] - a[i];
      ac[i] = c[i] - a[i];
    }
    Real cp[3];
    cross(cp, ab, ac);
    return 0.5 * mag(cp);
  }

  /** \brief  Computes the area of the spherical polygon, given its vertices and
    centroid

    \param ctr centroid of polygon
    \param verts vertices of polygon, in counter-clockwise (ccw) order
    \param nverts nubmer of vertices in polygon >= 3
  */
  template <typename CV1, typename CV2>
  KOKKOS_INLINE_FUNCTION static Real polygon_area(const CV1& ctr,
                                                  const CV2& verts,
                                                  const Int nverts) {
    Real ar = 0;
    for (int i = 0; i < nverts; ++i) {
      ar += tri_area(ctr, ko::subview(verts, i, ko::ALL),
                     ko::subview(verts, (i + 1) % nverts, ko::ALL));
    }
    return ar;
  }
};

}  // namespace Lpm
#endif
