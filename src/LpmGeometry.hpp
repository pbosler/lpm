#ifndef LPM_GEOMETRY_HPP
#define LPM_GEOMETRY_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmUtilities.hpp"
#include "Kokkos_Core.hpp"
#include "Kokkos_View.hpp"
#include "LpmKokkosUtil.hpp"

namespace Lpm {

/**
    Required members:
        Int ndim : number of dimensions in Euclidean space
        distance(a,b) : distance between two vectors in this space
        triArea(a,b,c) : area of the triangle formed by vertices a, b, c
        barycenter( vs, n) : barycenter of the n vectors in array vs
        polygonArea(ctr, vs, n): area of the polygon with ctr interior coordinate and vertices vs

    Required typedefs:
        crd_view_type : View type associated with vectors in Euclidean space
*/
struct PlaneGeometry {
    static std::string idString() {return "PlaneGeometry";}
    static constexpr Int ndim = 2;
    typedef ko::View<Real*[ndim],Dev> crd_view_type;
    typedef ko::View<Real*[ndim],Dev> vec_view_type;

    template <typename V> KOKKOS_INLINE_FUNCTION
    static void setzero(V v) {
        v[0] = 0.0;
        v[1] = 0.0;
    }

    template <typename V> KOKKOS_INLINE_FUNCTION
    static void scale (const Real& a, V v) {
        v[0] *= a;
        v[1] *= a;
    }

    template <typename CV> KOKKOS_INLINE_FUNCTION
    static Real dot(const CV a, const CV b) {
        return a[0]*b[0] + a[1]*b[1];
    }

    template <typename CV> KOKKOS_INLINE_FUNCTION
    static Real norm2(const CV v) {
        return dot(v,v);
    }

    template <typename CV> KOKKOS_INLINE_FUNCTION
    static Real mag(const CV v) {
        return std::sqrt(norm2(v));
    }

    template <typename CV> KOKKOS_INLINE_FUNCTION
    static Real distance(const CV a, const CV b) {
        Real bma[2];
        bma[0] = b[0] - a[0];
        bma[1] = b[1] - a[1];
        return mag(bma);
    }

    template <typename CV, typename CV2> KOKKOS_INLINE_FUNCTION
    static Real triArea(const CV& va, const CV2& vb, const CV2& vc) {
        Real bma[2], cma[2];
        bma[0] = vb[0] - va[0];
        bma[1] = vb[1] - va[1];
        cma[0] = vc[0] - va[0];
        cma[1] = vc[1] - va[1];
        const Real ar = bma[0]*cma[1] - bma[1]*cma[0];
        return 0.5*std::abs(ar);
    }

    template <typename V> KOKKOS_INLINE_FUNCTION
    static void normalize(V v) {
        scale(1.0/mag(v), v);
    }

    template <typename V, typename CV> KOKKOS_INLINE_FUNCTION
    static void barycenter(V v, const CV cv, const Int n) {
        setzero(v);
        for (int i=0; i<n; ++i) {
            v[0] += cv(i,0);
            v[1] += cv(i,1);
        }
        scale(1.0/n, v);
    }

    template <typename CV> KOKKOS_INLINE_FUNCTION
    static Real polygonArea(const CV& v, const Int n) {
        Real ar = 0;
        for (int i=0; i<n; ++i) {
            ar += triArea(slice(v,0), slice(v,i), slice(v,(i+1)%n));
        }
        return ar;
    }

    template <typename CV1, typename CV2> KOKKOS_INLINE_FUNCTION
    static Real polygonArea(const CV1& ctr, const CV2& verts, const Int nverts) {
        Real ar = 0.0;
        for (int i=0; i<nverts; ++i) {
            ar += triArea(ctr, slice(verts,i), slice(verts, (i+1)%nverts));
        }
        return ar;
    }

    template <typename V, typename CV> KOKKOS_INLINE_FUNCTION
    static void midpoint(V v, const CV& a, const CV& b) {
        v[0] = 0.5*(a[0] + b[0]);
        v[1] = 0.5*(a[1] + b[1]);
    }

    template <typename V, typename CV> KOKKOS_INLINE_FUNCTION
    static void copy(V d, const CV s) {
        d[0] = s[0];
        d[1] = s[1];
    }
};

/**
  \brief Class to handle computations related to spherical geometry.  Stateless.

  Spherical geometry is expressed in 3D Cartesian coordinates
  [x0,x1,x2] with \f$ x_0^2 + x_1^2 + x_2^2 = 1 \f$.
*/
struct SphereGeometry {
    static std::string idString() {return "SphereGeometry";}
    static constexpr Int ndim = 3; ///<  number of components in a position vector

    typedef ko::View<Real*[ndim],Dev> crd_view_type; ///< vector array type for, e.g., position and velocity
    typedef ko::View<Real*[ndim],Dev> vec_view_type;

    /** \brief Returns the latitude of a point represented in Cartesian coordinates.

      Same as azimuth except that the range is [0,2*pi] instead of [-pi,pi].

      \param v view of a position vector v = [v0,v1,v2]
    */
    template <typename CV> KOKKOS_INLINE_FUNCTION
    static Real latitude(const CV v) {
        return std::atan2(v[2], std::sqrt(v[0]*v[0] + v[1]*v[1]));
    }

    /** \brief Returns the latitude of a point represented in Cartesian coordinates

      \param v view of a position vector v = [v0,v1,v2]
    */
    template <typename CV> KOKKOS_INLINE_FUNCTION
    static Real longitude(const CV v) {
        return atan4(v[1], v[0]);
    }

    /** \brief Returns the colatitude of a point represented in Cartesian coordinates

      \param v view of a position vector v = [v0,v1,v2]
    */
    template <typename CV> KOKKOS_INLINE_FUNCTION
    static Real colatitude(const CV v) {
      return std::atan2(std::sqrt(v[0]*v[0] + v[1]*v[1]), v[2]);
    }

    /** \brief Returns the azimuth of a point represented in Cartesian coordinates.

      Same as longitude, except that the range is [-pi,pi] instead of [0, 2*pi];

      \param v view of a position vector v = [v0,v1,v2]
    */
    template <typename CV> KOKKOS_INLINE_FUNCTION
    static Real azimuth(const CV v) {
      return std::atan2(v[1],v[0]);
    }

    /** \brief Sets all components of vector to zero.

    \param v view of a position vector v = [v0,v1,v2]
    */
    template <typename V> KOKKOS_INLINE_FUNCTION
    static void setzero(V v) {
        v[0] = 0.0;
        v[1] = 0.0;
        v[2] = 0.0;
    }

    /** \brief  Multiplies all components of vector by the same scalar.

    \param a scalar multiplier
    \param v view of a position vector v = [v0,v1,v2]
    */
    template <typename V> KOKKOS_INLINE_FUNCTION
    static void scale(const Real& a, V v) {
        v[0] *= a;
        v[1] *= a;
        v[2] *= a;
    }

    /** \brief Computes the dot product of two vectors

      \param a view of a position vector a = [a0,a1,a2]
      \param b view of a position vector b = [b0,b1,b2]
      \return \f$ a\cdot \f$
    */
    template <typename CV, typename CV2> KOKKOS_INLINE_FUNCTION
    static Real dot(const CV a, const CV2 b) {
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    }

    /** \brief Computes the dot product of two vectors

      \param c output view for cross product
      \param a view of a position vector a = [a0,a1,a2]
      \param b view of a position vector b = [b0,b1,b2]
      \return \f$c = a \times b \f$
    */
    template <typename V, typename CV, typename CV2> KOKKOS_INLINE_FUNCTION
    static void cross(V c, const CV a, const CV2 b) {
        c[0] = a[1]*b[2] - a[2]*b[1];
        c[1] = a[2]*b[0] - a[0]*b[2];
        c[2] = a[0]*b[1] - a[1]*b[0];
    }

    /** \brief Computes the dot product of two vectors

      \param a view of a position vector a = [a0,a1,a2]
      \param b view of a position vector b = [b0,b1,b2]
      \return \f$a \times b \f$
    */
    template <typename CV> KOKKOS_INLINE_FUNCTION
    static ko::Tuple<Real,3> cross(const CV a, const CV b) {
        ko::Tuple<Real,3> c;
        c[0] = a[1]*b[2] - a[2]*b[1];
        c[1] = a[2]*b[0] - a[0]*b[2];
        c[2] = a[0]*b[1] - a[1]*b[0];
        return c;
    }


    /** \brief Computes the circumcenter of a triangle using its vertices a, b, c.

      \param cc output view for circumenter
      \param a view of a position vector a = [a0,a1,a2]
      \param b view of a position vector b = [b0,b1,b2]
      \param c view of a position vector c = [bc,c1,c2]
    */
    template <typename V, typename CV, typename CV1=CV> KOKKOS_INLINE_FUNCTION
    static void circumcenter(V& cc, const CV a, const CV1 b, const CV1 c) {
        Real cpab[3], cpbc[3], cpca[3];
        cross(cpab, a, b);
        cross(cpbc, b, c);
        cross(cpca, c, a);
        Real norm = 0.0;
        for (Short i=0; i<3; ++i) {
            norm += square(cpab[i] + cpbc[i] + cpca[i]);
        }
        norm = std::sqrt(norm);
        for (Short i=0; i<3; ++i) {
            cc[i] = (cpab[i] + cpbc[i] + cpca[i])/norm;
        }
    }


    /** \brief Computes the squared Euclidean norm of a vector

      \param v view of a position vector v = [v0,v1,v2]
    */
    template <typename CV> KOKKOS_INLINE_FUNCTION
    static Real norm2(const CV v) {
        return dot(v,v);
    }

    /** \brief Computes the Euclidean norm of a vector

      \param v view of a position vector v = [v0,v1,v2]
    */
    template <typename CV> KOKKOS_INLINE_FUNCTION
    static Real mag(const CV v) {
        return std::sqrt(norm2(v));
    }

    /** \brief Normalizes a vector so that its magnitude = 1

      \param v view of a position vector v = [v0,v1,v2]
    */
    template <typename V> KOKKOS_INLINE_FUNCTION
    static void normalize(V v) {
        scale(1.0/mag(v), v);
    }

    /** \brief Computes the great circle distance between two points on the sphere

      \param a view of a position vector a = [a0,a1,a2]
      \param b view of a position vector b = [b0,b1,b2]
    */
    template <typename CV, typename CV2> KOKKOS_INLINE_FUNCTION
    static Real distance(const CV a, const CV2 b) {
        Real cp[3];
        cross<Real*, CV>(cp, a, b);
        const Real dp = dot(a,b);
        return std::atan2(mag<Real*>(cp), dp);
    }


    /**\brief  Computes the squared Euclidean distance between two points on the sphere

      \param a view of a position vector a = [a0,a1,a2]
      \param b view of a position vector b = [b0,b1,b2]
    */
    template <typename CV, typename CV2> KOKKOS_INLINE_FUNCTION
    static Real sqEuclideanDistance(const CV a, const CV2 b) {
        Real result=0.0;
        for (int i=0; i<3; ++i) {
            result += square(b[i]-a[i]);
        }
        return result;
    }

    /** \brief copies the content of one vector view to another

      \param d destination vector
      \param s source vector
    */
    template <typename V, typename CV> KOKKOS_INLINE_FUNCTION
    static void copy(V d, const CV& s) {
        d[0] = s[0];
        d[1] = s[1];
        d[2] = s[2];
    }


    /** \brief Computes the spherical barycenter defined by n vertices on the sphere.

      \param v output view, contains coordinates of barycenter on the sphere
      \param cv input view of vertex vectors
      \param n number of vertices
    */
    template <typename V, typename CV> KOKKOS_INLINE_FUNCTION
    static void barycenter(V v, const CV& cv, const Int n) {
        setzero(v);
        for (int i=0; i<n; ++i) {
            v[0] += cv(i,0);
            v[1] += cv(i,1);
            v[2] += cv(i,2);
        }
        scale(1.0/n, v);
        normalize(v);
    }

    /** \brief Computes the spherical midpoint between two vectors on the sphere

      \param v output vector, contains the coordinates of the midpoint
      \param a vertex a = [a0,a1,a2]
      \param b vertex b = [b0,b1,b2]
    */
    template <typename V, typename CV, typename CV2> KOKKOS_INLINE_FUNCTION
    static void midpoint(V v, const CV& a, const CV2& b) {
        v[0] = 0.5*(a[0] + b[0]);
        v[1] = 0.5*(a[1] + b[1]);
        v[2] = 0.5*(a[2] + b[2]);
        normalize(v);
    }

    /** \brief  Computes the area of the spherical triangle whose vertices a defined (in ccw order) by a, b, c.

      \param a vertex a = [a0,a1,a2]
      \param b vertex b = [b0,b1,b2]
      \param c vertex c = [c0,c1,c2]
    */
    template <typename CV, typename CV2> KOKKOS_INLINE_FUNCTION
    static Real triArea(const CV& a, const CV2& b, const CV2& c) {
        const Real s1 = distance(a, b);
        const Real s2 = distance(b, c);
        const Real s3 = distance(c, a);
        const Real half_perim = 0.5*(s1 + s2 + s3);
        const Real zz = std::tan(0.5*half_perim) * std::tan(0.5*(half_perim-s1)) * std::tan(0.5*(half_perim-s2)) *
            std::tan(0.5*(half_perim-s3));
        return 4*std::atan(std::sqrt(zz));
    }

    /** \brief  Computes the area of the spherical polygon, given its vertices and centroid

      \param ctr centroid of polygon
      \param verts vertices of polygon, in counter-clockwise (ccw) order
      \param nverts nubmer of vertices in polygon >= 3
    */
    template <typename CV1, typename CV2> KOKKOS_INLINE_FUNCTION
    static Real polygonArea(const CV1& ctr, const CV2& verts, const Int nverts) {
        Real ar = 0;
        for (int i=0; i<nverts; ++i) {
            ar += triArea(ctr, slice(verts,i), slice(verts, (i+1)%nverts));
        }
        return ar;
    }
};

}
#endif
