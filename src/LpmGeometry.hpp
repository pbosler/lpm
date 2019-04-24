#ifndef LPM_GEOMETRY_HPP
#define LPM_GEOMETRY_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmUtilities.hpp"
#include "Kokkos_Core.hpp"
#include "Kokkos_View.hpp"

namespace Lpm {

struct PlaneGeometry {
    static constexpr Int ndim = 2;
    typedef ko::View<Real*[ndim],Dev> crd_view_type;

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

struct SphereGeometry {
    static constexpr Int ndim = 3;
    typedef ko::View<Real*[ndim],Dev> crd_view_type;

    template <typename V> KOKKOS_INLINE_FUNCTION
    static void setzero(V v) {
        v[0] = 0.0;
        v[1] = 0.0;
        v[2] = 0.0;
    }

    template <typename V> KOKKOS_INLINE_FUNCTION
    static void scale(const Real& a, V v) {
        v[0] *= a;
        v[1] *= a;
        v[2] *= a;
    }
    
    template <typename CV, typename CV2> KOKKOS_INLINE_FUNCTION
    static Real dot(const CV a, const CV2 b) {
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
    }
    
    template <typename V, typename CV, typename CV2> KOKKOS_INLINE_FUNCTION
    static void cross(V c, const CV a, const CV2 b) {
        c[0] = a[1]*b[2] - a[2]*b[1];
        c[1] = a[2]*b[0] - a[0]*b[2];
        c[2] = a[0]*b[1] - a[1]*b[0];
    }
    
    template <typename CV> KOKKOS_INLINE_FUNCTION
    static Real norm2(const CV v) {
        return dot(v,v);
    }
    
    template <typename CV> KOKKOS_INLINE_FUNCTION
    static Real mag(const CV v) {
        return std::sqrt(norm2(v));
    }
    
    template <typename V> KOKKOS_INLINE_FUNCTION
    static void normalize(V v) {
        scale(1.0/mag(v), v);
    }
    
    template <typename CV, typename CV2> KOKKOS_INLINE_FUNCTION
    static Real distance(const CV a, const CV2 b) {
        Real cp[3];
        cross<Real*, CV>(cp, a, b);
        const Real dp = dot(a,b);
        return std::atan2(mag<Real*>(cp), dp);
    }
    
    template <typename V, typename CV> KOKKOS_INLINE_FUNCTION
    static void copy(V d, const CV& s) {
        d[0] = s[0];
        d[1] = s[1];
        d[2] = s[2];
    }
    
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
    
    template <typename V, typename CV, typename CV2> KOKKOS_INLINE_FUNCTION
    static void midpoint(V v, const CV& a, const CV2& b) {
        v[0] = 0.5*(a[0] + b[0]);
        v[1] = 0.5*(a[1] + b[1]);
        v[2] = 0.5*(a[2] + b[2]);
        normalize(v);
    }
    
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