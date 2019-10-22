#ifndef LPM_BOX3D_HPP
#define LPM_BOX3D_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmKokkosUtil.hpp"
#include "Kokkos_Core.hpp"
#include "LpmUtilities.hpp"
#include <iostream>
#include <iomanip>
#include <cassert>

namespace Lpm {
namespace Octree {

struct BBox {
    Real xmin, xmax;
    Real ymin, ymax;
    Real zmin, zmax;
    
    KOKKOS_INLINE_FUNCTION
    BBox() {init();}

    KOKKOS_INLINE_FUNCTION
    BBox(const Real& xl, const Real& xr, const Real& yl, const Real& yr, const Real& zl, const Real& zr) :
        xmin(xl), xmax(xr), ymin(yl), ymax(yr), zmin(zl), zmax(zr) {}
        
    KOKKOS_INLINE_FUNCTION
    BBox(const BBox& other) {
        xmin = other.xmin;
        xmax = other.xmax;
        ymin = other.ymin;
        ymax = other.ymax;
        zmin = other.zmin;
        zmax = other.zmax;
    }

    KOKKOS_INLINE_FUNCTION
    void init() {
        xmin = ko::reduction_identity<Real>::min();
        xmax = ko::reduction_identity<Real>::max();
        ymin = ko::reduction_identity<Real>::min();
        ymax = ko::reduction_identity<Real>::max();
        zmin = ko::reduction_identity<Real>::min();
        zmax = ko::reduction_identity<Real>::max();
    }
    
    KOKKOS_INLINE_FUNCTION
    void operator = (const BBox& rhs) {
        xmin = rhs.xmin;
        xmax = rhs.xmax;
        ymin = rhs.ymin;
        ymax = rhs.ymax;
        zmin = rhs.zmin;
        zmax = rhs.zmax;
    }
    
    KOKKOS_INLINE_FUNCTION
    void operator = (const volatile BBox& rhs) {
        xmin = rhs.xmin;
        xmax = rhs.xmax;
        ymin = rhs.ymin;
        ymax = rhs.ymax;
        zmin = rhs.zmin;
        zmax = rhs.zmax;
    }
};

KOKKOS_INLINE_FUNCTION    
bool operator == (const BBox& lhs, const BBox& rhs) {
    return (lhs.xmin == rhs.xmin && 
            lhs.xmax == rhs.xmax && 
            lhs.ymin == rhs.ymin &&
            lhs.ymax == rhs.ymax &&
            lhs.zmin == rhs.zmin &&
            lhs.zmax == rhs.zmax);
}

KOKKOS_INLINE_FUNCTION    
bool operator != (const BBox& lhs, const BBox& rhs) {return !(lhs == rhs);}

struct BoxFunctor {
    typedef BBox value_type;
    ko::View<Real*[3]> pts;
    
    KOKKOS_INLINE_FUNCTION
    BoxFunctor(const ko::View<Real*[3]> p) : pts(p) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator () (const Index& i, value_type& bb) const {
        if (pts(i,0) < bb.xmin) bb.xmin = pts(i,0);
        if (pts(i,0) > bb.xmax) bb.xmax = pts(i,0);
        if (pts(i,1) < bb.ymin) bb.ymin = pts(i,1);
        if (pts(i,1) > bb.ymax) bb.ymax = pts(i,1);
        if (pts(i,2) < bb.zmin) bb.zmin = pts(i,2);
        if (pts(i,2) > bb.zmax) bb.zmax = pts(i,2);
    }

    KOKKOS_INLINE_FUNCTION
    void join(volatile value_type& dst, const volatile value_type& src) const {
        if (src.xmin < dst.xmin) dst.xmin = src.xmin;
        if (src.xmax > dst.xmax) dst.xmax = src.xmax;
        if (src.ymin < dst.ymin) dst.ymin = src.ymin;
        if (src.ymax > dst.ymax) dst.ymax = src.ymax;
        if (src.zmin < dst.zmin) dst.zmin = src.zmin;
        if (src.zmax > dst.zmax) dst.zmax = src.zmax;
    }
    
    KOKKOS_INLINE_FUNCTION
    void init(value_type& dst) const {dst.init();}
};

std::ostream& operator << (std::ostream& os, const BBox& b);

template <typename Space=Dev> 
struct BBoxReducer {
    public:
    typedef BBoxReducer reducer;
    typedef BBox value_type;
    typedef ko::View<value_type,Space> result_view_type;
    
    private:
        result_view_type value;
        bool references_scalar_v;
    
    public:
    KOKKOS_INLINE_FUNCTION
    BBoxReducer(value_type& val) : value(&val), references_scalar_v(true) {}
    
    KOKKOS_INLINE_FUNCTION
    BBoxReducer(const result_view_type& val) : value(val), references_scalar_v(false) {}
            
    KOKKOS_INLINE_FUNCTION
    void join(value_type& dest, const value_type& src) const {
        if (dest.xmin > src.xmin) dest.xmin = src.xmin;
        if (dest.xmax < src.xmax) dest.xmax = src.xmax;
        if (dest.ymin > src.ymin) dest.ymin = src.ymin;
        if (dest.ymax < src.ymax) dest.ymax = src.ymax;
        if (dest.zmin > src.zmin) dest.zmin = src.zmin;
        if (dest.zmax < src.zmax) dest.zmax = src.zmax;
    }
    
    KOKKOS_INLINE_FUNCTION
    void join(volatile value_type& dest, const volatile value_type& src) const {
        if (dest.xmin > src.xmin) dest.xmin = src.xmin;
        if (dest.xmax < src.xmax) dest.xmax = src.xmax;
        if (dest.ymin > src.ymin) dest.ymin = src.ymin;
        if (dest.ymax < src.ymax) dest.ymax = src.ymax;
        if (dest.zmin > src.zmin) dest.zmin = src.zmin;
        if (dest.zmax < src.zmax) dest.zmax = src.zmax;
    }
    
    KOKKOS_INLINE_FUNCTION
    void init(value_type& val) const {
        val.init();
    }
    
    KOKKOS_INLINE_FUNCTION
    result_view_type view() const {return value;}
    
    KOKKOS_INLINE_FUNCTION
    value_type& reference() const {return *value.data();}
    
    KOKKOS_INLINE_FUNCTION
    bool references_scalar() const {return references_scalar_v;}
};

KOKKOS_INLINE_FUNCTION
Real volume(const BBox& b) {
    const Real dx = b.xmax - b.xmin;
    const Real dy = b.ymax - b.ymin;
    const Real dz = b.zmax - b.zmin;
    return dx*dy*dz;
}

template <typename CPT> KOKKOS_INLINE_FUNCTION
bool boxContainsPoint(const BBox& b, const CPT& p) {
    const bool inx = (b.xmin <= p[0] && p[0] <= b.xmax);
    const bool iny = (b.ymin <= p[1] && p[1] <= b.ymax);
    const bool inz = (b.zmin <= p[2] && p[2] <= b.zmax);
    return ((inx && iny) && inz);
}

KOKKOS_INLINE_FUNCTION
Int child_index(const BBox& b, const Real pos[3]) {
    Int result = 0;
    if (boxContainsPoint(b, pos)) {
        if (pos[2] > 0.5*(b.zmin+b.zmax)) result += 1;
        if (pos[1] > 0.5*(b.ymin+b.ymax)) result += 2;
        if (pos[0] > 0.5*(b.xmin+b.xmax)) result += 4;
    }
    else {
        result = -1; // invalid value
    }
    return result;
}

KOKKOS_INLINE_FUNCTION
Real longestEdge(const BBox& b) {
    const Real dx = b.xmax - b.xmin;
    const Real dy = b.ymax - b.ymin;
    const Real dz = b.zmax - b.zmin;
    return max(max(dx,dy),dz);
}

KOKKOS_INLINE_FUNCTION
Real shortestEdge(const BBox& b) {
    const Real dx = b.xmax - b.xmin;
    const Real dy = b.ymax - b.ymin;
    const Real dz = b.zmax - b.zmin;
    return min(min(dx,dy),dz);
}

KOKKOS_INLINE_FUNCTION
Real boxAspectRatio(const BBox& b) {
    return longestEdge(b)/shortestEdge(b);
}

KOKKOS_INLINE_FUNCTION
void closestPointInBox(Real cp[3], const BBox& b, const Real querypt[3]) {
    if (boxContainsPoint(b, querypt)) {
        for (Int i=0; i<3; ++i) {
            cp[i] = querypt[i];
        }
    }
    else {
        Real ll[3];
        Real uu[3];
        ll[0] = b.xmin - querypt[0];
        ll[1] = b.ymin - querypt[1];
        ll[2] = b.zmin - querypt[2];
        uu[0] = b.xmax - querypt[0];
        uu[1] = b.ymax - querypt[1];
        uu[2] = b.zmax - querypt[2];
        for (Int i=0; i<3; ++i) {
            if (ll[i] == uu[i]) {
                cp[i] = ll[i];
            }
            else {
                if (ll[i] >= 0.0) {
                    cp[i] = ll[i];
                }
                else if (uu[i] < 0.0) {
                    cp[i] = uu[i];
                }
                else {
                    cp[i] = 0.0;
                }
            }
        }
        for (Int i=0; i<3; ++i) {
            cp[i] += querypt[i];
        }
    }
}

KOKKOS_INLINE_FUNCTION
void farthestPointInBox(Real fp[3], const BBox& b, const Real query[3]) {
    Real corners[8][3];
    for (short i=0; i<8; ++i) {
        corners[i][0] = (i%2 == 0 ? b.xmin : b.xmax);
        corners[i][1] = (((i>>1)&1) == 0 ? b.ymin : b.ymax);
        corners[i][2] = (i>>2 == 0 ? b.zmin : b.zmax);
    }
    Real dist = 0.0;
    Int c_ind = 0;
    for (Int i=0; i<8; ++i) {
        Real test_dist_sq = 0.0;
        for (Int j=0; j<3; ++j) {
            test_dist_sq += square(query[j] - corners[i][j]);
        }
        if (test_dist_sq > dist) {
            dist = test_dist_sq;
            c_ind = i;
        }
    }
    for (Int i=0; i<3; ++i) {
        fp[i] = corners[c_ind][i];
    }
}

KOKKOS_INLINE_FUNCTION
void boxCentroid(Real& cx, Real& cy, Real& cz, const BBox& bb) {
    cx = 0.5*(bb.xmin + bb.xmax);
    cy = 0.5*(bb.ymin + bb.ymax);
    cz = 0.5*(bb.zmin + bb.zmax);
}

KOKKOS_INLINE_FUNCTION
Real boxEdgeLength(const BBox& bb) {
    assert(std::abs(bb.boxAspectRatio(bb) - 1.0) < ZERO_TOL);
    return bb.xmax - bb.xmin;
}

KOKKOS_INLINE_FUNCTION
void box2cube(BBox& b) {
    if (std::abs(boxAspectRatio(b) - 1.0) > ZERO_TOL) {
        const Real half_len = 0.5*longestEdge(b);
        Real cx, cy, cz;
        boxCentroid(cx, cy, cz, b);
        b.xmin = cx - half_len;
        b.xmax = cx + half_len;
        b.ymin = cy - half_len;
        b.ymax = cy + half_len;
        b.zmin = cz - half_len;
        b.zmax = cz + half_len;
    }
}

KOKKOS_INLINE_FUNCTION
bool boxIntersectsSphere(const BBox& b, const Real sph_radius=1.0) {
    bool result = false;
    Real cp[3];
    Real fp[3];
    const Real origin[3] = {0.0,0.0,0.0};
    closestPointInBox(cp, b, origin);
    farthestPointInBox(fp, b, origin);
    Real cp_mag_sq = 0.0;
    Real fp_mag_sq = 0.0;
    for (Int i=0; i<3; ++i) {
        cp_mag_sq += square(cp[i]);
        fp_mag_sq += square(fp[i]);
    }
    const Real rsq = square(sph_radius);
    result = cp_mag_sq <= rsq && rsq <= fp_mag_sq;
    return result;
}

template <typename BoxView> KOKKOS_INLINE_FUNCTION
void bisectBoxAllDims(BoxView& kids, const BBox& parent) {
    const Real xmid = 0.5*(parent.xmin + parent.xmax);
    const Real ymid = 0.5*(parent.ymin + parent.ymax);
    const Real zmid = 0.5*(parent.zmin + parent.zmax);
    for (Int j=0; j<8; ++j) {
        kids(j).xmin = ((j&4) == 0 ? parent.xmin : xmid);
        kids(j).xmax = ((j&4) == 0 ? xmid : parent.xmax);
        kids(j).ymin = ((j&2) == 0 ? parent.ymin : ymid);
        kids(j).ymax = ((j&2) == 0 ? ymid : parent.ymax);
        kids(j).zmin = ((j&1) == 0 ? parent.zmin : zmid);
        kids(j).zmax = ((j&1) == 0 ? zmid : parent.zmax);
    }
}

}}
#endif