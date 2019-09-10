#ifndef LPM_BOX3D_HPP
#define LPM_BOX3D_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmKokkosUtil.hpp"
#include "Kokkos_Core.hpp"
#include "LpmUtilities.hpp"

namespace Lpm {

// A Box is an array of 6 Reals, (xmin, xmax, ymin, ymax, zmin, zmax)

static constexpr Real BOX_PAD_FACTOR = 1.01;

template <typename CBoxType> KOKKOS_INLINE_FUNCTION
Real volume(const CBoxType& b) {
    const Real dx = b(1) - b(0);
    const Real dy = b(3) - b(2);
    const Real dz = b(5) - b(4);
    return dx*dy*dz;
}

template <typename CBoxType, typename CPtType> KOKKOS_INLINE_FUNCTION
Int child_index(const CBoxType& b, const CPtType& pos) {
    Int result = 0;
    if (boxContainsPoint(b, pos)) {
        if (pos(0) > 0.5*(b(0)+b(1))) result += 1;
        if (pos(1) > 0.5*(b(2)+b(3))) result += 2;
        if (pos(3) > 0.5*(b(4)+b(5))) result += 4;
    }
    else {
        result = -1; // invalid value
    }
    return result;
}

template <typename CBoxType, typename CPtType> KOKKOS_INLINE_FUNCTION
bool boxContainsPoint(const CBoxType& b, const CPtType& p) {
    const bool inx = (b(0) <= p(0) && p(0) < b(1));
    const bool iny = (b(2) <= p(1) && p(1) < b(3));
    const bool inz = (b(4) <= p(2) && p(2) < b(5));
    return ((inx && iny) && inz);
}

template <typename CBoxType> KOKKOS_INLINE_FUNCTION
Real longestEdge(const CBoxType& b) {
    const Real dx = b(1) - b(0);
    const Real dy = b(3) - b(2);
    const Real dz = b(5) - b(4);
    return max(max(dx,dy),dz);
}

template <typename CBoxType> KOKKOS_INLINE_FUNCTION
Real shortestEdge(const CBoxType& b) {
    const Real dx = b(1) - b(0);
    const Real dy = b(3) - b(2);
    const Real dz = b(5) - b(4);
    return min(min(dx,dy),dz);
}

template <typename CBoxType> KOKKOS_INLINE_FUNCTION
Real boxAspectRatio(const CBoxType& b) {
    return longestEdge(b)/shortestEdge(b);
}

template <typename CBoxType, typename CPtType, typename PtType> KOKKOS_INLINE_FUNCTION
void closestPointInBox(const CBoxType& b, const CPtType& querypt, PtType& cp) {
    if (boxContainsPoint(b, querypt)) {
        for (Int i=0; i<3; ++i) {
            cp(i) = querypt(i);
        }
    }
    else {
        Real ll[3];
        Real uu[3];
        for (Int i=0; i<3; ++i) {
            ll[i] = b(2*i) - querypt(i);
            uu[i] = b(2*i+1) - querypt(i);
        }
        for (Int i=0; i<3; ++i) {
            if (ll[i] == uu[i]) {
                cp(i) = ll[i];
            }
            else {
                if (ll[i] >= 0.0) {
                    cp(i) = ll[i];
                }
                else if (uu[i] < 0.0) {
                    cp(i) = uu[i];
                }
                else {
                    cp(i) = 0.0;
                }
            }
        }
        for (Int i=0; i<3; ++i) {
            cp(i) += querypt(i);
        }
    }
}

template <typename CBoxType> KOKKOS_INLINE_FUNCTION
bool boxIntersectsSphere(const CBoxType& b, const Real sph_radius=1.0) {
    bool result = false;
    Real cp[3];
    Real fp[3];
    const Real origin[3] = {0.0,0.0,0.0};
    closestPointInBox(b, origin, cp);
    farthestPointInBox(b, origin, fp);
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

template <typename CBoxType, typename CPtType, typename PtType> KOKKOS_INLINE_FUNCTION
void farthestPointInBox(const CBoxType& b, const CPtType& query, PtType& fp) {
    Real corners[8][3];
    const Real xmin = b(0);
    const Real xmax = b(1);
    const Real ymin = b(2);
    const Real ymax = b(3);
    const Real zmin = b(4);
    const Real zmax = b(5);
    for (short i=0; i<8; ++i) {
        corners[i][0] = (i%2 == 0 ? xmin : xmax);
        corners[i][1] = (((i>>1)&1) == 0 ? ymin : ymax);
        corners[i][2] = (i>>2 == 0 ? zmin : zmax);
    }
    Real dist = 0.0;
    Int c_ind = 0;
    for (Int i=0; i<8; ++i) {
        Real test_dist_sq = 0.0;
        for (Int j=0; j<3; ++j) {
            test_dist_sq += square(query(j) - corners[i][j]);
        }
        if (test_dist_sq > dist) {
            dist = test_dist_sq;
            c_ind = i;
        }
    }
    for (Int i=0; i<3; ++i) {
        fp(i) = corners[c_ind][i];
    }
}

template <typename CBoxType, typename BoxView> KOKKOS_INLINE_FUNCTION
void bisectBoxAllDims(const CBoxType parent, BoxView& kids) {
    const Real xmin = parent(0);
    const Real xmax = parent(1);
    const Real ymin = parent(2);
    const Real ymax = parent(3);
    const Real zmin = parent(4);
    const Real zmax = parent(5);
    const Real xmid = 0.5*(xmin + xmax);
    const Real ymid = 0.5*(ymin + ymax);
    const Real zmid = 0.5*(zmin + zmax);
    for (Int j=0; j<8; ++j) {
        kids(j,0) = (j%2 == 0 ? xmin : xmid);
        kids(j,1) = (j%2 == 0 ? xmid : xmax);
        kids(j,2) = (((j>>1)&1) == 0 ? ymin : ymid);
        kids(j,3) = (((j>>1)&1) == 0 ? ymid : ymax);
        kids(j,4) = (j>>2 == 0 ? zmin : zmid);
        kids(j,5) = (j>>2 == 0 ? zmid : zmax);
    }
}

}
#endif