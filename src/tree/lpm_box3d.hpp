#ifndef LPM_BOX3D_HPP
#define LPM_BOX3D_HPP

#include "LpmConfig.h"

#include "lpm_kokkos_defs.hpp"
#include "util/lpm_math.hpp"
#include "util/lpm_tuple.hpp"
#include "util/lpm_floating_point.hpp"

#include "Kokkos_Core.hpp"

#include <iostream>
#include <iomanip>

namespace Lpm {
namespace tree {

#define BOX_PADDING 1e-5


/** @brief A simple struct representing a box (rectangular prism) in 3d Euclidean space.

*/
struct Box3d {
    Real xmin, xmax;
    Real ymin, ymax;
    Real zmin, zmax;

    KOKKOS_INLINE_FUNCTION
    Box3d() {init();}

    KOKKOS_INLINE_FUNCTION
    Box3d(const Real& xl, const Real& xr,
          const Real& yl, const Real& yr,
          const Real& zl, const Real& zr,
          const bool pad = true) :
        xmin(xl - (pad ? BOX_PADDING : 0)),
        xmax(xr + (pad ? BOX_PADDING : 0)),
        ymin(yl - (pad ? BOX_PADDING : 0)),
        ymax(yr + (pad ? BOX_PADDING : 0)),
        zmin(zl - (pad ? BOX_PADDING : 0)),
        zmax(zr + (pad ? BOX_PADDING : 0)) {}

    KOKKOS_INLINE_FUNCTION
    Box3d(const ko::Tuple<Real,6>& mm) :
      xmin(mm[0]),
      xmax(mm[1]),
      ymin(mm[2]),
      ymax(mm[3]),
      zmin(mm[4]),
      zmax(mm[5])
    {}

    KOKKOS_INLINE_FUNCTION
    Box3d(const Box3d& other) :
        xmin(other.xmin),
        xmax(other.xmax),
        ymin(other.ymin),
        ymax(other.ymax),
        zmin(other.zmin),
        zmax(other.zmax) {}

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
    void operator = (const Box3d& rhs) {
        xmin = rhs.xmin;
        xmax = rhs.xmax;
        ymin = rhs.ymin;
        ymax = rhs.ymax;
        zmin = rhs.zmin;
        zmax = rhs.zmax;
    }

    KOKKOS_INLINE_FUNCTION
    void operator = (const volatile Box3d& rhs) {
        xmin = rhs.xmin;
        xmax = rhs.xmax;
        ymin = rhs.ymin;
        ymax = rhs.ymax;
        zmin = rhs.zmin;
        zmax = rhs.zmax;
    }

    KOKKOS_INLINE_FUNCTION
    Real volume() const {
      const Real dx = xmax - xmin;
      const Real dy = ymax - ymin;
      const Real dz = zmax - zmin;
      return dx*dy*dz;
    }

    template <typename CPT> KOKKOS_INLINE_FUNCTION
    bool contains_pt(const CPT& p) const {
        const bool inx = (xmin <= p[0] && p[0] <= xmax);
        const bool iny = (ymin <= p[1] && p[1] <= ymax);
        const bool inz = (zmin <= p[2] && p[2] <= zmax);
        return ((inx && iny) && inz);
    }

    KOKKOS_INLINE_FUNCTION
    Int octree_child_index(const Real pos[3]) const {
        Int result = 0;
        LPM_KERNEL_ASSERT(contains_pt(pos));

        if (pos[2] > 0.5*(zmin+zmax)) result += 1;
        if (pos[1] > 0.5*(ymin+ymax)) result += 2;
        if (pos[0] > 0.5*(xmin+xmax)) result += 4;

        return result;
    }

    KOKKOS_INLINE_FUNCTION
    Real longest_edge() const {
        const Real dx = xmax - xmin;
        const Real dy = ymax - ymin;
        const Real dz = zmax - zmin;
        return max(max(dx,dy),dz);
    }

    KOKKOS_INLINE_FUNCTION
    Real shortest_edge() const {
        const Real dx = xmax - xmin;
        const Real dy = ymax - ymin;
        const Real dz = zmax - zmin;
        return min(min(dx,dy),dz);
    }

    KOKKOS_INLINE_FUNCTION
    Real aspect_ratio() const {
        return longest_edge() / shortest_edge();
    }

    KOKKOS_INLINE_FUNCTION
    void centroid(Real& cx, Real& cy, Real& cz) const {
        cx = 0.5*(xmin + xmax);
        cy = 0.5*(ymin + ymax);
        cz = 0.5*(zmin + zmax);
    }

    KOKKOS_INLINE_FUNCTION
    ko::Tuple<Real,3> centroid() const {
      ko::Tuple<Real,3> result;
      result[0] = 0.5*(xmin+xmax);
      result[1] = 0.5*(ymin+ymax);
      result[2] = 0.5*(zmin+zmax);
      return result;
    }

    KOKKOS_INLINE_FUNCTION
    bool is_cube() const {
      return FloatingPoint<Real>::equiv(aspect_ratio(), 1.0);
    }

    KOKKOS_INLINE_FUNCTION
    Real cube_edge_length() const {
        LPM_KERNEL_ASSERT(is_cube());
        return xmax - xmin;
    }

    KOKKOS_INLINE_FUNCTION
    void make_cube() {
      if (!is_cube()) {
        const Real half_len = 0.5*longest_edge();
        const auto c = centroid();
        xmin = c[0] - half_len;
        xmax = c[0] + half_len;
        ymin = c[1] - half_len;
        ymax = c[1] + half_len;
        zmin = c[2] - half_len;
        zmax = c[2] + half_len;
      }
    }

    template <typename PtType> KOKKOS_INLINE_FUNCTION
    Int pt_in_neighborhood(const PtType& p) const {
      Int result = -1;
      const bool inx = (xmin <= p[0] && p[0] <= xmax);
      const bool iny = (ymin <= p[1] && p[1] <= ymax);
      const bool inz = (zmin <= p[2] && p[2] <= zmax);
      if (p[0] < xmin) {
        if (p[1] < ymin) {
          if (p[2] < zmin) {
            result = 0;
          }
          else if (p[2] <= zmax) {
            LPM_KERNEL_ASSERT(inz);
            result = 1;
          }
          else {
            result = 2;
          }
        }
        else if (p[1] <= ymax) {
          LPM_KERNEL_ASSERT(iny);
          if (p[2] < zmin) {
            result = 3;
          }
          else if (p[2] <= zmax) {
            LPM_KERNEL_ASSERT(inz);
            result = 4;
          }
          else {
            result = 5;
          }
        }
        else {
          if (p[2] < zmin) {
            result = 6;
          }
          else if (p[2] <= zmax) {
            LPM_KERNEL_ASSERT(inz);
            result = 7;
          }
          else {
            result = 8;
          }
        }
      }
      else if (p[0] <= xmax) {
        LPM_KERNEL_ASSERT(inx);
        if (p[1] < ymin) {
          if (p[2] < zmin) {
            result = 9;
          }
          else if (p[2] <= zmax) {
            LPM_KERNEL_ASSERT(inz);
            result = 10;
          }
          else {
            result = 11;
          }
        }
        else if (p[1] <= ymax) {
          LPM_KERNEL_ASSERT(iny);
          if (p[2] < zmin) {
            result = 12;
          }
          else if (p[2] <= zmax) {
            LPM_KERNEL_ASSERT(inz);
            LPM_KERNEL_ASSERT(contains_pt(p));
            result = 13;
          }
          else {
            result = 14;
          }
        }
        else {
          if (p[2] < zmin) {
            result = 15;
          }
          else if (p[2] <= zmax) {
            LPM_KERNEL_ASSERT(inz);
            result = 16;
          }
          else {
            result = 17;
          }
        }
      }
      else {
        if (p[1] < ymin) {
          if (p[2] < zmin) {
            result = 18;
          }
          else if (p[2] <= zmax) {
            LPM_KERNEL_ASSERT(inz);
            result = 19;
          }
          else {
            result = 20;
          }
        }
        else if (p[1] <= ymax) {
          LPM_KERNEL_ASSERT(iny);
          if (p[2] < zmin) {
            result = 21;
          }
          else if (p[2] <= zmax) {
            LPM_KERNEL_ASSERT(inz);
            result = 22;
          }
          else {
            result = 23;
          }
        }
        else {
          if (p[2] < zmin) {
            result = 24;
          }
          else if (p[2] <= zmax) {
            LPM_KERNEL_ASSERT(inz);
            result = 25;
          }
          else {
            result = 26;
          }
        }
      }
      return result;
    }

    template <typename PtType> KOKKOS_INLINE_FUNCTION
    ko::Tuple<Real,3> closest_pt(const PtType& p) const {
      ko::Tuple<Real,3> result;
      for (int i=0; i<3; ++i) {
        result[i] = p[i];
      }
      const int nbrh = pt_in_neighborhood(p);
      if (nbrh%3 == 0) result[2] = zmin;
      if ((nbrh-2)%3 == 0) result[2] = zmax;
      if (nbrh <= 8) result[0] = xmin;
      if (nbrh >= 18) result[0] = xmax;
      return result;
    }

    std::vector<Box3d> bisect_all() const;
};

KOKKOS_INLINE_FUNCTION
bool operator == (const Box3d& lhs, const Box3d& rhs) {
    return (FloatingPoint<Real>::equiv(lhs.xmin, rhs.xmin) &&
            FloatingPoint<Real>::equiv(lhs.xmax, rhs.xmax) &&
            FloatingPoint<Real>::equiv(lhs.ymin, rhs.ymin) &&
            FloatingPoint<Real>::equiv(lhs.ymax, rhs.ymax) &&
            FloatingPoint<Real>::equiv(lhs.zmin, rhs.zmin) &&
            FloatingPoint<Real>::equiv(lhs.zmax, rhs.zmax));
}

KOKKOS_INLINE_FUNCTION
bool operator != (const Box3d& lhs, const Box3d& rhs) {return !(lhs == rhs);}

struct BoundingBoxFunctor {
    typedef Box3d value_type;
    ko::View<Real*[3]> pts;

    KOKKOS_INLINE_FUNCTION
    BoundingBoxFunctor(const ko::View<Real*[3]> p) : pts(p) {}

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

std::ostream& operator << (std::ostream& os, const Box3d& b);

template <typename Space=Dev>
struct Box3dReducer {
    public:
    typedef Box3dReducer reducer;
    typedef Box3d value_type;
    typedef ko::View<value_type,Space> result_view_type;

    private:
        result_view_type value;
        bool references_scalar_v;

    public:
    KOKKOS_INLINE_FUNCTION
    Box3dReducer(value_type& val) : value(&val), references_scalar_v(true) {}

    KOKKOS_INLINE_FUNCTION
    Box3dReducer(const result_view_type& val) : value(val), references_scalar_v(false) {}

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


// KOKKOS_INLINE_FUNCTION
// bool boxIntersectsSphere(const Box3d& b, const Real sph_radius=1.0) {
//     bool result = false;
//     Real cp[3];
//     Real fp[3];
//     const Real origin[3] = {0.0,0.0,0.0};
//     closestPointInBox(cp, b, origin);
//     farthestPointInBox(fp, b, origin);
//     Real cp_mag_sq = 0.0;
//     Real fp_mag_sq = 0.0;
//     for (Int i=0; i<3; ++i) {
//         cp_mag_sq += square(cp[i]);
//         fp_mag_sq += square(fp[i]);
//     }
//     const Real rsq = square(sph_radius);
//     result = cp_mag_sq <= rsq && rsq <= fp_mag_sq;
//     return result;
// }

template <typename BoxView> KOKKOS_INLINE_FUNCTION
void box_bisect_all(BoxView& kids, const Box3d& parent) {
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
