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
    /// min/max coordinates in each dimension
    Real xmin, xmax;
    Real ymin, ymax;
    Real zmin, zmax;

    KOKKOS_INLINE_FUNCTION
    Box3d() {init();}

    /** @brief constructor

      @param [in] xl left boundary in x
      @param [in] xr right boundary in x
      @param [in] yl left boundary in xy
      @param [in] yr right boundary in y
      @param [in] zl left boundary in z
      @param [in] zr right boundary in z
      @param [in] pad if true, add padding in all directions
    */
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

    /** @brief constructor

      @param [in] mm min/max values in each direction,
        ordered as min/max in x, then y, then z
    */
    KOKKOS_INLINE_FUNCTION
    Box3d(const ko::Tuple<Real,6>& mm) :
      xmin(mm[0]),
      xmax(mm[1]),
      ymin(mm[2]),
      ymax(mm[3]),
      zmin(mm[4]),
      zmax(mm[5])
    {}

    /** @brief Constuct a cube from centroid and side length
    */
    KOKKOS_INLINE_FUNCTION
    Box3d(const ko::Tuple<Real,3>& c, const Real len) :
      xmin(c[0] - 0.5*len),
      xmax(c[0] + 0.5*len),
      ymin(c[1] - 0.5*len),
      ymax(c[1] + 0.5*len),
      zmin(c[2] - 0.5*len),
      zmax(c[2] + 0.5*len) {}

    /// copy constructor
    KOKKOS_INLINE_FUNCTION
    Box3d(const Box3d& other) :
        xmin(other.xmin),
        xmax(other.xmax),
        ymin(other.ymin),
        ymax(other.ymax),
        zmin(other.zmin),
        zmax(other.zmax) {}

    /** @brief default initializer

      sets up box for use with custom reducer to determine bounding box
    */
    KOKKOS_INLINE_FUNCTION
    void init() {
        xmin = ko::reduction_identity<Real>::min();
        xmax = ko::reduction_identity<Real>::max();
        ymin = ko::reduction_identity<Real>::min();
        ymax = ko::reduction_identity<Real>::max();
        zmin = ko::reduction_identity<Real>::min();
        zmax = ko::reduction_identity<Real>::max();
    }

    /// @brief assignment operator
    KOKKOS_INLINE_FUNCTION
    void operator = (const Box3d& rhs) {
        xmin = rhs.xmin;
        xmax = rhs.xmax;
        ymin = rhs.ymin;
        ymax = rhs.ymax;
        zmin = rhs.zmin;
        zmax = rhs.zmax;
    }

    /// @brief assignment operator
    KOKKOS_INLINE_FUNCTION
    void operator = (const volatile Box3d& rhs) {
        xmin = rhs.xmin;
        xmax = rhs.xmax;
        ymin = rhs.ymin;
        ymax = rhs.ymax;
        zmin = rhs.zmin;
        zmax = rhs.zmax;
    }

    /// @brief compute & return volume
    KOKKOS_INLINE_FUNCTION
    Real volume() const {
      const Real dx = xmax - xmin;
      const Real dy = ymax - ymin;
      const Real dz = zmax - zmin;
      return dx*dy*dz;
    }

    /** @brief given a query point, returns true if the point lies within
      the box, false otherwise.
    */
    template <typename CPT> KOKKOS_INLINE_FUNCTION
    bool contains_pt(const CPT& p) const {
        const bool inx = (xmin <= p[0] && p[0] <= xmax);
        const bool iny = (ymin <= p[1] && p[1] <= ymax);
        const bool inz = (zmin <= p[2] && p[2] <= zmax);
        return ((inx && iny) && inz);
    }

    /** @brief given a point contained by *this,
      return the index of the child that also contains the point.
    */
    template <typename CPT> KOKKOS_INLINE_FUNCTION
    Int octree_child_idx(const CPT pos) const {
        Int result = 0;
        LPM_KERNEL_ASSERT(contains_pt(pos));

        if (pos[2] > 0.5*(zmin+zmax)) result += 1;
        if (pos[1] > 0.5*(ymin+ymax)) result += 2;
        if (pos[0] > 0.5*(xmin+xmax)) result += 4;

        return result;
    }

    /// return the longest edge of a box
    KOKKOS_INLINE_FUNCTION
    Real longest_edge() const {
        const Real dx = xmax - xmin;
        const Real dy = ymax - ymin;
        const Real dz = zmax - zmin;
        return max(max(dx,dy),dz);
    }

    /// return the shortest edge of a box
    KOKKOS_INLINE_FUNCTION
    Real shortest_edge() const {
        const Real dx = xmax - xmin;
        const Real dy = ymax - ymin;
        const Real dz = zmax - zmin;
        return min(min(dx,dy),dz);
    }

    /// compute the aspect ratio
    KOKKOS_INLINE_FUNCTION
    Real aspect_ratio() const {
        return longest_edge() / shortest_edge();
    }

    /// compute & return centroid
    KOKKOS_INLINE_FUNCTION
    void centroid(Real& cx, Real& cy, Real& cz) const {
        cx = 0.5*(xmin + xmax);
        cy = 0.5*(ymin + ymax);
        cz = 0.5*(zmin + zmax);
    }

    /// compute & return centroid
    KOKKOS_INLINE_FUNCTION
    ko::Tuple<Real,3> centroid() const {
      ko::Tuple<Real,3> result;
      result[0] = 0.5*(xmin+xmax);
      result[1] = 0.5*(ymin+ymax);
      result[2] = 0.5*(zmin+zmax);
      LPM_KERNEL_ASSERT(contains_pt(result));
      return result;
    }

    /// returns true if the box is a cube, false otherwise
    KOKKOS_INLINE_FUNCTION
    bool is_cube() const {
      return FloatingPoint<Real>::equiv(aspect_ratio(), 1.0);
    }

    /// returns the edge length of a cubic box
    KOKKOS_INLINE_FUNCTION
    Real cube_edge_length() const {
        LPM_KERNEL_REQUIRE(is_cube());
        return xmax - xmin;
    }

    /// re-compute box region to require the box to be a cube
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

    /** @brief given a query point, return the box-relative neighborhood that contains it.
    */
    template <typename PtType> KOKKOS_INLINE_FUNCTION
    Int pt_in_neighborhood(const PtType& p) const {
      Int result = -1;
      const short xreg = (p[0] < xmin ? 0 : (p[0] < xmax ? 1 : 2));
      const short yreg = (p[1] < ymin ? 0 : (p[1] < ymax ? 1 : 2));
      const short zreg = (p[2] < zmin ? 0 : (p[2] < zmax ? 1 : 2));
      result = 9*xreg + 3*yreg + zreg;
      return result;
    }

    /** @brief function returns the closest point inside a box relative to
      a query point using the L1 norm (not Euclidean distance).

      If the box contains the point, the point itself is returned.

      @param [in] p query point
      @return c such that @f$ c = \arg\min_{p^*\in B} \Vert p^*-p\Vert_1@f$
    */
    template <typename PtType> KOKKOS_INLINE_FUNCTION
    ko::Tuple<Real,3> closest_pt_l1(const PtType& p) const {
      ko::Tuple<Real,3> result;
      const auto nbrh = pt_in_neighborhood(p);
      const auto zreg = nbrh%3;
      const auto yreg = (nbrh/3)%3;
      const auto xreg = (nbrh/9)%3;
      result[2] = (zreg == 0 ? zmin : (zreg == 1 ? p[2] : zmax));
      result[1] = (yreg == 0 ? ymin : (yreg == 1 ? p[1] : ymax));
      result[0] = (xreg == 0 ? xmin : (xreg == 1 ? p[0] : xmax));
      return result;
    }

    /** @brief return the vertex coordinates of a box
    */
    template <typename PtType> KOKKOS_INLINE_FUNCTION
    void vertex_crds(PtType& crds, const int v) const {
      LPM_KERNEL_ASSERT( (v>=0 and v<8) );
      crds[0] = ( (v&4) == 0 ? xmin : xmax );
      crds[1] = ( (v&2) == 0 ? ymin : ymax );
      crds[2] = ( (v&1) == 0 ? zmin : zmax );
    }

    /** @brief generate child boxes by bisection in every dimension.
    */
    std::vector<Box3d> bisect_all() const;

    /** @brief generate all neighbor boxes
    */
    std::vector<Box3d> neighbors() const;
};


/** @brief return the default box spanning [-1,1] (with padding) in each
  direction.
*/
KOKKOS_INLINE_FUNCTION
Box3d std_box() {
  return Box3d(-1,1,-1,1,-1,1);
}

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

std::ostream& operator << (std::ostream& os, const Box3d& b);


/** @brief custom reducer to determine the smallest box that contains
  every point in a given point set.
*/
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

    /// default initializer for use with custom reducer
    KOKKOS_INLINE_FUNCTION
    void init(value_type& val) const {
        val.init();
    }

    /// return the object of a single-member view
    KOKKOS_INLINE_FUNCTION
    result_view_type view() const {return value;}

    /// return the reference to a view's data
    KOKKOS_INLINE_FUNCTION
    value_type& reference() const {return *value.data();}

    /// true if *this references its own instance (false if it references a view)
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

/** @brief given a child box, whose child index is known, build the parent box.
*/
Box3d parent_from_child(const Box3d& child, const Int child_idx);

/// Generate a box's children by bisection in every coordinate dimension.
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
