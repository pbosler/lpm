#ifndef LPM_BOX2D_HPP
#define LPM_BOX2D_HPP

#include "LpmConfig.h"
#include "lpm_kokkos_defs.hpp"
#include "tree/lpm_tree_defs.hpp"
#include "util/lpm_math.hpp"
#include "util/lpm_tuple.hpp"
#include "util/lpm_floating_point.hpp"

#include <iostream>
#include <iomanip>

namespace Lpm {

struct Box2d {
  Real xmin, xmax;
  Real ymin, ymax;

  KOKKOS_INLINE_FUNCTION
  Box2d() {init();}

  KOKKOS_INLINE_FUNCTION
  void init() {
    xmin = Kokkos::reduction_identity<Real>::min();
    xmax = Kokkos::reduction_identity<Real>::max();
    ymin = Kokkos::reduction_identity<Real>::min();
    ymax = Kokkos::reduction_identity<Real>::max();
  }

  /** @brief constructor

    @param [in] xl left boundary in x
    @param [in] xr right boundary in x
    @param [in] yl left boundary in xy
    @param [in] yr right boundary in y
    @param [in] pad if true, add padding in all directions
  */
  KOKKOS_INLINE_FUNCTION
  Box2d(const Real& xl, const Real& xr,
        const Real& yl, const Real& yr,
        const bool pad = true) :
      xmin(xl - (pad ? BOX_PADDING : 0)),
      xmax(xr + (pad ? BOX_PADDING : 0)),
      ymin(yl - (pad ? BOX_PADDING : 0)),
      ymax(yr + (pad ? BOX_PADDING : 0)) {}

  /** @brief Constuct a square from centroid and side length
  */
  KOKKOS_INLINE_FUNCTION
  Box2d(const Kokkos::Tuple<Real,2>& c, const Real len) :
    xmin(c[0] - 0.5*len),
    xmax(c[0] + 0.5*len),
    ymin(c[1] - 0.5*len),
    ymax(c[1] + 0.5*len) {}


  /// @brief assignment operator
  KOKKOS_INLINE_FUNCTION
  void operator = (const Box2d& rhs) {
      xmin = rhs.xmin;
      xmax = rhs.xmax;
      ymin = rhs.ymin;
      ymax = rhs.ymax;
  }

  /// @brief assignment operator
  KOKKOS_INLINE_FUNCTION
  void operator = (const volatile Box2d& rhs) {
      xmin = rhs.xmin;
      xmax = rhs.xmax;
      ymin = rhs.ymin;
      ymax = rhs.ymax;
  }

  /// @brief compute & return area
  KOKKOS_INLINE_FUNCTION
  Real area() const {
    const Real dx = xmax - xmin;
    const Real dy = ymax - ymin;
    return dx*dy;
  }

  /** @brief given a query point, returns true if the point lies within
    the box, false otherwise.
  */
  template <typename CPT> KOKKOS_INLINE_FUNCTION
  bool contains_pt(const CPT& p) const {
      const bool inx = (xmin <= p[0] && p[0] <= xmax);
      const bool iny = (ymin <= p[1] && p[1] <= ymax);
      return (inx && iny);
  }

  template <typename CPT> KOKKOS_INLINE_FUNCTION
  Int quadtree_child_idx(const CPT pos) const {
    LPM_KERNEL_ASSERT(contains_pt(pos));
    Int result = 0;
    if (pos[1] > 0.5*(ymin + ymax)) result += 1;
    if (pos[0] > 0.5*(xmin + xmax)) result += 2;
    return result;
  }

  /// return the longest edge of a box
  KOKKOS_INLINE_FUNCTION
  Real longest_edge() const {
      const Real dx = xmax - xmin;
      const Real dy = ymax - ymin;
      return max(dx,dy);
  }

  /// return the shortest edge of a box
  KOKKOS_INLINE_FUNCTION
  Real shortest_edge() const {
      const Real dx = xmax - xmin;
      const Real dy = ymax - ymin;
      return min(dx,dy);
  }

  /// compute the aspect ratio
  KOKKOS_INLINE_FUNCTION
  Real aspect_ratio() const {
      return longest_edge() / shortest_edge();
  }

  /// compute & return centroid
  KOKKOS_INLINE_FUNCTION
  void centroid(Real& cx, Real& cy) const {
    cx = 0.5*(xmin + xmax);
    cy = 0.5*(ymin + ymax);
  }

  /// compute & return centroid
  KOKKOS_INLINE_FUNCTION
  Kokkos::Tuple<Real,2> centroid() const {
    Kokkos::Tuple<Real,2> result;
    result[0] = 0.5*(xmin+xmax);
    result[1] = 0.5*(ymin+ymax);
    LPM_KERNEL_ASSERT(contains_pt(result));
    return result;
  }

  /// returns true if the box is a cube, false otherwise
  KOKKOS_INLINE_FUNCTION
  bool is_square() const {
    return FloatingPoint<Real>::equiv(aspect_ratio(), 1.0);
  }

  /// returns the edge length of a cubic box
  KOKKOS_INLINE_FUNCTION
  Real square_edge_length() const {
    LPM_KERNEL_REQUIRE(is_square());
    return xmax - xmin;
  }

  /// re-compute box region to require the box to be a cube
  KOKKOS_INLINE_FUNCTION
  void make_square() {
    if (!is_square()) {
      const Real half_len = 0.5*longest_edge();
      const auto c = centroid();
      xmin = c[0] - half_len;
      xmax = c[0] + half_len;
      ymin = c[1] - half_len;
      ymax = c[1] + half_len;
    }
  }

  template <typename PtType> KOKKOS_INLINE_FUNCTION
  Int pt_in_neighborhood(const PtType& p) const {
    const short xreg = (p[0] < xmin ? 0 : (p[0] < xmax ? 1 : 2));
    const short yreg = (p[1] < ymin ? 0 : (p[1] < ymax ? 1 : 2));
    return 3*xreg + yreg;
  }

  template <typename PtType> KOKKOS_INLINE_FUNCTION
  Kokkos::Tuple<Real,2> closest_pt_l1(const PtType& p) const {
    Kokkos::Tuple<Real,2> result;
    const auto nbrh = pt_in_neighborhood(p);
    const auto yreg = nbrh%3;
    const auto xreg = nbrh/3;
    result[0] = (xreg == 0 ? xmin : (xreg == 1 ? p[0] : xmax));
    result[1] = (yreg == 0 ? ymin : (yreg == 1 ? p[1] : ymax));
    return result;
  }

  KOKKOS_INLINE_FUNCTION
  Kokkos::Tuple<Real,2> vertex_crds(const int v) const {
    LPM_KERNEL_ASSERT( v>=0 and v<4 );
    Kokkos::Tuple<Real,2> result;
    result[0] = ((v&2) == 0 ? xmin : xmax);
    result[1] = ((v&1) == 0 ? ymin : ymax);
    return result;
  }

  std::vector<Box2d> bisect_all() const;

  std::vector<Box2d> neighbors() const;
};

KOKKOS_INLINE_FUNCTION
Box2d default_box() {
  return Box2d(-1,1, -1,1);
}

KOKKOS_INLINE_FUNCTION
bool operator == (const Box2d& lhs, const Box2d& rhs) {
    return (FloatingPoint<Real>::equiv(lhs.xmin, rhs.xmin) &&
            FloatingPoint<Real>::equiv(lhs.xmax, rhs.xmax) &&
            FloatingPoint<Real>::equiv(lhs.ymin, rhs.ymin) &&
            FloatingPoint<Real>::equiv(lhs.ymax, rhs.ymax));
}

KOKKOS_INLINE_FUNCTION
bool operator != (const Box2d& lhs, const Box2d& rhs) {return !(lhs == rhs);}

std::ostream& operator << (std::ostream& os, const Box2d& b);

Box2d parent_from_child(const Box2d& kid, const Int kid_idx);

struct BoundingBoxFunctor {
  Kokkos::View<Real*[2]> pts;

  KOKKOS_INLINE_FUNCTION
  BoundingBoxFunctor(const Kokkos::View<Real*[2]> p) : pts(p) {}


  KOKKOS_INLINE_FUNCTION
  void operator() (const int i, Box2d& bb) const {
    if (pts(i,0) < bb.xmin) bb.xmin = pts(i,0);
    if (pts(i,0) > bb.xmax) bb.xmax = pts(i,0);
    if (pts(i,1) < bb.ymin) bb.ymin = pts(i,1);
    if (pts(i,1) > bb.ymax) bb.ymax = pts(i,1);
  }
};

template <typename Space=Dev>
struct Box2dReducer {
  typedef Box2dReducer reducer;
  typedef Box2d value_type;
  typedef Kokkos::View<Box2d,Space> result_view_type;

  KOKKOS_INLINE_FUNCTION
  Box2dReducer(value_type& v) : value(&v), references_scalar_v(true) {}

  KOKKOS_INLINE_FUNCTION
  Box2dReducer(const result_view_type& val) : value(val), references_scalar_v(false) {}

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dst, const value_type& src) const {
    if (src.xmin < dst.xmin) dst.xmin = src.xmin;
    if (src.xmax > dst.xmax) dst.xmax = src.xmax;
    if (src.ymin < dst.ymin) dst.ymin = src.ymin;
    if (src.ymax > dst.ymax) dst.ymax = src.ymax;
  }

  KOKKOS_INLINE_FUNCTION
  void join(volatile value_type& dst, const volatile value_type& src) const {
    if (src.xmin < dst.xmin) dst.xmin = src.xmin;
    if (src.xmax > dst.xmax) dst.xmax = src.xmax;
    if (src.ymin < dst.ymin) dst.ymin = src.ymin;
    if (src.ymax > dst.ymax) dst.ymax = src.ymax;
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

  private:
    result_view_type value;
    bool references_scalar_v;
};


} // namespace Lpm

#endif

