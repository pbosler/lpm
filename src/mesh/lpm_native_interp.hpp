#ifndef LPM_NATIVE_INTERP_HPP
#define LPM_NATIVE_INTERP_HPP

#include "LpmConfig.h"
#include "lpm_assert.hpp"
#include "lpm_constants.hpp"
#include "lpm_geometry.hpp"
#include "lpm_kokkos_defs.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#include "mesh/lpm_polymesh2d_functions.hpp"
#include "util/lpm_floating_point.hpp"
#include "util/lpm_math.hpp"
#include "util/lpm_tuple.hpp"

#ifdef LPM_USE_COMPOSE
#include "siqk_sqr.hpp"
#endif

namespace Lpm {

template <typename Geo, typename VT, typename CVT1, typename CVT2, typename Tri>
KOKKOS_INLINE_FUNCTION void triangular_barycentric_coords(VT& bc,
                                                          const CVT1& pt,
                                                          const CVT2& pts,
                                                          const Tri& tri) {
  const auto vertexA = Kokkos::subview(pts, tri[0], Kokkos::ALL);
  const auto vertexB = Kokkos::subview(pts, tri[1], Kokkos::ALL);
  const auto vertexC = Kokkos::subview(pts, tri[2], Kokkos::ALL);
  const Real total_area = Geo::tri_area(vertexA, vertexB, vertexC);
  const Real area_a = Geo::tri_area(pt, vertexB, vertexC);
  const Real area_b = Geo::tri_area(pt, vertexC, vertexA);
  const Real area_c = Geo::tri_area(pt, vertexA, vertexB);
  bc(0) = area_a / total_area;
  bc(1) = area_b / total_area;
  bc(2) = area_c / total_area;
}

KOKKOS_INLINE_FUNCTION
Real quad_quadratic_solve(const Real& a, const Real& b, const Real& c) {
  const Real det = square(b) - 4 * a * c;
  LPM_KERNEL_ASSERT(det > 0);
  const Real r1 = (-b + sqrt(det)) / (2 * a);
  const Real r2 = (-b - sqrt(det)) / (2 * a);
  if (FloatingPoint<Real>::in_bounds(r1, -1, 1)) {
    LPM_KERNEL_ASSERT(!FloatingPoint<Real>::in_bounds(r2, -1, 1));
    return r1;
  } else {
    LPM_KERNEL_ASSERT(FloatingPoint<Real>::in_bounds(r2));
    return r2;
  }
}

template <typename Geo, typename VT, typename CVT, typename Quad>
KOKKOS_INLINE_FUNCTION void quad_bilinear_to_ref(Real& xi, Real& eta,
                                                 const VT pt, const CVT& pts,
                                                 const Quad& quad) {
  const Real a1 =
      -pts(quad(0), 0) + pts(quad(1), 0) - pts(quad(2), 0) + pts(quad(3), 0);
  const Real a2 =
      -pts(quad(0), 1) + pts(quad(1), 1) - pts(quad(2), 1) + pts(quad(3), 1);
  const Real b1 =
      -pts(quad(0), 0) - pts(quad(1), 0) + pts(quad(2), 0) + pts(quad(3), 0);
  const Real b2 =
      -pts(quad(0), 1) - pts(quad(1), 1) + pts(quad(2), 1) + pts(quad(3), 1);
  const Real c1 =
      pts(quad(0), 0) - pts(quad(1), 0) - pts(quad(2), 0) + pts(quad(3), 0);
  const Real c2 =
      pts(quad(0), 1) - pts(quad(1), 1) - pts(quad(2), 1) + pts(quad(3), 1);
  const Real d1 = 4 * pt(0) - (pts(quad(0), 0) + pts(quad(1), 0) +
                               pts(quad(2), 0) + pts(quad(3), 0));
  const Real d2 = 4 * pt(1) - (pts(quad(0), 1) + pts(quad(1), 1) +
                               pts(quad(2), 1) + pts(quad(3), 1));

  LPM_KERNEL_ASSERT(!FloatingPoint<Real>::equiv(a1, b1));
  LPM_KERNEL_ASSERT(!FloatingPoint<Real>::equiv(a2, c2));

  if (FloatingPoint<Real>::zero(a1)) {
    // Case I.
    if (FloatingPoint<Real>::zero(a2)) {
      // Case I.A.
      xi = (d1 * c2 - d2 * c1) / (b1 * c2 - b2 * c1);
      eta = (b1 * d2 - b2 * d1) / (b1 * c2 - b2 * c1);
    } else {
      // Case I.B.
      LPM_KERNEL_ASSERT(!FloatingPoint<Real>::zero(b1));
      if (FloatingPoint<Real>::zero(c1)) {
        // Case 1.B.a.
        xi = d1 / b1;
        eta = (b1 * d2 - b2 * d1) / (a2 * d1 + b1 * c2);
      } else {
        // Case I.B.b.
        const Real qa = a2 * b1;
        const Real qb = c2 * b1 - a2 * d1 - b2 * c1;
        const Real qc = d2 * c1 - c2 * d1;
        xi = quad_quadratic_solve(qa, qb, qc);
        eta = (d1 - b1 * xi) / c1;
      }
    }
  } else {
    // Case II
    if (FloatingPoint<Real>::zero(a2)) {
      // Case II.B.
      if (FloatingPoint<Real>::zero(b2)) {
        // Case II.B.a.
        xi = (d1 * c2 - c1 * d2) / (a1 * d2 + b1 * c2);
        eta = d2 / c2;
      } else {
        // Case II.B.b.
        const Real qa = a1 * b2;
        const Real qb = c1 * b2 - a1 * d2 - b1 * c2;
        const Real qc = d1 * c2 - c1 * d2;
        xi = quad_quadratic_solve(qa, qb, qc);
        eta = (d2 - b2 * xi) / c2;
      }
    } else {
      // Case II.A.
      const Real ab = a2 * b1 - a1 * b2;
      const Real ac = a2 * c1 - a1 * c2;
      const Real ad = a2 * d1 - a1 * d2;
      if (FloatingPoint<Real>::zero(ab)) {
        // Case II.A.b.
        LPM_KERNEL_ASSERT(!FloatingPoint<Real>::zero(ac));
        xi = (d1 * ac - c1 * ad) / (b1 * ac + a1 * ad);
        eta = ad / ac;
      } else {
        // Case II.A.a.
        if (FloatingPoint<Real>::zero(ac)) {
          // Case II.A.a.2.
          LPM_KERNEL_ASSERT(FloatingPoint<Real>::equiv(a1 / a2, c1 / c2));
          xi = ad / ab;
          eta = (d1 * ab - b1 * ad) / (c1 * ab + a1 * ad);
        } else {
          // Case II.A.a.1.
          const Real qa = a1 * ab;
          const Real qb = c1 * ab - a1 * ad - b1 * ac;
          const Real qc = d1 * ac - c1 * ad;
          xi = quad_quadratic_solve(qa, qb, qc);
          eta = (ad - ab * xi) / ac;
        }
      }
    }
  }
}

template <typename VT, typename CVT, typename Quad>
KOKKOS_INLINE_FUNCTION void quad_ref_to_bilinear(VT& crds, const Real& xi,
                                                 const Real& eta,
                                                 const CVT& pts,
                                                 const Quad& e) {
  for (int i = 0; i < Geo::ndim; ++i) {
    crds[i] = (1 - xi) * (1 - eta) * pts(e[0], i) / 4;
    crds[i] += (1 + xi) * (1 - eta) * pts(e[1], i) / 4;
    crds[i] += (1 + xi) * (1 + eta) * pts(e[2], i) / 4;
    crds[i] += (1 - xi) * (1 + eta) * pts(e[3], i) / 4;
  }
}

template <typename VT1, typename VT2, typename Quad, typename Geo>
KOKKOS_INLINE_FUNCTION void quad_ref_coords(Real& a, Real& b, const VT1& pt,
                                            const VT2& pts, const Quad& quad) {}

template <typename PolymeshType, typename FieldType>
struct NativeInterpolator {
  typename FieldType::view_type tgt_view;
  typename PolymeshType::seed_type::geo::crd_view_type tgt_crds;
  typename PolymeshType::seed_type::geo::crd_view_type vert_crds;
  typename PolymeshType::seed_type::geo::crd_view_type face_crds;
  FieldType& src_field;
  MaskViewType exclude_mask;
  static constexpr Int ndim = FieldType::ndim;
  const typename Edges::edge_view_type edge_lefts;
  const typename Edges::edge_view_type edge_rights;
  const typename Edges::edge_tree_view edge_kids;
  const typename Faces<typename PolymeshType::FaceType,
                       typename PolymeshType::seed_type::geo>::vertex_view_type
      face_verts;
  const typename Faces<typename PolymeshType::FaceType,
                       typename PolymeshType::seed_type::geo>::face_tree_view
      face_kids,
      const typename Faces < typename PolymeshType::FaceType;
  typename PolymeshType::seed_type::geo > ::edge_view_type face_edges;

  NativeInterpolator(
      FieldType& dst,
      const typename PolymeshType::seed_type::geo::crd_view_type dst_crds,
      const FieldType& src, const PolymeshType& pm)
      : tgt_view(dst.view),
        tgt_crds(dst_crds),
        vert_crds(pm.vertices->phys_crds.crds),
        face_crds(pm.faces->phys_crds.crds),
        src_field(src),
        exclude_mask(pm.faces.mask),
        edge_lefts(pm.edges.lefts),
        edge_rights(pm.edges.rights),
        edge_kids(pm.edges.kids),
        face_verts(pm.faces.verts),
        face_kids(pm.faces.kids),
        face_edges(pm.faces.edges) {}

  static_assert(FieldType::field_loc == FieldLocation::VertexField,
                "vertex data required");
  static_assert(FieldType::ndim == 1, "scalar fields only");

  typename std::enable_if<
      std::is_same<typename PolymeshType::FaceType, TriFace>::value, void>::type
      KOKKOS_INLINE_FUNCTION
      operator()(const Index tgt_idx) const {
    const auto m_crd = Kokkos::subview(tgt_crds, tgt_idx, Kokkos::ALL);
    const Index f_idx = locate_face_containing_pt<decltype(m_crd), TriFace>(
        m_crd, edge_lefts, edge_rights, edge_kids, face_crds, face_kids,
        face_edges);

    const auto m_tri = Kokkos::subview(face_verts, f_idx, Kokkos::ALL);

    Real bc[3];
    triangular_barycentric_coords<typename PolymeshType::Geo, Real*,
                                  decltype(m_crd), decltype(vert_crds),
                                  decltype(m_tri)>(bc, m_crd, vert_crds, m_tri);

    tgt_view(tgt_idx) = 0;
    for (int j = 0; j < 3; ++j) {
      tgt_view(tgt_idx) += bc[j] * src_field.view(m_tri(j));
    }
  }

  typename std::enable_if<
      std::is_same<typename PolymeshType::FaceType, QuadFace>::value,
      void>::type KOKKOS_INLINE_FUNCTION
  operator()(const Index i) const {
    const auto m_crd = Kokkos::subview(tgt_crds, tgt_idx, Kokkos::ALL);
    const Index f_idx = locate_face_containing_pt<decltype(m_crd), QuadFace>(
        m_crd, edge_lefts, edge_rights, edge_kids, face_crds, face_kids,
        face_edges);

    const auto m_quad = Kokkos::subview(face_verts, f_idx, Kokkos::ALL);

    Real ab[2];
  }
};

}  // namespace Lpm

#endif
