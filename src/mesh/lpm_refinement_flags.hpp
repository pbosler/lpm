#ifndef LPM_REFINEMENT_FLAGS_HPP
#define LPM_REFINEMENT_FLAGS_HPP

#include "LpmConfig.h"

namespace Lpm {

template <typename FlagType>
std::string flag_info_string(const FlagType& f); // fwd decl

/** All refinement flag functors toggle a flag view entry to True
if their condition is met.

They do not change an existing True in that same entry.
*/

/** A flag view is 1-1 with a mesh's panels.

  flag_view(i) = true will trigger refinement of that panel in an
          adaptive refinement workflow.
  flag_view(i) = false will not.
*/
typedef Kokkos::View<bool*> flag_view;

template <typename MeshSeedType>
struct FlowMapVariationFlag {
  typedef typename MeshSeedType::geo::crd_view_type crd_view_type;

  flag_view flags;
  crd_view_type vertex_lag_crds;
  Kokkos::View<Index**> face_vertex_view;
  mask_view_type facemask;
  Index nfaces;
  Real relative_tol;
  Real tol;
  static constexpr Int nverts = MeshSeedType::faceKind::nverts;

  KOKKOS_INLINE_FUNCTION
  FlowMapVariationFlag(const FlowMapVariationFlag& other) = default;

  FlowMapVariationFlag(
      flag_view f, const PolyMesh2d<MeshSeedType>& mesh,
      const Real rtol)
      : flags(f),
        vertex_lag_crds(mesh.vertices.lag_crds.view),
        face_vertex_view(mesh.faces.verts),
        facemask(mesh.faces.mask),
        nfaces(mesh.n_faces_host()),
        relative_tol(rtol),
        tol(rtol) {}

  std::string description() const {
    return "FlowMapVariationFlag";
  }

  void set_tol_from_relative_value() {
    Real max_var;
    const auto fverts = face_vertex_view;
    const auto vlag_crds = vertex_lag_crds;
    const auto mask = facemask;
    Kokkos::parallel_reduce(nfaces,
      KOKKOS_LAMBDA (const Index i, Real& m) {
        if (!mask(i)) {
          Real min_lag_crds[MeshSeedType::geo::ndim];
          Real max_lag_crds[MeshSeedType::geo::ndim];
          for (int j = 0; j < MeshSeedType::geo::ndim; ++j) {
            min_lag_crds[j] = vlag_crds(fverts(i, 0), j);
            max_lag_crds[j] = vlag_crds(fverts(i, 0), j);
          }
          for (int j = 1; j < nverts; ++j) {
            const auto x0 = Kokkos::subview(vlag_crds, fverts(i, j),
                                            Kokkos::ALL);
            for (int k = 0; k < MeshSeedType::geo::ndim; ++k) {
              if (x0[k] < min_lag_crds[k]) min_lag_crds[k] = x0[k];
              if (x0[k] > max_lag_crds[k]) max_lag_crds[k] = x0[k];
            }
          }
          Real dsum = 0;
          for (int j = 0; j < MeshSeedType::geo::ndim; ++j) {
            dsum += max_lag_crds[j] - min_lag_crds[j];
          }
          m = (dsum > m ? dsum : m);
        }
      }, Kokkos::Max<Real>(max_var));
    tol = relative_tol * max_var;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index i) const {
    if (!facemask(i)) {
      Real min_lag_crds[MeshSeedType::geo::ndim];
      Real max_lag_crds[MeshSeedType::geo::ndim];
      for (int j = 0; j < MeshSeedType::geo::ndim; ++j) {
        min_lag_crds[j] = vertex_lag_crds(face_vertex_view(i, 0), j);
        max_lag_crds[j] = vertex_lag_crds(face_vertex_view(i, 0), j);
      }

      for (int j = 1; j < nverts; ++j) {
        const auto x0 = Kokkos::subview(vertex_lag_crds, face_vertex_view(i, j),
                                        Kokkos::ALL);
        for (int k = 0; k < MeshSeedType::geo::ndim; ++k) {
          if (x0[k] < min_lag_crds[k]) min_lag_crds[k] = x0[k];
          if (x0[k] > max_lag_crds[k]) max_lag_crds[k] = x0[k];
        }
      }

      Real dsum = 0;
      for (int j = 0; j < MeshSeedType::geo::ndim; ++j) {
        dsum += max_lag_crds[j] - min_lag_crds[j];
      }

      flags(i) = (flags(i) or (dsum > tol));
    }
  }

  std::string info_string() const {return flag_info_string(*this);}
};

/** Functor that will flag all faces whose scalar field's absolute value
  exceeds a tolerance.
*/
struct ScalarMaxFlag {
  flag_view flags;
  scalar_view_type face_vals;
  mask_view_type facemask;
  Index nfaces;
  Real relative_tol;
  Real tol;

  KOKKOS_INLINE_FUNCTION
  ScalarMaxFlag(const ScalarMaxFlag& other) = default;

  /** Constructor.

    @param [in/out] f flag_view
    @param [in] fv scalar field values at faces
    @param [in] m face mask to exclude divided faces
    @param [in] n number of faces
    @param [in] tol relative tolerance in (0,1)
  */
  ScalarMaxFlag(flag_view f, const scalar_view_type fv,
    const mask_view_type m, const Index n, const Real rtol) :
    flags(f), face_vals(fv), facemask(m), nfaces(n),
    relative_tol(rtol),
    tol(rtol) {}

  /** Reset tolerance to absolute (not relative) values.  Should be
    called before using this functor to flag panels.

    sets tol = relative_tol * max_{faces} abs(field_values)
  */
  void set_tol_from_relative_value() {
    Real max_abs;
    const auto fvals = face_vals;
    Kokkos::parallel_reduce(nfaces,
      KOKKOS_LAMBDA (const Index i, Real& m) {
        m = (m > abs(fvals(i)) ? m : abs(fvals(i)));
      }, Kokkos::Max<Real>(max_abs));
    tol = relative_tol * max_abs;
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index i) const {
    if (not facemask(i) ) {
      flags(i) = ( flags(i) or ( abs(face_vals(i)) > tol ) );
    }
  }

  std::string description() const {
    return "ScalarMaxFlag";
  }

  std::string info_string() const {return flag_info_string(*this);}
};

struct ScalarIntegralFlag {
  flag_view flags;
  scalar_view_type face_vals;
  scalar_view_type area;
  mask_view_type facemask;
  Index nfaces;
  Real relative_tol;
  Real tol;

  KOKKOS_INLINE_FUNCTION
  ScalarIntegralFlag(const ScalarIntegralFlag& other) = default;

  ScalarIntegralFlag(flag_view f, const scalar_view_type fv,
                     const scalar_view_type a, const mask_view_type m,
                     const Index n,
                     const Real rtol)
      : flags(f), face_vals(fv), area(a), facemask(m), nfaces(n),
        relative_tol(rtol),
        tol(rtol) {}

  void set_tol_from_relative_value() {
    Real max_abs;
    const auto fvals = face_vals;
    const auto ar = area;
    Kokkos::parallel_reduce(nfaces,
      KOKKOS_LAMBDA (const Index i, Real& m) {
        const Real val = abs(fvals(i))*ar(i);
        m = (val > m ? val : m);
      }, Kokkos::Max<Real>(max_abs));
    tol = relative_tol * max_abs;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index i) const {
    if (!facemask(i)) {
      flags(i) = (flags(i) or (abs(face_vals(i)) * area(i) > tol));
    }
  }

  std::string description() const {
    return "ScalarIntegralFlag";
  }

  std::string info_string() const {return flag_info_string(*this);}
};

struct ScalarVariationFlag {
  flag_view flags;
  scalar_view_type face_vals;
  scalar_view_type vert_vals;
  Kokkos::View<Index**> face_vertex_view;
  mask_view_type facemask;
  Index nfaces;
  Real relative_tol;
  Real tol;

  KOKKOS_INLINE_FUNCTION
  ScalarVariationFlag(const ScalarVariationFlag& other) = default;

  ScalarVariationFlag(flag_view f, const scalar_view_type fv,
                      const scalar_view_type vv,
                      const Kokkos::View<Index**> verts, const mask_view_type m,
                      const Index n,
                      const Real rtol)
      : flags(f),
        face_vals(fv),
        vert_vals(vv),
        face_vertex_view(verts),
        facemask(m),
        nfaces(n),
        relative_tol(rtol),
        tol(rtol) {}

  void set_tol_from_relative_value() {
    Real max_abs;
    const auto fvals = face_vals;
    const auto vvals = vert_vals;
    const auto fverts = face_vertex_view;
    const auto nverts = face_vertex_view.extent(1);
    const auto mask = facemask;
    Kokkos::parallel_reduce(nfaces,
      KOKKOS_LAMBDA (const Index i, Real& m) {
        if (!mask(i)) {
          Real minval = fvals(i);
          Real maxval = fvals(i);
          for (int j=0; j<nverts; ++j) {
            const auto v_idx = fverts(i,j);
            if (vvals(v_idx) < minval ) minval = vvals(v_idx);
            if (vvals(v_idx) > maxval ) maxval = vvals(v_idx);
          }
          const Real dval = maxval - minval;
          m = (dval > m ? dval : m);
        }
      }, Kokkos::Max<Real>(max_abs));
    tol = relative_tol * max_abs;
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index i) const {
    if (!facemask(i)) {
      Real minval = face_vals(i);
      Real maxval = face_vals(i);
      for (int j = 0; j < face_vertex_view.extent(1); ++j) {
        const auto vidx = face_vertex_view(i, j);
        if (vert_vals(vidx) < minval) minval = vert_vals(vidx);
        if (vert_vals(vidx) > maxval) maxval = vert_vals(vidx);
      }
      flags(i) = (flags(i) or (maxval - minval > tol));
    }
  }

  std::string description() const {
    return "ScalarVariationFlag";
  }

  std::string info_string() const {return flag_info_string(*this);}
};

template <typename FlagType>
std::string flag_info_string(const FlagType& f) {
  std::ostringstream ss;
  ss << f.description() << ": relative_tol = " << f.relative_tol << ", tol = " << f.tol << "\n";
  return ss.str();
}

}  // namespace Lpm

#endif
