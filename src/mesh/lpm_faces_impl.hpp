#ifndef LPM_FACES_IMPL_HPP
#define LPM_FACES_IMPL_HPP

#include <iomanip>
#include <iostream>
#include <sstream>

#include "lpm_assert.hpp"
#include "lpm_constants.hpp"
#include "lpm_faces.hpp"
#include "util/lpm_floating_point.hpp"
#include "util/lpm_string_util.hpp"

namespace Lpm {

template <typename FaceKind, typename Geo>
void Faces<FaceKind, Geo>::leaf_crd_view(
    const typename Geo::crd_view_type leaf_crds) const {
  LPM_REQUIRE(leaf_crds.extent(0) >= n_leaves_host());

  const auto l_idx = leaf_idx;
  Kokkos::parallel_for(
      "Faces::leaf_crd_view", _nh(), KOKKOS_LAMBDA(const Index i) {
        if (!has_kids(i)) {
          for (int j = 0; j < Geo::ndim; ++j) {
            leaf_crds(l_idx(i), j) = phys_crds.view(i, j);
          }
        }
      });
}

template <typename FaceKind, typename Geo>
void Faces<FaceKind, Geo>::leaf_crd_view(typename Geo::crd_view_type leaf_crds, const typename Geo::crd_view_type face_crds) const {
  LPM_REQUIRE(leaf_crds.extent(0) >= n_leaves_host());

  const auto l_idx = leaf_idx;
  Kokkos::parallel_for(
    "Faces::leaf_crd_view", _nh(),
    KOKKOS_LAMBDA (const Index i) {
      if (!has_kids(i)) {
        for (int j=0; j<Geo::ndim; ++j) {
          leaf_crds(l_idx(i), j) = face_crds(i,j);
        }
      }
    });
}

template <typename FaceKind, typename Geo>
typename Geo::crd_view_type Faces<FaceKind, Geo>::leaf_crd_view() const {
  typename Geo::crd_view_type result("leaf_crds", n_leaves_host());
  leaf_crd_view(result);
  return result;
}

template <typename FaceKind, typename Geo>
void Faces<FaceKind, Geo>::leaf_field_vals(
    const scalar_view_type& vals, const ScalarField<FaceField>& field) const {
  LPM_ASSERT(vals.extent(0) >= n_leaves_host());
  const auto l_idx = leaf_idx;
  Kokkos::parallel_for(
      "Faces::leaf_field_vals", _nh(), KOKKOS_LAMBDA(const Index i) {
        if (!has_kids(i)) {
          vals(l_idx(i)) = field.view(i);
        }
      });
}

template <typename FaceKind, typename Geo>
scalar_view_type Faces<FaceKind, Geo>::leaf_field_vals(
    const ScalarField<FaceField>& field) const {
  scalar_view_type result(field.name, n_leaves_host());
  leaf_field_vals(result, field);
  return result;
}

template <typename FaceKind, typename Geo>
void Faces<FaceKind, Geo>::leaf_field_vals(
    const typename Geo::vec_view_type& vals,
    const VectorField<Geo, FaceField>& field) const {
  LPM_ASSERT(vals.extent(0) >= n_leaves_host());
  const auto l_idx = leaf_idx;
  Kokkos::parallel_for(
      "Faces::leaf_field_vals", _nh(), KOKKOS_LAMBDA(const Index i) {
        if (!has_kids(i)) {
          for (int j = 0; j < Geo::ndim; ++j) {
            vals(l_idx(i), j) = field.view(i, j);
          }
        }
      });
}

template <typename FaceKind, typename Geo>
typename Geo::vec_view_type Faces<FaceKind, Geo>::leaf_field_vals(
    const VectorField<Geo, FaceField>& field) const {
  typename Geo::vec_view_type result("leaf_field_vals", n_leaves_host());
  leaf_field_vals(result, field);
  return result;
}

template <typename FaceKind, typename Geo>
void Faces<FaceKind, Geo>::scan_leaves() const {
  auto scan = leaf_idx;
  Index result;
  Kokkos::parallel_scan(
      "Faces::scan_leaves", _nh(),
      KOKKOS_LAMBDA(const Index i, Index& psum, bool is_final) {
        if (is_final) scan(i) = psum;
        psum += (has_kids(i) ? 0 : 1);
      },
      result);

#ifndef NDEBUG
  if (result != n_leaves_host()) {
    auto h_leaf_idx = Kokkos::create_mirror_view(leaf_idx);
    Kokkos::deep_copy(h_leaf_idx, leaf_idx);
    std::ostringstream ss;
    ss << "Faces::scan_leaves error: expected " << n_leaves_host()
       << " but result = " << result << "\n"
       << sprarr("faces.leaf_idx", h_leaf_idx.data(), n_leaves_host()) << "\n";
    std::cout << ss.str();
  }
#endif
  LPM_ASSERT(result == n_leaves_host());
}

template <typename FaceKind, typename Geo>
void Faces<FaceKind, Geo>::insert_host(const Index ctr_ind,
                                       ko::View<Index*, Host> vertinds,
                                       ko::View<Index*, Host> edgeinds,
                                       const Index prt, const Real ar) {

  LPM_ASSERT(prt >= 0);
  LPM_REQUIRE_MSG(_nh() + 1 <= _nmax,
                  "Faces::insert error: not enough memory.");
  const Index ins = _nh();
  for (int i = 0; i < FaceKind::nverts; ++i) {
    _hostverts(ins, i) = vertinds(i);
    _hostedges(ins, i) = edgeinds(i);
  }
  for (int i = 0; i < 4; ++i) {
    _hostkids(ins, i) = constants::NULL_IND;
  }
  _host_crd_inds(ins) = ctr_ind;
  _hostparent(ins) = prt;
  _hostarea(ins) = ar;
  _hlevel(ins) = (prt >= 0 ? _hlevel(prt) + 1 : 0);
  _hmask(ins) = false;
  _nh() += 1;
  _hn_leaves() += 1;
}

template <typename FaceKind, typename Geo>
template <typename PtViewType>
void Faces<FaceKind, Geo>::insert_host(
    const PtViewType& pcrd, const PtViewType& lcrd, const Index ctr_idx,
    const Kokkos::View<Index*, Host> vertinds,
    const Kokkos::View<Index*, Host> edgeinds, const Index parent_idx,
    const Real ar) {
  LPM_ASSERT(parent_idx < _nh());
  LPM_REQUIRE(_nh() + 1 <= _nmax);

  const Index insert_idx = _nh();
  phys_crds.insert_host(pcrd);
  lag_crds.insert_host(lcrd);
  for (int i = 0; i < FaceKind::nverts; ++i) {
    _hostverts(insert_idx, i) = vertinds(i);
    _hostedges(insert_idx, i) = edgeinds(i);
  }
  for (int i = 0; i < 4; ++i) {
    _hostkids(insert_idx, i) = constants::NULL_IND;
  }
  _host_crd_inds(insert_idx) = ctr_idx;
  _hostparent(insert_idx) = parent_idx;
  _hostarea(insert_idx) = ar;
  _hlevel(insert_idx) = (parent_idx >= 0 ? _hlevel(parent_idx) + 1 : 0);
  _hmask(insert_idx) = false;
  ++_nh();
  ++_hn_leaves();
}

template <typename FaceKind, typename Geo>
template <typename SeedType>
void Faces<FaceKind, Geo>::init_from_seed(const MeshSeed<SeedType>& seed) {
  LPM_REQUIRE_MSG(_nmax >= SeedType::nfaces,
                  "Faces::init_from_seed error: not enough memory.");
  for (int i = 0; i < SeedType::nfaces; ++i) {
    const auto mverts = ko::subview(seed.seed_face_verts, i, ko::ALL);
    const auto medges = ko::subview(seed.seed_face_edges, i, ko::ALL);
    const auto mcrd =
        ko::subview(seed.seed_crds, SeedType::nverts + i, ko::ALL);

    this->insert_host(mcrd, mcrd, i, mverts, medges, constants::NULL_IND,
                      seed.face_area(i));
  }
#ifndef NDEBUG
  if (std::is_same<Geo, SphereGeometry>::value) {
    Real surfarea = 0;
    for (int i = 0; i < SeedType::nfaces; ++i) {
      surfarea += _hostarea(i);
    }
    LPM_ASSERT(FloatingPoint<Real>::equiv(surfarea, 4 * constants::PI,
                                          constants::ZERO_TOL));
  }
#endif
}

template <typename FaceKind, typename Geo>
Real Faces<FaceKind, Geo>::surface_area_host() const {
  Real result = 0;
  for (Index i = 0; i < _nh(); ++i) {
    result += _hostarea(i);
  }
  return result;
}

template <typename FaceKind, typename Geo>
Real Faces<FaceKind, Geo>::appx_mesh_size() const {
  Real result = 0;
  const auto ar = area;
  const auto m = mask;
  Kokkos::parallel_reduce(_nh(),
    KOKKOS_LAMBDA (const Index i, Real& s) {
      if (!m(i)) {
        s += ar(i);
      }
    }, result);
  result /= _hn_leaves();
  return sqrt(result);
}

template <typename FaceKind, typename Geo>
Real Faces<FaceKind, Geo>::appx_min_mesh_size() const {
  Real result = 0;
  const auto ar = area;
  const auto m = mask;
  Kokkos::parallel_reduce(_nh(),
    KOKKOS_LAMBDA (const Index i, Real& a) {
      if (!m(i)) {
        a = (ar(i) < a ? ar(i) : a);
      }
    }, Kokkos::Min<Real>(result));
  return sqrt(result);
}

template <typename FaceKind, typename Geo>
Real Faces<FaceKind, Geo>::min_face_area() const {
  Real result = 0;
  const auto ar = area;
  const auto m = mask;
  Kokkos::parallel_reduce(_nh(),
    KOKKOS_LAMBDA (const Index i, Real& a) {
      if (!m(i)) {
        a = (ar(i) < a ? ar(i) : a);
      }
    }, Kokkos::Min<Real>(result));
  return result;
}

template <typename FaceKind, typename Geo>
Real Faces<FaceKind, Geo>::max_face_area() const {
  Real result = 0;
  const auto ar = area;
  const auto m = mask;
  Kokkos::parallel_reduce(_nh(),
    KOKKOS_LAMBDA (const Index i, Real& a) {
      if (!m(i)) {
        a = (ar(i) > a ? ar(i) : a);
      }
    }, Kokkos::Max<Real>(result));
  return result;
}

template <typename FaceKind, typename Geo>
Real Faces<FaceKind, Geo>::avg_face_area() const {
  return total_leaf_area() / n_leaves_host();
}

template <typename FaceKind, typename Geo>
Real Faces<FaceKind, Geo>::total_leaf_area() const {
  Real result = 0;
  const auto ar = area;
  const auto m = mask;
  Kokkos::parallel_reduce(_nh(),
    KOKKOS_LAMBDA (const Index& i, Real& a) {
      if (!m(i)) a += ar(i);
    }, result);
  return result;
}

template <typename FaceKind, typename Geo>
std::string Faces<FaceKind, Geo>::info_string(const std::string& label,
                                              const int& tab_level,
                                              const bool& dump_all) const {
  std::ostringstream oss;
  const auto idnt = indent_string(tab_level);
  const auto bigidnt = indent_string(tab_level + 1);
  oss << idnt << "Faces " << label << " info: nh = (" << _nh()
      << ") of nmax = " << _nmax << " in memory; " << _hn_leaves() << " leaves.\n"
      << " appx avg mesh size = " << appx_mesh_size() << "\n"
      << " appx min mesh size = " << appx_min_mesh_size() << "\n";

  if (dump_all) {
    for (Index i = 0; i < _nmax; ++i) {
      if (i == _nh()) oss << "---------------------------------" << std::endl;
      oss << "face(" << i << ") : ";
      oss << "verts = (";
      for (int j = 0; j < FaceKind::nverts; ++j) {
        oss << _hostverts(i, j) << (j == FaceKind::nverts - 1 ? ") " : ",");
      }
      oss << "edges = (";
      for (int j = 0; j < FaceKind::nverts; ++j) {
        oss << _hostedges(i, j) << (j == FaceKind::nverts - 1 ? ") " : ",");
      }
      oss << "center = (" << _host_crd_inds(i) << ") ";
      oss << "level = (" << _hlevel(i) << ") ";
      oss << "parent = (" << _hostparent(i) << ") ";
      oss << "kids = (" << _hostkids(i, 0) << "," << _hostkids(i, 1) << ","
          << _hostkids(i, 2) << "," << _hostkids(i, 3) << ") ";
      oss << "area = (" << _hostarea(i) << ")";
      oss << std::endl;
    }
    oss << phys_crds.info_string("faces.phys_crds", tab_level + 1, dump_all);
  }
  oss << bigidnt << "total area = " << std::setprecision(18)
      << surface_area_host() << std::endl;
  oss << bigidnt << "avg. mesh size = " << appx_mesh_size() << std::endl;
  oss << phys_crds.info_string(label, tab_level + 1, dump_all);
  //   oss << lag_crds.info_string(label, tab_level+1, dump_all);
  return oss.str();
}

template <typename Geo>
void FaceDivider<Geo, TriFace>::divide(const Index faceInd,
                                       Vertices<Coords<Geo>>& verts,
                                       Edges& edges,
                                       Faces<TriFace, Geo>& faces) {
  LPM_ASSERT(faceInd < faces.nh());

  LPM_REQUIRE_MSG(faces.n_max() >= faces.nh() + 4,
                  "Faces::divide error: not enough memory.");
  LPM_REQUIRE_MSG(!faces.has_kids_host(faceInd),
                  "Faces::divide error: called on previously divided face.");

  ko::View<Index[4][3], Host> new_face_edge_inds(
      "new_face_edge_inds");  // (child face index, edge index)
  ko::View<Index[4][3], Host> new_face_vert_inds(
      "new_face_vert_inds");  // (child face index, vertex index)

  // for debugging, set to invalid value
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      new_face_vert_inds(i, j) = constants::NULL_IND;
      new_face_edge_inds(i, j) = constants::NULL_IND;
    }
  }
  /// pull data from parent face
  auto parent_vert_inds = ko::subview(faces._hostverts, faceInd, ko::ALL());
  auto parent_edge_inds = ko::subview(faces._hostedges, faceInd, ko::ALL());

  /// determine child face indices
  const Index face_insert_pt = faces.nh();
  ko::View<Index[4], Host> new_face_kids("new_face_kids");
  for (int i = 0; i < 4; ++i) {
    new_face_kids(i) = face_insert_pt + i;
  }
  /// connect parent vertices to child faces
  for (int i = 0; i < 3; ++i) {
    new_face_vert_inds(i, i) = parent_vert_inds(i);
  }
  /// loop over parent edges, replace with child edges
  for (int i = 0; i < 3; ++i) {
    const Index parent_edge = parent_edge_inds(i);
    ko::View<Index[2], Host> edge_kids("edge_kids");
    if (edges.has_kids_host(parent_edge)) {  // edge already divided
      edge_kids(0) = edges.kid_host(parent_edge, 0);
      edge_kids(1) = edges.kid_host(parent_edge, 1);
    } else {  // divide edge
      edge_kids(0) = edges.nh();
      edge_kids(1) = edges.nh() + 1;

      edges.divide(parent_edge, verts);
    }

    // connect child edges to child faces
    if (faces.edge_is_positive(faceInd, i,
                               edges)) {  // edge has positive orientation
      new_face_edge_inds(i, i) = edge_kids(0);
      edges.set_left(edge_kids(0), new_face_kids(i));

      new_face_edge_inds((i + 1) % 3, i) = edge_kids(1);
      edges.set_left(edge_kids(1), new_face_kids((i + 1) % 3));
    } else {  // edge has negative orientation
      new_face_edge_inds(i, i) = edge_kids(1);
      edges.set_right(edge_kids(1), new_face_kids(i));

      new_face_edge_inds((i + 1) % 3, i) = edge_kids(0);
      edges.set_right(edge_kids(0), new_face_kids((i + 1) % 3));
    }

    const Index c1dest = edges.dest_host(edge_kids(0));
    if (i == 0) {
      new_face_vert_inds(0, 1) = c1dest;
      new_face_vert_inds(1, 0) = c1dest;
      new_face_vert_inds(3, 2) = c1dest;
    } else if (i == 1) {
      new_face_vert_inds(1, 2) = c1dest;
      new_face_vert_inds(2, 1) = c1dest;
      new_face_vert_inds(3, 0) = c1dest;
    } else {
      new_face_vert_inds(2, 0) = c1dest;
      new_face_vert_inds(0, 2) = c1dest;
      new_face_vert_inds(3, 1) = c1dest;
    }
  }
  LPM_ASSERT(verts.nh() == verts.phys_crds.nh());

#ifndef NDEBUG
  // debug: check vertex connectivity
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      LPM_REQUIRE_MSG(new_face_vert_inds(i, j) != constants::NULL_IND,
                      "TriFace::divide error: vertex connectivity");
    }
  }
#endif

  /// create new interior edges
  const Index edge_ins_pt = edges.nh();
  for (int i = 0; i < 3; ++i) {
    new_face_edge_inds(3, i) = edge_ins_pt + i;
  }
  new_face_edge_inds(0, 1) = edge_ins_pt + 1;
  new_face_edge_inds(1, 2) = edge_ins_pt + 2;
  new_face_edge_inds(2, 0) = edge_ins_pt;
  edges.insert_host(new_face_vert_inds(2, 1), new_face_vert_inds(2, 0),
                    new_face_kids(3), new_face_kids(2));
  edges.insert_host(new_face_vert_inds(0, 2), new_face_vert_inds(0, 1),
                    new_face_kids(3), new_face_kids(0));
  edges.insert_host(new_face_vert_inds(1, 0), new_face_vert_inds(1, 2),
                    new_face_kids(3), new_face_kids(1));

  /// create new center coordinates
  ko::View<Real[3][Geo::ndim], Host> vert_crds("vert_crds");
  ko::View<Real[3][Geo::ndim], Host> vert_lag_crds("vert_lag_crds");
  ko::View<Real[4][Geo::ndim], Host> face_crds("face_crds");
  ko::View<Real[4][Geo::ndim], Host> face_lag_crds("face_lag_crds");
  ko::View<Real[4], Host> face_area("face_area");
  for (int i = 0; i < 4; ++i) {              // loop over child Faces
    for (int j = 0; j < 3; ++j) {            // loop over vertices
      for (int k = 0; k < Geo::ndim; ++k) {  // loop over components
        vert_crds(j, k) = verts.phys_crds.get_crd_component_host(
            verts.host_crd_ind(new_face_vert_inds(i, j)), k);
        vert_lag_crds(j, k) = verts.lag_crds.get_crd_component_host(
            verts.host_crd_ind(new_face_vert_inds(i, j)), k);
      }
    }
    auto ctr = ko::subview(face_crds, i, ko::ALL());
    auto lagctr = ko::subview(face_lag_crds, i, ko::ALL());
    Geo::barycenter(ctr, vert_crds, 3);
    Geo::barycenter(lagctr, vert_lag_crds, 3);
    face_area(i) = Geo::polygon_area(ctr, vert_crds, 3);
    LPM_ASSERT(face_area(i) > constants::ZERO_TOL);
  }

  /// create new child Faces
  const int crd_insert_pt = faces.phys_crds.nh();
  for (int i = 0; i < 4; ++i) {
    faces.phys_crds.insert_host(ko::subview(face_crds, i, ko::ALL));
    faces.lag_crds.insert_host(ko::subview(face_lag_crds, i, ko::ALL));
    faces.insert_host(
        crd_insert_pt + i, ko::subview(new_face_vert_inds, i, ko::ALL()),
        ko::subview(new_face_edge_inds, i, ko::ALL()), faceInd, face_area(i));
  }
  /// Remove parent from leaf computations
  faces.set_kids_host(faceInd, new_face_kids);
  faces.set_area_host(faceInd, 0.0);
  faces.set_leaf_mask(faceInd, true);
  faces.decrement_leaves();
}

template <typename Geo>
void FaceDivider<Geo, QuadFace>::divide(const Index faceInd,
                                        Vertices<Coords<Geo>>& verts,
                                        Edges& edges,
                                        Faces<QuadFace, Geo>& faces) {
  LPM_ASSERT((faceInd >= 0 and faceInd < faces.nh()));
  LPM_REQUIRE_MSG(faces.n_max() >= faces.nh() + 4,
                  "Faces::divide error: not enough memory.");
  LPM_REQUIRE_MSG(!faces.has_kids_host(faceInd),
                  "Faces::divide error: called on previously divided face.");

  ko::View<Index[4][4], Host> new_face_edge_inds(
      "new_face_edge_inds");  // (child face index, edge index)
  ko::View<Index[4][4], Host> new_face_vert_inds(
      "new_face_vert_inds");  // (child face index, vertex index)

  // init: set to invalid value
  ko::deep_copy(new_face_vert_inds, constants::NULL_IND);
  ko::deep_copy(new_face_edge_inds, constants::NULL_IND);

  /// pull data from parent
  auto parent_vert_inds = ko::subview(faces._hostverts, faceInd, ko::ALL());
  auto parent_edge_inds = ko::subview(faces._hostedges, faceInd, ko::ALL());
  // determine child face indices
  const Index face_insert_pt = faces.nh();
  ko::View<Index[4], Host> new_face_kids("new_face_kids");
  for (int i = 0; i < 4; ++i) {
    new_face_kids(i) = face_insert_pt + i;
  }
  /// connect parent vertices to child faces
  for (int i = 0; i < 4; ++i) {
    new_face_vert_inds(i, i) = parent_vert_inds(i);
  }

  ko::View<Index[2], Host> edge_kids("edge_kids");
  /// loop over parent edges
  for (int i = 0; i < 4; ++i) {
    const Index parent_edge = parent_edge_inds(i);
    if (edges.has_kids_host(parent_edge)) {  // edge already divided
      edge_kids(0) = edges.kid_host(parent_edge, 0);
      edge_kids(1) = edges.kid_host(parent_edge, 1);
    } else {  // divide edge
      edge_kids(0) = edges.nh();
      edge_kids(1) = edges.nh() + 1;

      edges.divide(parent_edge, verts);
    }

    /// connect child edges to child faces
    if (faces.edge_is_positive(faceInd, i,
                               edges)) {  /// edge has positive orientation
      new_face_edge_inds(i, i) = edge_kids(0);
      edges.set_left(edge_kids(0), new_face_kids(i));

      new_face_edge_inds((i + 1) % 4, i) = edge_kids(1);
      edges.set_left(edge_kids(1), new_face_kids((i + 1) % 4));
    } else {  /// edge has negative orientation
      new_face_edge_inds(i, i) = edge_kids(1);
      edges.set_right(edge_kids(1), new_face_kids(i));

      new_face_edge_inds((i + 1) % 4, i) = edge_kids(0);
      edges.set_right(edge_kids(0), new_face_kids((i + 1) % 4));
    }
    const Index c1dest = edges.dest_host(edge_kids(0));
    new_face_vert_inds(i, (i + 1) % 4) = c1dest;
    new_face_vert_inds((i + 1) % 4, i) = c1dest;
  }

  LPM_ASSERT(verts.nh() == verts.phys_crds.nh());

  /// special case for QuadFace: parent center becomes a vertex
  // we don't overwrite the face coordinate because it's still the face
  // coordinate of the parent face
  const Index parent_center_ind = faces._host_crd_inds(faceInd);
  ko::View<Real[Geo::ndim], Host> newcrd("newcrd");
  ko::View<Real[Geo::ndim], Host> newlagcrd("newlagcrd");
  for (int i = 0; i < Geo::ndim; ++i) {
    newcrd(i) = faces.phys_crds.get_crd_component_host(parent_center_ind, i);
    newlagcrd(i) = faces.lag_crds.get_crd_component_host(parent_center_ind, i);
  }
  Index vert_insert_pt = verts.nh();
  verts.insert_host(newcrd, newlagcrd);
  for (int i = 0; i < 4; ++i) {
    new_face_vert_inds(i, (i + 2) % 4) = vert_insert_pt;
  }

  /// create new interior edges
  const Index edge_ins_pt = edges.nh();
  edges.insert_host(new_face_vert_inds(0, 1), new_face_vert_inds(0, 2),
                    new_face_kids(0), new_face_kids(1));
  new_face_edge_inds(0, 1) = edge_ins_pt;
  new_face_edge_inds(1, 3) = edge_ins_pt;
  edges.insert_host(new_face_vert_inds(2, 0), new_face_vert_inds(2, 3),
                    new_face_kids(3), new_face_kids(2));
  new_face_edge_inds(2, 3) = edge_ins_pt + 1;
  new_face_edge_inds(3, 1) = edge_ins_pt + 1;
  edges.insert_host(new_face_vert_inds(2, 1), new_face_vert_inds(2, 0),
                    new_face_kids(1), new_face_kids(2));
  new_face_edge_inds(1, 2) = edge_ins_pt + 2;
  new_face_edge_inds(2, 0) = edge_ins_pt + 2;
  edges.insert_host(new_face_vert_inds(3, 1), new_face_vert_inds(3, 0),
                    new_face_kids(0), new_face_kids(3));
  new_face_edge_inds(0, 2) = edge_ins_pt + 3;
  new_face_edge_inds(3, 0) = edge_ins_pt + 3;

  /// create new center coordinates
  ko::View<Real[4][Geo::ndim], Host> vert_crds("vert_crds");
  ko::View<Real[4][Geo::ndim], Host> vert_lag_crds("vert_lag_crds");
  ko::View<Real[4][Geo::ndim], Host> face_crds("face_crds");
  ko::View<Real[4][Geo::ndim], Host> face_lag_crds("face_lag_crds");
  ko::View<Real[4], Host> face_area("face_area");
  for (int i = 0; i < 4; ++i) {              // loop over child faces
    for (int j = 0; j < 4; ++j) {            // loop over vertices
      for (int k = 0; k < Geo::ndim; ++k) {  // loop over components
        vert_crds(j, k) =
            verts.phys_crds.get_crd_component_host(new_face_vert_inds(i, j), k);
        vert_lag_crds(j, k) =
            verts.lag_crds.get_crd_component_host(new_face_vert_inds(i, j), k);
      }
    }
    auto ctr = ko::subview(face_crds, i, ko::ALL());
    auto lagctr = ko::subview(face_lag_crds, i, ko::ALL());
    Geo::barycenter(ctr, vert_crds, 4);
    Geo::barycenter(lagctr, vert_lag_crds, 4);
    face_area(i) = Geo::polygon_area(ctr, vert_crds, 4);
  }
  const Index face_crd_ins_pt = faces.phys_crds.nh();

  for (int i = 0; i < 4; ++i) {
    faces.phys_crds.insert_host(ko::subview(face_crds, i, ko::ALL));
    faces.lag_crds.insert_host(ko::subview(face_lag_crds, i, ko::ALL));
    faces.insert_host(
        face_crd_ins_pt + i, ko::subview(new_face_vert_inds, i, ko::ALL()),
        ko::subview(new_face_edge_inds, i, ko::ALL()), faceInd, face_area(i));
  }

  /// remove parent from leaf computations
  faces.set_kids_host(faceInd, new_face_kids);
  faces.set_area_host(faceInd, 0.0);
  faces.set_leaf_mask(faceInd, true);
  faces.decrement_leaves();
}

}  // namespace Lpm
#endif
