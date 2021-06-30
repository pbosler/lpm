#include "mesh/lpm_faces.hpp"
#include "lpm_assert.hpp"
#include "lpm_constants.hpp"
#include "util/lpm_string_util.hpp"
#ifdef LPM_HAVE_NETCDF
#include "LpmNetCDF.hpp"
#endif

namespace Lpm {

template <typename FaceKind, typename Geo>
void Faces<FaceKind,Geo>::insert_host(const Index ctr_ind, ko::View<Index*,Host> vertinds, ko::View<Index*,Host> edgeinds, const Index prt, const Real ar) {
  LPM_REQUIRE_MSG(_nh()+1 <= _nmax, "Faces::insert error: not enough memory.");
  const Index ins = _nh();
  for (int i=0; i<FaceKind::nverts; ++i) {
    _hostverts(ins, i) = vertinds(i);
    _hostedges(ins, i) = edgeinds(i);
  }
  for (int i=0; i<4; ++i) {
    _hostkids(ins, i) = constants::NULL_IND;
  }
  _host_crd_inds(ins) = ctr_ind;
  _hostparent(ins) = prt;
  _hostarea(ins) = ar;
  _hlevel(ins) = _hlevel(prt)+1;
  _hmask(ins) = false;
  _nh() += 1;
  _hn_leaves() += 1;
}

template <typename FaceKind, typename Geo> template<typename SeedType>
void Faces<FaceKind,Geo>::init_from_seed(const MeshSeed<SeedType>& seed) {
  LPM_REQUIRE_MSG(_nmax >= SeedType::nfaces, "Faces::init_from_seed error: not enough memory.");
  for (int i=0; i<SeedType::nfaces; ++i) {
    for (int j=0; j<SeedType::nfaceverts; ++j) {
      _hostverts(i,j) = seed.seed_face_verts(i,j);
      _hostedges(i,j) = seed.seed_face_edges(i,j);
    }
    _host_crd_inds(i) = i;
    _hostarea(i) = seed.face_area(i);
    _hostparent(i) = constants::NULL_IND;
    for (int j=0; j<4; ++j) {
      _hostkids(i, j) = constants::NULL_IND;
    }
    _hlevel(i) = 0;
    _hmask(i) = false;
  }
  _nh() = SeedType::nfaces;
  _hn_leaves() = SeedType::nfaces;
  if (seed.id_string() == "UnitDiskSeed") _hostkids(0,1) = 0;
}

template <typename FaceKind, typename Geo>
Real Faces<FaceKind,Geo>::surface_area_host() const {
  Real result = 0;
  for (Index i=0; i<_nh(); ++i) {
    result += _hostarea(i);
  }
  return result;
}

template <typename FaceKind, typename Geo>
std::string Faces<FaceKind,Geo>::info_string(const std::string& label, const int& tab_level, const bool& dump_all) const {
  std::ostringstream oss;
  const auto idnt = indent_string(tab_level);
  const auto bigidnt = indent_string(tab_level+1);
  oss << idnt <<  "Faces " << label << " info: nh = (" << _nh() << ") of nmax = " << _nmax << " in memory; "
    << _hn_leaves() << " leaves." << std::endl;

  if (dump_all) {
    for (Index i=0; i<_nmax; ++i) {
      if (i==_nh()) oss << "---------------------------------" << std::endl;
      oss << "face(" << i << ") : ";
      oss << "verts = (";
      for (int j=0; j<FaceKind::nverts; ++j) {
        oss << _hostverts(i,j) << (j==FaceKind::nverts-1 ? ") " : ",");
      }
      oss << "edges = (";
      for (int j=0; j<FaceKind::nverts; ++j) {
        oss << _hostedges(i,j) << (j==FaceKind::nverts-1 ? ") " : ",");
      }
      oss << "center = (" << _host_crd_inds(i) << ") ";
      oss << "level = (" << _hlevel(i) << ") ";
      oss << "parent = (" << _hostparent(i) << ") ";
      oss << "kids = (" << _hostkids(i,0) << "," << _hostkids(i,1) << ","
        << _hostkids(i,2) << "," << _hostkids(i,3) <<") ";
      oss << "area = (" << _hostarea(i) << ")";
      oss << std::endl;
    }
  }
  oss << bigidnt << "total area = " << surface_area_host() << std::endl;
  return oss.str();
}

template <typename Geo>
void FaceDivider<Geo, TriFace>::divide(const Index faceInd, Vertices<Coords<Geo>>& verts,
    Edges& edges, Faces<TriFace,Geo>& faces, Coords<Geo>& physFaces, Coords<Geo>& lagFaces){

  LPM_ASSERT(faceInd < faces.nh());

  LPM_REQUIRE_MSG(faces.n_max() >= faces.nh() + 4, "Faces::divide error: not enough memory.");
  LPM_REQUIRE_MSG(!faces.has_kids_host(faceInd), "Faces::divide error: called on previously divided face.");

  ko::View<Index[4][3], Host> new_face_edge_inds("new_face_edge_inds");  // (child face index, edge index)
  ko::View<Index[4][3], Host> new_face_vert_inds("new_face_vert_inds");  // (child face index, vertex index)

  // for debugging, set to invalid value
  for (int i=0; i<4; ++i) {
    for (int j=0; j<3; ++j) {
      new_face_vert_inds(i,j) = constants::NULL_IND;
      new_face_edge_inds(i,j) = constants::NULL_IND;
    }
  }
  /// pull data from parent face
  auto parent_vert_inds = ko::subview(faces._hostverts, faceInd, ko::ALL());
  auto parent_edge_inds = ko::subview(faces._hostedges, faceInd, ko::ALL());

  /// determine child face indices
  const Index face_insert_pt = faces.nh();
  ko::View<Index[4], Host> new_face_kids("new_face_kids");
  for (int i=0; i<4; ++i) {
    new_face_kids(i) = face_insert_pt+i;
  }
  /// connect parent vertices to child faces
  for (int i=0; i<3; ++i) {
    new_face_vert_inds(i,i) = parent_vert_inds(i);
  }
  /// loop over parent edges, replace with child edges
  for (int i=0; i<3; ++i) {
    const Index parent_edge = parent_edge_inds(i);
    ko::View<Index[2], Host> edge_kids("edge_kids");
    if (edges.has_kids_host(parent_edge)) { // edge already divided
      edge_kids(0) = edges.kid_host(parent_edge, 0);
      edge_kids(1) = edges.kid_host(parent_edge, 1);
    }
    else { // divide edge
      edge_kids(0) = edges.nh();
      edge_kids(1) = edges.nh() + 1;

      edges.divide(parent_edge, verts);
    }

    // connect child edges to child faces
    if (faces.edge_is_positive(faceInd, i, edges)) { // edge has positive orientation
      new_face_edge_inds(i,i) = edge_kids(0);
      edges.set_left(edge_kids(0), new_face_kids(i));

      new_face_edge_inds((i+1)%3, i) = edge_kids(1);
      edges.set_left(edge_kids(1), new_face_kids((i+1)%3));
    }
    else { // edge has negative orientation
      new_face_edge_inds(i,i) = edge_kids(1);
      edges.set_right(edge_kids(1), new_face_kids(i));

      new_face_edge_inds((i+1)%3, i) = edge_kids(0);
      edges.set_right(edge_kids(0), new_face_kids((i+1)%3));
    }

    const Index c1dest = edges.dest_host(edge_kids(0));
    if (i==0) {
      new_face_vert_inds(0,1) = c1dest;
      new_face_vert_inds(1,0) = c1dest;
      new_face_vert_inds(3,2) = c1dest;
    }
    else if (i==1) {
      new_face_vert_inds(1,2) = c1dest;
      new_face_vert_inds(2,1) = c1dest;
      new_face_vert_inds(3,0) = c1dest;
    }
    else {
      new_face_vert_inds(2,0) = c1dest;
      new_face_vert_inds(0,2) = c1dest;
      new_face_vert_inds(3,1) = c1dest;
    }
  }

  // debug: check vertex connectivity
  for (int i=0; i<4; ++i) {
    for (int j=0; j<3; ++j) {
      LPM_REQUIRE_MSG(new_face_vert_inds(i,j) != constants::NULL_IND, "TriFace::divide error: vertex connectivity");
    }
  }

  /// create new interior edges
  const Index edge_ins_pt = edges.nh();
  for (int i=0; i<3; ++i) {
    new_face_edge_inds(3,i) = edge_ins_pt+i;
  }
  new_face_edge_inds(0,1) = edge_ins_pt+1;
  new_face_edge_inds(1,2) = edge_ins_pt+2;
  new_face_edge_inds(2,0) = edge_ins_pt;
  edges.insert_host(new_face_vert_inds(2,1), new_face_vert_inds(2,0), new_face_kids(3), new_face_kids(2));
  edges.insert_host(new_face_vert_inds(0,2), new_face_vert_inds(0,1), new_face_kids(3), new_face_kids(0));
  edges.insert_host(new_face_vert_inds(1,0), new_face_vert_inds(1,2), new_face_kids(3), new_face_kids(1));

  /// create new center coordinates
  ko::View<Real[3][Geo::ndim], Host> vert_crds("vert_crds");
  ko::View<Real[3][Geo::ndim], Host> vert_lag_crds("vert_lag_crds");
  ko::View<Real[4][Geo::ndim], Host> face_crds("face_crds");
  ko::View<Real[4][Geo::ndim], Host> face_lag_crds("face_lag_crds");
  ko::View<Real[4], Host> face_area("face_area");
  for (int i=0; i<4; ++i) { // loop over child Faces
    auto ctr = ko::subview(face_crds, i, ko::ALL());
    auto lagctr = ko::subview(face_lag_crds, i, ko::ALL());
    for (int j=0; j<3; ++j) { // loop over vertices
      for (int k=0; k<Geo::ndim; ++k) { // loop over components
        vert_crds(j,k) = verts.phys_crds->get_crd_component_host(new_face_vert_inds(i,j),k);
        vert_lag_crds(j,k) = verts.lag_crds->get_crd_component_host(new_face_vert_inds(i,j),k);
      }
    }
    Geo::barycenter(ctr, vert_crds, 3);
    Geo::barycenter(lagctr, vert_lag_crds, 3);
    face_area(i) = Geo::polygon_area(ctr, vert_crds, 3);
  }

  /// create new child Faces
  const Index crd_ins_pt = physFaces.nh();
  for (int i=0; i<4; ++i) {
    physFaces.insert_host(slice(face_crds,i));
    lagFaces.insert_host(slice(face_lag_crds,i));
    faces.insert_host(crd_ins_pt+i, ko::subview(new_face_vert_inds,i, ko::ALL()), ko::subview(new_face_edge_inds, i, ko::ALL()), faceInd, face_area(i));
  }
  /// Remove parent from leaf computations
  faces.set_kids_host(faceInd, new_face_kids);
  faces.set_area_host(faceInd, 0.0);
  faces.set_leaf_mask(faceInd, true);
  faces.decrement_leaves();
}

template <typename Geo>
void FaceDivider<Geo,QuadFace>::divide(const Index faceInd, Vertices<Coords<Geo>>& verts,
  Edges& edges, Faces<QuadFace,Geo>& faces, Coords<Geo>& physFaces, Coords<Geo>& lagFaces)
{
  LPM_ASSERT(faceInd < faces.nh());

  LPM_REQUIRE_MSG(faces.n_max() >= faces.nh() + 4, "Faces::divide error: not enough memory.");
  LPM_REQUIRE_MSG(!faces.has_kids_host(faceInd), "Faces::divide error: called on previously divided face.");

  ko::View<Index[4][4], Host> new_face_edge_inds("new_face_edge_inds");  // (child face index, edge index)
  ko::View<Index[4][4], Host> new_face_vert_inds("new_face_vert_inds");  // (child face index, vertex index)

  // for debugging, set to invalid value
  for (int i=0; i<4; ++i) {
    for (int j=0; j<4; ++j) {
      new_face_vert_inds(i,j) = constants::NULL_IND;
      new_face_edge_inds(i,j) = constants::NULL_IND;
    }
  }
  /// pull data from parent
  auto parent_vert_inds = ko::subview(faces._hostverts, faceInd, ko::ALL());
  auto parent_edge_inds = ko::subview(faces._hostedges, faceInd, ko::ALL());
  // determine child face indices
  const Index face_insert_pt = faces.nh();
  ko::View<Index[4], Host> new_face_kids("new_face_kids");
  for (int i=0; i<4; ++i) {
    new_face_kids(i) = face_insert_pt+i;
  }

  /// connect parent vertices to child faces
  for (int i=0; i<4; ++i) {
    new_face_vert_inds(i,i) = parent_vert_inds(i);
  }

  ko::View<Index[2],Host> edge_kids("edge_kids");
  /// loop over parent edges
  for (int i=0; i<4; ++i) {
    const Index parent_edge = parent_edge_inds(i);
    if (edges.has_kids_host(parent_edge)) { // edge already divided
      edge_kids(0) = edges.kid_host(parent_edge,0);
      edge_kids(1) = edges.kid_host(parent_edge,1);
    }
    else { // divide edge
      edge_kids(0) = edges.nh();
      edge_kids(1) = edges.nh() + 1;

      edges.divide(parent_edge, verts);
    }

    /// connect child edges to child faces
    if (faces.edge_is_positive(faceInd, i, edges)) { /// edge has positive orientation
      new_face_edge_inds(i,i) = edge_kids(0);
      edges.set_left(edge_kids(0), new_face_kids(i));

      new_face_edge_inds((i+1)%4,i) = edge_kids(1);
      edges.set_left(edge_kids(1), new_face_kids((i+1)%4));
    }
    else { /// edge has negative orientation
      new_face_edge_inds(i,i) = edge_kids(1);
      edges.set_right(edge_kids(1), new_face_kids(i));

      new_face_edge_inds((i+1)%4,i) = edge_kids(0);
      edges.set_right(edge_kids(0), new_face_kids((i+1)%4));
    }
    const Index c1dest = edges.dest_host(edge_kids(0));
    new_face_vert_inds(i,(i+1)%4) = c1dest;
    new_face_vert_inds((i+1)%4,i) = c1dest;
  }

  /// special case for QuadFace: parent center becomes a vertex
  const Index parent_center_ind = faces._host_crd_inds(faceInd);
  ko::View<Real[Geo::ndim],Host> newcrd("newcrd");
  ko::View<Real[Geo::ndim],Host> newlagcrd("newlagcrd");
  for (int i=0; i<Geo::ndim; ++i) {
    newcrd(i) = faces.phys_crds->get_crd_component_host(parent_center_ind, i);
    newlagcrd(i) = faces.lag_crds->get_crd_component_host(parent_center_ind,i);
  }
  Index bndry_insert_pt = verts.phys_crds->nh();
  verts.phys_crds->insert_host(newcrd);
  verts.lag_crds->insert_host(newlagcrd);
  for (int i=0; i<4; ++i) {
    new_face_vert_inds(i,(i+2)%4) = bndry_insert_pt;
  }

  /// create new interior edges
  const Index edge_ins_pt = edges.nh();
  edges.insert_host(new_face_vert_inds(0,1), new_face_vert_inds(0,2), new_face_kids(0), new_face_kids(1));
  new_face_edge_inds(0,1) = edge_ins_pt;
  new_face_edge_inds(1,3) = edge_ins_pt;
  edges.insert_host(new_face_vert_inds(2,0), new_face_vert_inds(2,3), new_face_kids(3), new_face_kids(2));
  new_face_edge_inds(2,3) = edge_ins_pt+1;
  new_face_edge_inds(3,1) = edge_ins_pt+1;
  edges.insert_host(new_face_vert_inds(2,1), new_face_vert_inds(2,0), new_face_kids(1), new_face_kids(2));
  new_face_edge_inds(1,2) = edge_ins_pt+2;
  new_face_edge_inds(2,0) = edge_ins_pt+2;
  edges.insert_host(new_face_vert_inds(3,1), new_face_vert_inds(3,0), new_face_kids(0), new_face_kids(3));
  new_face_edge_inds(0,2) = edge_ins_pt+3;
  new_face_edge_inds(3,0) = edge_ins_pt+3;

  /// create new center coordinates
  ko::View<Real[4][Geo::ndim],Host> vert_crds("vert_crds");
  ko::View<Real[4][Geo::ndim],Host> vert_lag_crds("vert_lag_crds");
  ko::View<Real[4][Geo::ndim],Host> face_crds("face_crds");
  ko::View<Real[4][Geo::ndim],Host> face_lag_crds("face_lag_crds");
  ko::View<Real[4],Host> face_area("face_area");
  for (int i=0; i<4; ++i) { // loop over child faces
    auto ctr = ko::subview(face_crds,i,ko::ALL());
    auto lagctr = ko::subview(face_lag_crds,i,ko::ALL());
    for (int j=0; j<4; ++j) { // loop over vertices
      for (int k=0; k<Geo::ndim; ++k) { // loop over components
        vert_crds(j,k) = verts.phys_crds->get_crd_component_host(new_face_vert_inds(i,j),k);
        vert_lag_crds(j,k) = verts.lag_crds->get_crd_component_host(new_face_vert_inds(i,j),k);
      }
    }
    Geo::barycenter(ctr, vert_crds, 4);
    Geo::barycenter(lagctr, vert_lag_crds,4);
    face_area(i) = Geo::polygon_area(ctr, vert_crds, 4);
  }
  const Index face_crd_ins_pt = physFaces.nh();

  for (int i=0; i<4; ++i) {
    physFaces.insert_host(slice(face_crds,i));
    lagFaces.insert_host(slice(face_lag_crds,i));
    faces.insert_host(face_crd_ins_pt+i, ko::subview(new_face_vert_inds,i,ko::ALL()),
      ko::subview(new_face_edge_inds,i,ko::ALL()), faceInd, face_area(i));
  }

  /// remove parent from leaf computations
  faces.set_kids_host(faceInd, new_face_kids);
  faces.set_area_host(faceInd, 0.0);
  faces.set_leaf_mask(faceInd,true);
  faces.decrement_leaves();
}


/// ETI
template class Faces<TriFace,PlaneGeometry>;
template class Faces<TriFace,SphereGeometry>;
template class Faces<QuadFace,PlaneGeometry>;
template class Faces<QuadFace,SphereGeometry>;

template void Faces<TriFace,PlaneGeometry>::init_from_seed(const MeshSeed<TriHexSeed>& seed);
template void Faces<TriFace,SphereGeometry>::init_from_seed(const MeshSeed<IcosTriSphereSeed>& seed);
template void Faces<QuadFace,PlaneGeometry>::init_from_seed(const MeshSeed<QuadRectSeed>& seed);
template void Faces<QuadFace,SphereGeometry>::init_from_seed(const MeshSeed<CubedSphereSeed>& seed);

template struct FaceDivider<PlaneGeometry, TriFace>;
template struct FaceDivider<SphereGeometry, TriFace>;
template struct FaceDivider<PlaneGeometry, QuadFace>;
template struct FaceDivider<SphereGeometry, QuadFace>;

}
