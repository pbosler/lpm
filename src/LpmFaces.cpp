#include "LpmFaces.hpp"
#include "LpmUtilities.hpp"
#ifdef LPM_HAVE_NETCDF
#include "LpmNetCDF.hpp"
#endif

namespace Lpm {

template <typename FaceKind>
void Faces<FaceKind>::insertHost(const Index ctr_ind, ko::View<Index*,Host> vertinds, ko::View<Index*,Host> edgeinds, const Index prt, const Real ar) {
  LPM_THROW_IF(_nh()+1 > _nmax, "Faces::insert error: not enough memory.");
  const Index ins = _nh();
  for (int i=0; i<FaceKind::nverts; ++i) {
    _hostverts(ins, i) = vertinds(i);
    _hostedges(ins, i) = edgeinds(i);
  }
  for (int i=0; i<4; ++i) {
    _hostkids(ins, i) = NULL_IND;
  }
  _hostcenters(ins) = ctr_ind;
  _hostparent(ins) = prt;
  _hostarea(ins) = ar;
  _hlevel(ins) = _hlevel(prt)+1;
  _hmask(ins) = false;
  _nh() += 1;
  _hnLeaves() += 1;
}

#ifdef LPM_HAVE_NETCDF
  template <typename FaceKind>
  Faces<FaceKind>::Faces(const PolyMeshReader& reader) : Faces(reader.nFaces()) {
    reader.fill_facemask(_hmask);
    reader.fill_face_connectivity(_hostverts, _hostedges);
    reader.fill_face_centers(_hostcenters);
    reader.fill_face_levels(_hlevel);
    reader.fill_face_tree(_hostparent, _hostkids, _hnLeaves());
    _nh() = _hmask.extent(0);
    reader.fill_face_area(_hostarea);
    updateDevice();
  }
#endif

template <typename FaceKind> template<typename SeedType>
void Faces<FaceKind>::initFromSeed(const MeshSeed<SeedType>& seed) {
  LPM_THROW_IF(_nmax < SeedType::nfaces, "Faces::initFromSeed error: not enough memory.");
  for (int i=0; i<SeedType::nfaces; ++i) {
    for (int j=0; j<SeedType::nfaceverts; ++j) {
      _hostverts(i,j) = seed.sfaceverts(i,j);
      _hostedges(i,j) = seed.sfaceedges(i,j);
    }
    _hostcenters(i) = i;
    _hostarea(i) = seed.faceArea(i);
    _hostparent(i) = NULL_IND;
    for (int j=0; j<4; ++j) {
      _hostkids(i, j) = NULL_IND;
    }
    _hlevel(i) = 0;
    _hmask(i) = false;
  }
  _nh() = SeedType::nfaces;
  _hnLeaves() = SeedType::nfaces;
  if (seed.idString() == "UnitDiskSeed") _hostkids(0,1) = 0;
}

template <typename FaceKind>
Real Faces<FaceKind>::surfAreaHost() const {
  Real result = 0;
  for (Index i=0; i<_nh(); ++i) {
    result += _hostarea(i);
  }
  return result;
}

template <typename FaceKind>
std::string Faces<FaceKind>::infoString(const std::string& label, const int& tab_level, const bool& dump_all) const {
  std::ostringstream oss;
  const auto idnt = indentString(tab_level);
  const auto bigidnt = indentString(tab_level+1);
  oss << idnt <<  "Faces " << label << " info: nh = (" << _nh() << ") of nmax = " << _nmax << " in memory; "
    << _hnLeaves() << " leaves." << std::endl;

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
      oss << "center = (" << _hostcenters(i) << ") ";
      oss << "level = (" << _hlevel(i) << ") ";
      oss << "parent = (" << _hostparent(i) << ") ";
      oss << "kids = (" << _hostkids(i,0) << "," << _hostkids(i,1) << ","
        << _hostkids(i,2) << "," << _hostkids(i,3) <<") ";
      oss << "area = (" << _hostarea(i) << ")";
      oss << std::endl;
    }
  }
  oss << bigidnt << "total area = " << surfAreaHost() << std::endl;
  return oss.str();
}

template <typename FaceKind>
void Faces<FaceKind>::setKids(const Index parent, const Index* kids) {
  assert(parent < _nh());
  for (int i=0; i<4; ++i) {
    _hostkids(parent, i) = kids[i];
  }
}

template <typename Geo>
void FaceDivider<Geo, TriFace>::divide(const Index faceInd, Coords<Geo>& physVerts, Coords<Geo>& lagVerts,
    Edges& edges, Faces<TriFace>& faces, Coords<Geo>& physFaces, Coords<Geo>& lagFaces){

  assert(faceInd < faces.nh());

  LPM_THROW_IF(faces.nMax() < faces.nh() + 4, "Faces::divide error: not enough memory.");
  LPM_THROW_IF(faces.hasKidsHost(faceInd), "Faces::divide error: called on previously divided face.");

  ko::View<Index[4][3], Host> newFaceEdgeInds("newFaceEdges");  // (child face index, edge index)
  ko::View<Index[4][3], Host> newFaceVertInds("newFaceVerts");  // (child face index, vertex index)

  // for debugging, set to invalid value
  for (int i=0; i<4; ++i) {
    for (int j=0; j<3; ++j) {
      newFaceVertInds(i,j) = NULL_IND;
      newFaceEdgeInds(i,j) = NULL_IND;
    }
  }
  /// pull data from parent face
  auto parentVertInds = ko::subview(faces._hostverts, faceInd, ko::ALL());
  auto parentEdgeInds = ko::subview(faces._hostedges, faceInd, ko::ALL());

  /// determine child face indices
  const Index face_insert_pt = faces.nh();
  ko::View<Index[4], Host> newFaceKids("newFaceKids");
  for (int i=0; i<4; ++i) {
    newFaceKids(i) = face_insert_pt+i;
  }
  /// connect parent vertices to child faces
  for (int i=0; i<3; ++i) {
    newFaceVertInds(i,i) = parentVertInds(i);
  }
  /// loop over parent edges, replace with child edges
  for (int i=0; i<3; ++i) {
    const Index parentEdge = parentEdgeInds(i);
    ko::View<Index[2], Host> edgekids("edgekids");
    if (edges.hasKidsHost(parentEdge)) { // edge already divided
      edgekids(0) = edges.getEdgeKidHost(parentEdge, 0);
      edgekids(1) = edges.getEdgeKidHost(parentEdge, 1);
    }
    else { // divide edge
      edgekids(0) = edges.nh();
      edgekids(1) = edges.nh() + 1;

      edges.divide<Geo>(parentEdge, physVerts, lagVerts);
    }

    // connect child edges to child faces
    if (faces.edgeIsPositive(faceInd, i, edges)) { // edge has positive orientation
      newFaceEdgeInds(i,i) = edgekids(0);
      edges.setLeft(edgekids(0), newFaceKids(i));

      newFaceEdgeInds((i+1)%3, i) = edgekids(1);
      edges.setLeft(edgekids(1), newFaceKids((i+1)%3));
    }
    else { // edge has negative orientation
      newFaceEdgeInds(i,i) = edgekids(1);
      edges.setRight(edgekids(1), newFaceKids(i));

      newFaceEdgeInds((i+1)%3, i) = edgekids(0);
      edges.setRight(edgekids(0), newFaceKids((i+1)%3));
    }

    const Index c1dest = edges.getDestHost(edgekids(0));
    if (i==0) {
      newFaceVertInds(0,1) = c1dest;
      newFaceVertInds(1,0) = c1dest;
      newFaceVertInds(3,2) = c1dest;
    }
    else if (i==1) {
      newFaceVertInds(1,2) = c1dest;
      newFaceVertInds(2,1) = c1dest;
      newFaceVertInds(3,0) = c1dest;
    }
    else {
      newFaceVertInds(2,0) = c1dest;
      newFaceVertInds(0,2) = c1dest;
      newFaceVertInds(3,1) = c1dest;
    }
  }

  // debug: check vertex connectivity
  for (int i=0; i<4; ++i) {
    for (int j=0; j<3; ++j) {
      LPM_THROW_IF(newFaceVertInds(i,j) == NULL_IND, "TriFace::divide error: vertex connectivity");
    }
  }

  /// create new interior edges
  const Index edge_ins_pt = edges.nh();
  for (int i=0; i<3; ++i) {
    newFaceEdgeInds(3,i) = edge_ins_pt+i;
  }
  newFaceEdgeInds(0,1) = edge_ins_pt+1;
  newFaceEdgeInds(1,2) = edge_ins_pt+2;
  newFaceEdgeInds(2,0) = edge_ins_pt;
  edges.insertHost(newFaceVertInds(2,1), newFaceVertInds(2,0), newFaceKids(3), newFaceKids(2));
  edges.insertHost(newFaceVertInds(0,2), newFaceVertInds(0,1), newFaceKids(3), newFaceKids(0));
  edges.insertHost(newFaceVertInds(1,0), newFaceVertInds(1,2), newFaceKids(3), newFaceKids(1));

  /// create new center coordinates
  ko::View<Real[3][Geo::ndim], Host> vertCrds("vertCrds");
  ko::View<Real[3][Geo::ndim], Host> vertLagCrds("vertLagCrds");
  ko::View<Real[4][Geo::ndim], Host> faceCrds("faceCrds");
  ko::View<Real[4][Geo::ndim], Host> faceLagCrds("faceLagCrds");
  ko::View<Real[4], Host> faceArea("faceArea");
  for (int i=0; i<4; ++i) { // loop over child Faces
    auto ctr = ko::subview(faceCrds, i, ko::ALL());
    auto lagctr = ko::subview(faceLagCrds, i, ko::ALL());
    for (int j=0; j<3; ++j) { // loop over vertices
      for (int k=0; k<Geo::ndim; ++k) { // loop over components
        vertCrds(j,k) = physVerts.getCrdComponentHost(newFaceVertInds(i,j),k);
        vertLagCrds(j,k) = lagVerts.getCrdComponentHost(newFaceVertInds(i,j),k);
      }
    }
    Geo::barycenter(ctr, vertCrds, 3);
    Geo::barycenter(lagctr, vertLagCrds, 3);
    faceArea(i) = Geo::polygonArea(ctr, vertCrds, 3);
  }

  /// create new child Faces
  const Index crd_ins_pt = physFaces.nh();
  for (int i=0; i<4; ++i) {
    physFaces.insertHost(slice(faceCrds,i));
    lagFaces.insertHost(slice(faceLagCrds,i));
    faces.insertHost(crd_ins_pt+i, ko::subview(newFaceVertInds,i, ko::ALL()), ko::subview(newFaceEdgeInds, i, ko::ALL()), faceInd, faceArea(i));
  }
  /// Remove parent from leaf computations
  faces.setKidsHost(faceInd, newFaceKids);
  faces.setAreaHost(faceInd, 0.0);
  faces.setMask(faceInd, true);
  faces.decrementnLeaves();
}

template <typename Geo>
void FaceDivider<Geo,QuadFace>::divide(const Index faceInd, Coords<Geo>& physVerts, Coords<Geo>& lagVerts,
  Edges& edges, Faces<QuadFace>& faces, Coords<Geo>& physFaces, Coords<Geo>& lagFaces)
{
  assert(faceInd < faces.nh());

  LPM_THROW_IF(faces.nMax() < faces.nh() + 4, "Faces::divide error: not enough memory.");
  LPM_THROW_IF(faces.hasKidsHost(faceInd), "Faces::divide error: called on previously divided face.");

  ko::View<Index[4][4], Host> newFaceEdgeInds("newFaceEdges");  // (child face index, edge index)
  ko::View<Index[4][4], Host> newFaceVertInds("newFaceVerts");  // (child face index, vertex index)

  // for debugging, set to invalid value
  for (int i=0; i<4; ++i) {
    for (int j=0; j<4; ++j) {
      newFaceVertInds(i,j) = NULL_IND;
      newFaceEdgeInds(i,j) = NULL_IND;
    }
  }
  /// pull data from parent
  auto parentVertInds = ko::subview(faces._hostverts, faceInd, ko::ALL());
  auto parentEdgeInds = ko::subview(faces._hostedges, faceInd, ko::ALL());
  // determine child face indices
  const Index face_insert_pt = faces.nh();
  ko::View<Index[4], Host> newFaceKids("newFaceKids");
  for (int i=0; i<4; ++i) {
    newFaceKids(i) = face_insert_pt+i;
  }

  /// connect parent vertices to child faces
  for (int i=0; i<4; ++i) {
    newFaceVertInds(i,i) = parentVertInds(i);
  }

  ko::View<Index[2],Host> edgekids("edgekids");
  /// loop over parent edges
  for (int i=0; i<4; ++i) {
    const Index parentEdge = parentEdgeInds(i);
    if (edges.hasKidsHost(parentEdge)) { // edge already divided
      edgekids(0) = edges.getEdgeKidHost(parentEdge,0);
      edgekids(1) = edges.getEdgeKidHost(parentEdge,1);
    }
    else { // divide edge
      edgekids(0) = edges.nh();
      edgekids(1) = edges.nh() + 1;

      edges.divide<Geo>(parentEdge, physVerts, lagVerts);
    }

    /// connect child edges to child faces
    if (faces.edgeIsPositive(faceInd, i, edges)) { /// edge has positive orientation
      newFaceEdgeInds(i,i) = edgekids(0);
      edges.setLeft(edgekids(0), newFaceKids(i));

      newFaceEdgeInds((i+1)%4,i) = edgekids(1);
      edges.setLeft(edgekids(1), newFaceKids((i+1)%4));
    }
    else { /// edge has negative orientation
      newFaceEdgeInds(i,i) = edgekids(1);
      edges.setRight(edgekids(1), newFaceKids(i));

      newFaceEdgeInds((i+1)%4,i) = edgekids(0);
      edges.setRight(edgekids(0), newFaceKids((i+1)%4));
    }
    const Index c1dest = edges.getDestHost(edgekids(0));
    newFaceVertInds(i,(i+1)%4) = c1dest;
    newFaceVertInds((i+1)%4,i) = c1dest;
  }

  /// special case for QuadFace: parent center becomes a vertex
  const Index parent_center_ind = faces.getCenterIndHost(faceInd);
  ko::View<Real[Geo::ndim],Host> newcrd("newcrd");
  ko::View<Real[Geo::ndim],Host> newlagcrd("newlagcrd");
  for (int i=0; i<Geo::ndim; ++i) {
    newcrd(i) = physFaces.getCrdComponentHost(parent_center_ind, i);
    newlagcrd(i) = lagFaces.getCrdComponentHost(parent_center_ind,i);
  }
  Index bndry_insert_pt = physVerts.nh();
  physVerts.insertHost(newcrd);
  lagVerts.insertHost(newlagcrd);
  for (int i=0; i<4; ++i) {
    newFaceVertInds(i,(i+2)%4) = bndry_insert_pt;
  }

  /// create new interior edges
  const Index edge_ins_pt = edges.nh();
  edges.insertHost(newFaceVertInds(0,1), newFaceVertInds(0,2), newFaceKids(0), newFaceKids(1));
  newFaceEdgeInds(0,1) = edge_ins_pt;
  newFaceEdgeInds(1,3) = edge_ins_pt;
  edges.insertHost(newFaceVertInds(2,0), newFaceVertInds(2,3), newFaceKids(3), newFaceKids(2));
  newFaceEdgeInds(2,3) = edge_ins_pt+1;
  newFaceEdgeInds(3,1) = edge_ins_pt+1;
  edges.insertHost(newFaceVertInds(2,1), newFaceVertInds(2,0), newFaceKids(1), newFaceKids(2));
  newFaceEdgeInds(1,2) = edge_ins_pt+2;
  newFaceEdgeInds(2,0) = edge_ins_pt+2;
  edges.insertHost(newFaceVertInds(3,1), newFaceVertInds(3,0), newFaceKids(0), newFaceKids(3));
  newFaceEdgeInds(0,2) = edge_ins_pt+3;
  newFaceEdgeInds(3,0) = edge_ins_pt+3;

  /// create new center coordinates
  ko::View<Real[4][Geo::ndim],Host> vertCrds("vertCrds");
  ko::View<Real[4][Geo::ndim],Host> vertLagCrds("vertLagCrds");
  ko::View<Real[4][Geo::ndim],Host> faceCrds("faceCrds");
  ko::View<Real[4][Geo::ndim],Host> faceLagCrds("faceLagCrds");
  ko::View<Real[4],Host> faceArea("faceArea");
  for (int i=0; i<4; ++i) { // loop over child faces
    auto ctr = ko::subview(faceCrds,i,ko::ALL());
    auto lagctr = ko::subview(faceLagCrds,i,ko::ALL());
    for (int j=0; j<4; ++j) { // loop over vertices
      for (int k=0; k<Geo::ndim; ++k) { // loop over components
        vertCrds(j,k) = physVerts.getCrdComponentHost(newFaceVertInds(i,j),k);
        vertLagCrds(j,k) = lagVerts.getCrdComponentHost(newFaceVertInds(i,j),k);
      }
    }
    Geo::barycenter(ctr, vertCrds, 4);
    Geo::barycenter(lagctr, vertLagCrds,4);
    faceArea(i) = Geo::polygonArea(ctr, vertCrds, 4);
  }
  const Index face_crd_ins_pt = physFaces.nh();

  for (int i=0; i<4; ++i) {
    physFaces.insertHost(slice(faceCrds,i));
    lagFaces.insertHost(slice(faceLagCrds,i));
    faces.insertHost(face_crd_ins_pt+i, ko::subview(newFaceVertInds,i,ko::ALL()),
      ko::subview(newFaceEdgeInds,i,ko::ALL()), faceInd, faceArea(i));
  }

  /// remove parent from leaf computations
  faces.setKidsHost(faceInd, newFaceKids);
  faces.setAreaHost(faceInd, 0.0);
  faces.setMask(faceInd,true);
  faces.decrementnLeaves();
}


void FaceDivider<CircularPlaneGeometry,QuadFace>::divide(const Index faceInd,
  Coords<CircularPlaneGeometry>& physVerts, Coords<CircularPlaneGeometry>& lagVerts,
  Edges& edges, Faces<QuadFace>& faces, Coords<CircularPlaneGeometry>& physFaces,
  Coords<CircularPlaneGeometry>& lagFaces) {

  assert(faceInd < faces.nh());

  LPM_THROW_IF(faces.nMax() < faces.nh() + 4, "Faces::divide error: not enough memory.");
  LPM_THROW_IF(faces.hasKidsHost(faceInd), "Faces::divide error: called on previously divided face.");

  ko::View<Index[4][4],Host> newFaceEdgeInds("newFaceEdgeInds");
  ko::View<Index[4][4],Host> newFaceVertInds("newFaceVertInds");

  // for debugging, set to invalid value
  for (int i=0; i<4; ++i) {
    for (int j=0; j<4; ++j) {
      newFaceVertInds(i,j) = NULL_IND;
      newFaceEdgeInds(i,j) = NULL_IND;
    }
  }
  /// pull data from parent
  auto parentVertInds = ko::subview(faces._hostverts, faceInd, ko::ALL());
  auto parentEdgeInds = ko::subview(faces._hostedges, faceInd, ko::ALL());

  const Index face_insert_pt = faces.nh();
  const Index face_crd_ins_pt = physFaces.nh();

  ko::View<Index[4],Host> newFaceKids("newFaceKids");
  for (Short i=0; i<4; ++i) {
    newFaceKids(i) = face_insert_pt+i;
  }

  /// special case for faceInd = 0
  if (faceInd == 0) {
    // assert(faces.getCenterIndHost(faceInd) == 0);
//     const Index vert_ins_pt = physVerts.nh();
//     const Index edge_ins_pt = edges.nh();
//
//     /**
//       Make new vertices at locations of 0,1,2,3; connect to parent edges
//
//         new vertex indices = vert_ins_pt + (0,1,2,3)
//     */
//     ko::View<Real[4][2],Host> special_physverts("special_physverts");
//     ko::View<Real[4][2],Host> special_lagverts("special_lagverts");
//     const Real r0 = CircularPlaneGeometry::mag(ko::subview(special_physverts, 0, ko::ALL));
//     for (Short i=0; i<4; ++i) {
//      for (Short j=0; j<2; ++j) {
//       special_physverts(i,j) = physVerts.getCrdComponentHost(i,j);
//       special_lagverts(i,j) = lagVerts.getCrdComponentHost(i,j);
//      }
//      physVerts.insertHost(ko::subview(special_physverts,i,ko::ALL));
//      lagVerts.insertHost(ko::subview(special_lagverts,i,ko::ALL));
//      edges.setOrig(parentEdgeInds(i), vert_ins_pt+i);
//      edges.setDest(parentEdgeInds(i), vert_ins_pt+(i+1)%4);
//     }
//
//     /**
//       divide parent edges
//
//         new vertex indices = vert_ins_pt + (4,5,6,7)
//         new edgde indices = edge_ins_pt + [(0,1), (2,3), (4,5), (6,7)]
//     */
//     for (Short i=0; i<4; ++i) {
//       edges.divide<CircularPlaneGeometry>(parentEdgeInds(i), physVerts, lagVerts);
//     }
//
//     /**
//       Move vertices 0,1,2,3 to half-radius positions
//     */
//     for (Short i=0; i<4; ++i) {
//       auto vcrd = ko::subview(special_physverts, i, ko::ALL);
//       const Real the = (i+1)*0.5*PI;
//       vcrd(0) = 0.5*r0*std::cos(the);
//       vcrd(1) = 0.5*r0*std::sin(the);
//       physVerts.setCrdsHost(i, vcrd);
//       lagVerts.setCrdsHost(i, vcrd);
//     }
//     /**
//       Put new vertices between new 0,1,2,3 positions
//
//         new vertex indices = vert_ins_pt + (8,9,10,11)
//         these do not connect to panel 0 -- it (always) only connects to vertices 0,1,2,3
//
//         create new edges from vertices (0,1,2,3) to new vertex indices
//         these edges connect to panel 0, but panel 0 does not connect to them.
//         new edge indices = edge_ins_pt + [(8,9),(10,11),(12,13),(14,15)]
//     */
//     ko::View<Real[2],Host> newvertcrds("newvertcrds");
//     for (Short i=0; i<4; ++i) {
//       const Real the = (2*i+3)%8 * 0.25*PI;
//       const Real rr = 0.5*r0;
//       newvertcrds(0) = rr*std::cos(the);
//       newvertcrds(1) = rr*std::sin(the);
//       physVerts.insertHost(newvertcrds);
//       lagVerts.insertHost(newvertcrds);
//       //              orig       dest             left    right
//       edges.insertHost(i,       vert_ins_pt+8+i,      0, face_insert_pt+2*i);
//       edges.insertHost(vert_ins_pt+8+1,  (i+1)%4,     0, face_insert_pt+2*i+1);
//     }
//
//     /**
//       create new edges from outer radius to new inner radius
//         new edge indices = edge_ins_pt + [(16,17,18,19,20,21,22,23)]
//     */
//     edges.insertHost(vert_ins_pt+8, vert_ins_pt+4, face_insert_pt+1, face_insert_pt,0); //16
//     edges.insertHost(vert_ins_pt+1,1, face_insert_pt+1,face_insert_pt+2,0); //17
//     edges.insertHost(vert_ins_pt+9, vert_ins_pt+5,face_insert_pt+3,face_insert_pt+2,0);//18
//     edges.insertHost(vert_ins_pt+2, 2, face_insert_pt+3, face_insert_pt+4,0);//19
//     edges.insertHost(vert_ins_pt+10,vert_ins_pt+6, face_insert_pt+5,face_insert_pt+4,0);//20
//     edges.insertHost(vert_ins_pt+3,3, face_insert_pt+5, face_insert_pt+6,0);//21
//     edges.insertHost(vert_ins_pt+11,vert_ins_pt+7,face_insert_pt+7,face_insert_pt+6,0);//22
//     edges.insertHost(vert_ins_pt, 0, face_insert_pt+7,face_insert_pt,0);//23
//
//     /**
//       create new panels around shrunken panel 0
//
//         new face indices = face_insert_pt + (0,1,2,3,5,6,7)
//         new faces are all leaves with parent = 0
//     */
//     const Real newfacearea = 0.5*(0.25*PI)*(square(r0) - square(0.5*r0));
//     ko::View<Index[8][4],Host> newFaceVertInds("newFaceVertInds");
//     ko::View<Index[8][4],Host> newFaceEdgeInds("newFaceEdgeInds");
//     {
//     newFaceVertInds(0,0) = vert_ins_pt+4;
//     newFaceVertInds(0,1) = vert_ins_pt+8;
//     newFaceVertInds(0,2) = 0;
//     newFaceVertInds(0,3) = vert_ins_pt;
//
//     newFaceEdgeInds(0,0) = edge_ins_pt+16;
//     newFaceEdgeInds(0,1) = edge_ins_pt+8;
//     newFaceEdgeInds(0,2) = edge_ins_pt+23;
//     newFaceEdgeInds(0,3) = edge_ins_pt;
//
//     newFaceVertInds(1,0) = vert_ins_pt+1;
//     newFaceVertInds(1,1) = 1;
//     newFaceVertInds(1,2) = vert_ins_pt+8;
//     newFaceVertInds(1,3) = vert_ins_pt+4;
//
//     newFaceEdgeInds(1,0) = edge_ins_pt+17;
//     newFaceEdgeInds(1,1) = edge_ins_pt+9;
//     newFaceEdgeInds(1,2) = edge_ins_pt+16;
//     newFaceEdgeInds(1,3) = edge_ins_pt+1;
//
//     newFaceVertInds(2,0) = vert_ins_pt+5;
//     newFaceVertInds(2,1) = vert_ins_pt+9;
//     newFaceVertInds(2,2) = 1;
//     newFaceVertInds(2,3) = vert_ins_pt+1;
//
//     newFaceEdgeInds(2,0) = edge_ins_pt+18;
//     newFaceEdgeInds(2,1) = edge_ins_pt+10;
//     newFaceEdgeInds(2,2) = edge_ins_pt+17;
//     newFaceEdgeInds(2,3) = edge_ins_pt+2;
//
//     newFaceVertInds(3,0) = vert_ins_pt+2;
//     newFaceVertInds(3,1) = 2;
//     newFaceVertInds(3,2) = vert_ins_pt+9;
//     newFaceVertInds(3,3) = vert_ins_pt+5;
//
//     newFaceEdgeInds(3,0) = edge_ins_pt+19;
//     newFaceEdgeInds(3,1) = edge_ins_pt+11;
//     newFaceEdgeInds(3,2) = edge_ins_pt+18;
//     newFaceEdgeInds(3,3) = edge_ins_pt+3;
//
//     newFaceVertInds(4,0) = vert_ins_pt+6;
//     newFaceVertInds(4,1) = vert_ins_pt+10;
//     newFaceVertInds(4,2) = 2;
//     newFaceVertInds(4,3) = vert_ins_pt+2;
//
//     newFaceEdgeInds(4,0) = edge_ins_pt+20;
//     newFaceEdgeInds(4,1) = edge_ins_pt+12;
//     newFaceEdgeInds(4,2) = edge_ins_pt+19;
//     newFaceEdgeInds(4,3) = edge_ins_pt+4;
//
//     newFaceVertInds(5,0) = vert_ins_pt+3;
//     newFaceVertInds(5,1) = 3;
//     newFaceVertInds(5,2) = vert_ins_pt+10;
//     newFaceVertInds(5,3) = vert_ins_pt+6;
//
//     newFaceEdgeInds(5,0) = edge_ins_pt+21;
//     newFaceEdgeInds(5,1) = edge_ins_pt+13;
//     newFaceEdgeInds(5,2) = edge_ins_pt+20;
//     newFaceEdgeInds(5,3) = edge_ins_pt+5;
//
//     newFaceVertInds(6,0) = vert_ins_pt+7;
//     newFaceVertInds(6,1) = vert_ins_pt+11;
//     newFaceVertInds(6,2) = 3;
//     newFaceVertInds(6,3) = vert_ins_pt+3;
//
//     newFaceEdgeInds(6,0) = edge_ins_pt+22;
//     newFaceEdgeInds(6,1) = edge_ins_pt+14;
//     newFaceEdgeInds(6,2) = edge_ins_pt+21;
//     newFaceEdgeInds(6,3) = edge_ins_pt+6;
//
//     newFaceVertInds(7,0) = vert_ins_pt;
//     newFaceVertInds(7,1) = 0;
//     newFaceVertInds(7,2) = vert_ins_pt+11;
//     newFaceVertInds(7,3) = vert_ins_pt+1;
//
//     newFaceEdgeInds(7,0) = edge_ins_pt+23;
//     newFaceEdgeInds(7,1) = edge_ins_pt+15;
//     newFaceEdgeInds(7,2) = edge_ins_pt+22;
//     newFaceEdgeInds(7,3) = edge_ins_pt+7;
//
//     }
//     for (Short i=0; i<8; ++i) {
//       faces.insertHost(face_crd_ins_pt+i, ko::subview(newFaceVertInds,i,ko::ALL),
//         ko::subview(newFaceEdgeInds,i,ko::ALL),0, newfacearea);
//     }
//     /**
//       create new face coordinates
//     */
//     ko::View<Real[2],Host> fcrd("fcrd");
//     const Real dth = 2*PI/8;
//     const Real rf = 0.75*r0;
//     for (Short i=0; i<8; ++i) {
//       const Real the = 0.5*PI + dth;
//       fcrd(0) = rf*std::cos(the);
//       fcrd(1) = rf*std::sin(the);
//       physFaces.insertHost(fcrd);
//       lagFaces.insertHost(fcrd);
//     }
//
//     /**
//       set new face 0 area
//       ensure that face 0 is always a leaf
//     */
//     faces.setAreaHost(0, square(0.5*r0)*PI);
//     faces._hlevel(0) += 1;
  }
  else {

    /// set kid indices & connect parent vertices to child faces
    for (Short i=0; i<4; ++i) {
      newFaceVertInds(i,i) = parentVertInds(i);
    }

    /// parent edge loop
    ko::View<Index[2],Host> edgekids("edgekids");
    for (Short i=0; i<4; ++i) {
      const Index parentEdge = parentEdgeInds(i);
      if (edges.hasKidsHost(parentEdge)) {
        edgekids(0) = edges.getEdgeKidHost(parentEdge, 0);
        edgekids(1) = edges.getEdgeKidHost(parentEdge, 1);
      }
      else {
        edgekids(0) = edges.nh();
        edgekids(1) = edges.nh()+1;

        edges.divide<CircularPlaneGeometry>(parentEdge, physVerts, lagVerts);
      }

      /// connect child faces to child edges
      if (faces.edgeIsPositive(faceInd, i, edges)) {
        newFaceEdgeInds(i,i) = edgekids(0);
        edges.setLeft(edgekids(0), newFaceKids(i));

        newFaceEdgeInds((i+1)%4,i) = edgekids(1);
        edges.setLeft(edgekids(1), newFaceKids((i+1)%4));
      }
      else {
        newFaceEdgeInds(i,i) = edgekids(1);
        edges.setRight(edgekids(1), newFaceKids(i));

        newFaceEdgeInds((i+1)%4,i) = edgekids(0);
        edges.setRight(edgekids(0), newFaceKids((i+1)%4));
      }
      const Index c1dest = edges.getDestHost(edgekids(0));
      newFaceVertInds(i,(i+1)%4) = c1dest;
      newFaceVertInds((i+1)%4,i) = c1dest;
    }

    /// special case for QuadFace: parent center becomes a vertex
    const Index parent_center_ind = faces.getCenterIndHost(faceInd);
    ko::View<Real[2],Host> newcrd("newcrd");
    ko::View<Real[2],Host> newlagcrd("newlagcrd");
    for (Short i=0; i<2; ++i) {
      newcrd(i) = physFaces.getCrdComponentHost(parent_center_ind, i);
      newlagcrd(i) = lagFaces.getCrdComponentHost(parent_center_ind,i);
    }
    const Index vert_ins_pt = physVerts.nh();
    physVerts.insertHost(newcrd);
    lagVerts.insertHost(newlagcrd);
    for (Short i=0; i<4; ++i) {
      newFaceVertInds(i,(i+2)%4) = vert_ins_pt;
    }

    /// create new interior edges
    const Index edge_ins_pt = edges.nh();
    edges.insertHost(newFaceVertInds(0,1), newFaceVertInds(0,2), newFaceKids(0), newFaceKids(1));
    newFaceEdgeInds(0,1) = edge_ins_pt;
    newFaceEdgeInds(1,3) = edge_ins_pt;
    edges.insertHost(newFaceVertInds(2,0), newFaceVertInds(2,3), newFaceKids(3), newFaceKids(2));
    newFaceEdgeInds(2,3) = edge_ins_pt+1;
    newFaceEdgeInds(3,1) = edge_ins_pt+1;
    edges.insertHost(newFaceVertInds(2,1), newFaceVertInds(2,0), newFaceKids(1), newFaceKids(2));
    newFaceEdgeInds(1,2) = edge_ins_pt+2;
    newFaceEdgeInds(2,0) = edge_ins_pt+2;
    edges.insertHost(newFaceVertInds(3,1), newFaceVertInds(3,0), newFaceKids(0), newFaceKids(3));
    newFaceEdgeInds(0,2) = edge_ins_pt+3;
    newFaceEdgeInds(3,0) = edge_ins_pt+3;

    /// create new center coordinates
    ko::View<Real[4][2],Host> vertCrds("vertCrds");
    ko::View<Real[4][2],Host> vertLagCrds("vertLagCrds");
    ko::View<Real[4][2],Host> faceCrds("faceCrds");
    ko::View<Real[4][2],Host> faceLagCrds("faceLagCrds");
    ko::View<Real[4],Host> faceArea("faceArea");
    for (int i=0; i<4; ++i) { // loop over child faces
      auto ctr = ko::subview(faceCrds,i,ko::ALL());
      auto lagctr = ko::subview(faceLagCrds,i,ko::ALL());
      for (int j=0; j<4; ++j) { // loop over vertices
        for (int k=0; k<2; ++k) { // loop over components
          vertCrds(j,k) = physVerts.getCrdComponentHost(newFaceVertInds(i,j),k);
          vertLagCrds(j,k) = lagVerts.getCrdComponentHost(newFaceVertInds(i,j),k);
        }
      }
      CircularPlaneGeometry::barycenter(ctr, vertCrds, 4);
      CircularPlaneGeometry::barycenter(lagctr, vertLagCrds,4);
      faceArea(i) = CircularPlaneGeometry::polygonArea(ctr, vertCrds, 4);
    }
    for (Short i=0; i<4; ++i) {
      physFaces.insertHost(slice(faceCrds,i));
      lagFaces.insertHost(slice(faceLagCrds,i));
      faces.insertHost(face_crd_ins_pt+i, ko::subview(newFaceVertInds,i,ko::ALL()),
        ko::subview(newFaceEdgeInds,i,ko::ALL()), faceInd, faceArea(i));
    }

    /// remove parent from leaf computations
    faces.setKidsHost(faceInd, newFaceKids);
    faces.setAreaHost(faceInd, 0.0);
    faces.setMask(faceInd,true);
    faces.decrementnLeaves();
  }
}

/// ETI
template class Faces<TriFace>;
template class Faces<QuadFace>;

template void Faces<TriFace>::initFromSeed(const MeshSeed<TriHexSeed>& seed);
template void Faces<TriFace>::initFromSeed(const MeshSeed<IcosTriSphereSeed>& seed);
template void Faces<QuadFace>::initFromSeed(const MeshSeed<QuadRectSeed>& seed);
template void Faces<QuadFace>::initFromSeed(const MeshSeed<CubedSphereSeed>& seed);
template void Faces<QuadFace>::initFromSeed(const MeshSeed<UnitDiskSeed>& seed);

template struct FaceDivider<PlaneGeometry, TriFace>;
template struct FaceDivider<SphereGeometry, TriFace>;
template struct FaceDivider<PlaneGeometry, QuadFace>;
template struct FaceDivider<SphereGeometry, QuadFace>;
template struct FaceDivider<CircularPlaneGeometry, QuadFace>;

}
