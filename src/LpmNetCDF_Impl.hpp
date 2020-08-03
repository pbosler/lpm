#ifndef LPM_NETCDF_IMPL_HPP
#define LPM_NETCDF_IMPL_HPP

#include "LpmNetCDF.hpp"
#ifdef LPM_HAVE_NETCDF

namespace Lpm {

using namespace netCDF;

NcWriter::NcWriter(const std::string& filename) :
  fname(filename) {

  ///TODO: check valid filename (end in .nc)
  ncfile =
    std::unique_ptr<NcFile>(new netCDF::NcFile(fname, NcFile::replace, NcFile::nc4));

}

template <typename SeedType>
void NcWriter::writePolymesh(const std::shared_ptr<PolyMesh2d<SeedType>>& mesh) {
  NcDim crd_dim = ncfile->addDim("ndim", SeedType::geo::ndim);
  NcDim nvertices = ncfile->addDim("nverts", mesh->nvertsHost());
  NcDim nedges = ncfile->addDim("nedges", mesh->nedgesHost());
  NcDim nfaces = ncfile->addDim("nfaces", mesh->nfacesHost());
  NcDim nfaceverts = ncfile->addDim("nfaceverts", SeedType::nfaceverts);
  NcDim two = ncfile->addDim("two", 2);
  NcDim four = ncfile->addDim("four", 4);

  {
    /**
      Vertices
    */
    std::vector<NcDim> vert_dims = {nvertices, crd_dim};
    NcVar vert_phys = ncfile->addVar("phys_crds_verts", ncDouble, vert_dims);
    NcVar vert_lag = ncfile->addVar("lag_crds_verts", ncDouble, vert_dims);

    const auto physx = ko::subview(mesh->physVerts.getHostCrdView(), ko::ALL(), 0);
    const auto physy = ko::subview(mesh->physVerts.getHostCrdView(), ko::ALL(), 1);
    const auto physz = ko::subview(mesh->physVerts.getHostCrdView(), ko::ALL(), 0);
    const auto lagx = ko::subview(mesh->lagVerts.getHostCrdView(), ko::ALL(), 0);
    const auto lagy = ko::subview(mesh->lagVerts.getHostCrdView(), ko::ALL(), 1);
    const auto lagz = ko::subview(mesh->lagVerts.getHostCrdView(), ko::ALL(), 0);
    if (SeedType::geo::ndim == 3) {
      const auto physz = ko::subview(mesh->physVerts.getHostCrdView(), ko::ALL(), 2);
      const auto lagz = ko::subview(mesh->lagVerts.getHostCrdView(), ko::ALL(), 2);
    }

    std::vector<size_t> start_ind(2), count(2);
    start_ind[0] = 0;
    start_ind[1] = 0;
    count[0] = 1;
    count[1] = SeedType::geo::ndim;
    for (Index i=0; i<mesh->nvertsHost(); ++i) {
      Real phys_xyz[3] = {physx(i), physy(i), (SeedType::geo::ndim == 3 ? physz(i) : 0.0)};
      Real lag_xyz[3] = {lagx(i), lagy(i), (SeedType::geo::ndim == 3 ? lagz(i) : 0.0)};
      start_ind[0] = i;
      vert_phys.putVar(start_ind, count, &phys_xyz[0]);
      vert_lag.putVar(start_ind, count, &lag_xyz[0]);
    }
  }

  {
    /**
      edges
    */
    NcVar edge_origs = ncfile->addVar("edge_origs", ncInt64, nedges);
    NcVar edge_dests = ncfile->addVar("edge_dests", ncInt64, nedges);
    NcVar edge_lefts = ncfile->addVar("edge_lefts", ncInt64, nedges);
    NcVar edge_rights = ncfile->addVar("edge_rights", ncInt64, nedges);
    NcVar edge_parents = ncfile->addVar("edge_parents", ncInt64, nedges);
    std::vector<NcDim> tree_dims = {nedges, two};
    NcVar edge_kids = ncfile->addVar("edge_kids", ncInt64, tree_dims);

    std::vector<size_t> scalar_start_ind(1), scalar_count(1);
    scalar_start_ind[0] = 0;
    scalar_count[0] = mesh->nedgesHost();
    const auto eo = mesh->edges.getOrigsHost();
    const auto ed = mesh->edges.getDestsHost();
    const auto el = mesh->edges.getLeftsHost();
    const auto er = mesh->edges.getRightsHost();
    const auto ep = mesh->edges.getParentsHost();
    const auto ek = mesh->edges.getKidsHost();

    edge_origs.putVar(scalar_start_ind, scalar_count, eo.data());
    edge_dests.putVar(scalar_start_ind, scalar_count, ed.data());
    edge_lefts.putVar(scalar_start_ind, scalar_count, el.data());
    edge_rights.putVar(scalar_start_ind, scalar_count, er.data());
    edge_parents.putVar(scalar_start_ind, scalar_count, ep.data());

    std::vector<size_t> tree_start_ind(2), tree_count(2);
    tree_start_ind[0] = 0;
    tree_start_ind[1] = 0;
    tree_count[0] = 1;
    tree_count[1] = 2;
    for (Index i = 0; i<mesh->nedgesHost(); ++i) {
      Index kids[2] = {ek(i,0), ek(i,1)};
      edge_kids.putVar(tree_start_ind, tree_count, &kids[0]);
    }
  }

  {
    /**
      faces
    */
    std::vector<NcDim> face_dims = {nfaces, crd_dim};
    NcVar face_phys = ncfile->addVar("phys_crds_faces", ncDouble, face_dims);
    NcVar face_lag = ncfile->addVar("lag_crds_faces", ncDouble, face_dims);
    NcVar face_area = ncfile->addVar("face_area", ncDouble, nfaces);
    NcVar face_mask = ncfile->addVar("face_mask", ncByte, nfaces);
    std::vector<NcDim> topo_dims = {nfaces, nfaceverts};
    NcVar face_edges = ncfile->addVar("face_edges", ncInt64, topo_dims);
    NcVar face_verts = ncfile->addVar("face_verts", ncInt64, topo_dims);
    NcVar face_centers = ncfile->addVar("face_centers", ncInt64, nfaces);
    NcVar face_tree_level = ncfile->addVar("face_tree_level", ncInt, nfaces);
    NcVar face_parents = ncfile->addVar("face_parents", ncInt64, nfaces);
    std::vector<NcDim> tree_dims = {nfaces, four};
    NcVar face_kids = ncfile->addVar("face_kids", ncInt64, tree_dims);

    const auto physx = ko::subview(mesh->physFaces.getHostCrdView(), ko::ALL(), 0);
    const auto physy = ko::subview(mesh->physFaces.getHostCrdView(), ko::ALL(), 1);
    const auto physz = ko::subview(mesh->physFaces.getHostCrdView(), ko::ALL(), 0);
    const auto lagx = ko::subview(mesh->lagFaces.getHostCrdView(), ko::ALL(), 0);
    const auto lagy = ko::subview(mesh->lagFaces.getHostCrdView(), ko::ALL(), 1);
    const auto lagz = ko::subview(mesh->lagFaces.getHostCrdView(), ko::ALL(), 0);
    if (SeedType::geo::ndim == 3) {
      const auto physz = ko::subview(mesh->physFaces.getHostCrdView(), ko::ALL(), 2);
      const auto lagz = ko::subview(mesh->lagFaces.getHostCrdView(), ko::ALL(), 2);
    }

    const auto fmask = mesh->faces.getMaskHost();
    const auto fverts = mesh->faces.getVertsHost();
    const auto fedges = mesh->faces.getEdgesHost();
    const auto fcenters = mesh->faces.getCentersHost();
    const auto flev = mesh->faces.getLevelsHost();
    const auto fp = mesh->faces.getParentsHost();
    const auto fkids = mesh->faces.getKidsHost();
    const auto farea = mesh->faces.getAreaHost();

    std::vector<size_t> start_ind(2), count(2), tree_count(2);
    std::vector<size_t> topo_count(2);
    start_ind[0] = 0;
    start_ind[1] = 0;
    count[0] = 1;
    count[1] = SeedType::geo::ndim;
    tree_count[0] = 1;
    tree_count[1] = 4;
    topo_count[0] = 1;
    topo_count[1] = SeedType::nfaceverts;
    for (Index i=0; i<mesh->nfacesHost(); ++i) {
      Real phys_xyz[3] = {physx(i), physy(i), (SeedType::geo::ndim == 3 ? physz(i) : 0.0)};
      Real lag_xyz[3] = {lagx(i), lagy(i), (SeedType::geo::ndim == 3 ? lagz(i) : 0.0)};
      start_ind[0] = i;
      face_phys.putVar(start_ind, count, &phys_xyz[0]);
      face_lag.putVar(start_ind, count, &lag_xyz[0]);

      Index fv[SeedType::nfaceverts];
      Index fe[SeedType::nfaceverts];
      Index fk[4];
      for (Short j=0; j<SeedType::nfaceverts; ++j) {
        fv[j] = fverts(i,j);
        fe[j] = fedges(i,j);
      }
      for (Short j=0; j<4; ++j) {
        fk[j] = fkids(i,j);
      }

      face_verts.putVar(start_ind, topo_count, &fv[0]);
      face_edges.putVar(start_ind, topo_count, &fe[0]);
      face_kids.putVar(start_ind, tree_count, &fk[0]);
    }
    std::vector<size_t> scalar_start_ind(1), scalar_count(1);
    scalar_start_ind[0] = 0;
    scalar_count[0] = mesh->nfacesHost();
    face_mask.putVar(scalar_start_ind, scalar_count, fmask.data());
    face_area.putVar(scalar_start_ind, scalar_count, farea.data());
    face_parents.putVar(scalar_start_ind, scalar_count, fp.data());
    face_centers.putVar(scalar_start_ind, scalar_count, fcenters.data());
    face_tree_level.putVar(scalar_start_ind, scalar_count, flev.data());
  }



  ncfile->putAtt("MeshSeed", SeedType::idString());
  ncfile->putAtt("FaceKind", SeedType::faceStr());
  ncfile->putAtt("Geom", SeedType::geo::idString());
  ncfile->putAtt("baseTreeDepth", ncInt, mesh->baseTreeDepth);
}

}
#endif
#endif
