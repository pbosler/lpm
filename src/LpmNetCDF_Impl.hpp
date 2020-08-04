#ifndef LPM_NETCDF_IMPL_HPP
#define LPM_NETCDF_IMPL_HPP

#include "LpmConfig.h"
#ifdef LPM_HAVE_NETCDF
#include "LpmNetCDF.hpp"
#include <string>
#include <sstream>
#include <exception>

namespace Lpm {

using namespace netCDF;

NcWriter::NcWriter(const std::string& filename) :
  fname(filename) {
  if (has_nc_file_extension(fname)) {
    ncfile =
      std::unique_ptr<NcFile>(new NcFile(fname, NcFile::replace, NcFile::nc4));
  }
  else {
    std::ostringstream ss;
    ss << "NcWriter::NcWriter error: file "
      << filename << " has invalid extension (must be .nc)";
    throw std::runtime_error(ss.str());
  }
}

NcReader::NcReader(const std::string& filename) : fname(filename) {
  if (has_nc_file_extension(fname)) {
    ncfile =
      std::unique_ptr<const NcFile>(new NcFile(fname, NcFile::read));

    dims = ncfile->getDims();
    vars = ncfile->getVars();
    atts = ncfile->getAtts();
  }
  else {
    std::ostringstream ss;
    ss << "NcWriter::NcWriter error: file "
      << filename << " has invalid extension (must be .nc)";
    throw std::runtime_error(ss.str());
  }
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

  dims.emplace("crd_dim", crd_dim);
  dims.emplace("nverts", nvertices);
  dims.emplace("nedges", nedges);
  dims.emplace("nfaces", nfaces);
  dims.emplace("nfaceverts", nfaceverts);
  dims.emplace("two", two);
  dims.emplace("four", four);

  {
    /**
      Vertices
    */
    std::vector<NcDim> vert_dims = {nvertices, crd_dim};
    NcVar vert_phys = ncfile->addVar("phys_crds_verts", nc_real_type(), vert_dims);
    NcVar vert_lag = ncfile->addVar("lag_crds_verts", nc_real_type(), vert_dims);

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
    NcVar edge_origs = ncfile->addVar("edge_origs", nc_index_type(), nedges);
    NcVar edge_dests = ncfile->addVar("edge_dests", nc_index_type(), nedges);
    NcVar edge_lefts = ncfile->addVar("edge_lefts", nc_index_type(), nedges);
    NcVar edge_rights = ncfile->addVar("edge_rights", nc_index_type(), nedges);
    NcVar edge_parents = ncfile->addVar("edge_parents", nc_index_type(), nedges);
    std::vector<NcDim> tree_dims = {nedges, two};
    NcVar edge_kids = ncfile->addVar("edge_kids", nc_index_type(), tree_dims);

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
      tree_start_ind[0] = i;
      Index kids[2] = {ek(i,0), ek(i,1)};
      edge_kids.putVar(tree_start_ind, tree_count, &kids[0]);
    }
  }

  {
    /**
      faces
    */
    std::vector<NcDim> face_dims = {nfaces, crd_dim};
    NcVar face_phys = ncfile->addVar("phys_crds_faces", nc_real_type(), face_dims);
    NcVar face_lag = ncfile->addVar("lag_crds_faces", nc_real_type(), face_dims);
    NcVar face_area = ncfile->addVar("face_area", nc_real_type(), nfaces);
    NcVar face_mask = ncfile->addVar("face_mask", ncByte, nfaces);
    std::vector<NcDim> topo_dims = {nfaces, nfaceverts};
    NcVar face_edges = ncfile->addVar("face_edges", nc_index_type(), topo_dims);
    NcVar face_verts = ncfile->addVar("face_verts", nc_index_type(), topo_dims);
    NcVar face_centers = ncfile->addVar("face_centers", nc_index_type(), nfaces);
    NcVar face_tree_level = ncfile->addVar("face_tree_level", ncInt, nfaces);
    NcVar face_parents = ncfile->addVar("face_parents", nc_index_type(), nfaces);
    std::vector<NcDim> tree_dims = {nfaces, four};
    NcVar face_kids = ncfile->addVar("face_kids", nc_index_type(), tree_dims);

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

template <typename ViewType>
void NcWriter::writeScalarField(const ViewType& s,
  const FieldKind& fk, const std::string& name, const std::string& units) {

  const auto hs = ko::create_mirror_view(s);
  ko::deep_copy(hs,s);

  std::multimap<std::string, NcDim>::iterator dim_it;
  switch (fk) {
    case (VertexField) : {
      dim_it = dims.find("nverts");
      break;
    }
    case (EdgeField) : {
      dim_it = dims.find("nedges");
      break;
    }
    case (FaceField) : {
      dim_it = dims.find("nfaces");
      break;
    }
  }
  NcVar scalar_var = ncfile->addVar((name.empty() ? s.label() : name),
    nc_real_type(), dim_it->second);
  scalar_var.putAtt("units",units);
  std::vector<size_t> ind(1);
  for (Index i=0; i<dim_it->second.getSize(); ++i) {
    ind[0] = i;
    scalar_var.putVar(ind, hs.data() + i);
  }
}

template <typename ViewType>
void NcWriter::writeVectorField(const ViewType& v, const FieldKind& fk, const std::string& name,
  const std::string& units) {

  const auto hv = ko::create_mirror_view(v);
  ko::deep_copy(hv,v);

  std::multimap<std::string, NcDim>::iterator dim_it, crd_it;
  switch (fk) {
    case (VertexField) : {
      dim_it = dims.find("nverts");
      break;
    }
    case (EdgeField) : {
      dim_it = dims.find("nedges");
      break;
    }
    case (FaceField) : {
      dim_it = dims.find("nfaces");
      break;
    }
  }
  crd_it = dims.find("crd_dim");

  std::vector<NcDim> vardims = {dim_it->second, crd_it->second};
  NcVar vec_var = ncfile->addVar((name.empty() ? v.label() : name),
    nc_real_type(), vardims);
  vec_var.putAtt("units", units);
  std::vector<size_t> ind(2), count(2);
  ind[0] = 0;
  ind[1] = 0;
  count[0] = 1;
  const Int ndim = crd_it->second.getSize();
  count[1] = ndim;
  for (Index i=0; i<dim_it->second.getSize(); ++i) {
    ind[0] = i;
    Real vecvals[3] = {hv(i,0), hv(i,1), (ndim == 3 ? hv(i,2) : 0.0)};
    vec_var.putVar(ind, count, &vecvals[0]);
  }
}



}
#endif
#endif
