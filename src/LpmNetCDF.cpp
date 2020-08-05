#include "LpmNetCDF.hpp"
#ifdef LPM_HAVE_NETCDF

namespace Lpm {

using namespace netCDF;

bool has_nc_file_extension(const std::string& filename) {
  const auto dot_pos = filename.find_last_of('.');
  const bool nc = filename.substr(dot_pos+1) == "nc";
  return nc;
}

ko::View<Real**> PolyMeshReader::getVertPhysCrdView() const {
  std::multimap<std::string,NcVar>::const_iterator var_it;
  var_it = vars.find("phys_crds_verts");
  const auto crd_var = var_it->second;
  return getCrdView(crd_var);
}

ko::View<Real**> PolyMeshReader::getVertLagCrdView() const {
  const auto var_it = vars.find("lag_crds_verts");
  const auto crd_var = var_it->second;
  return getCrdView(crd_var);
}

ko::View<Real**> PolyMeshReader::getFacePhysCrdView() const {
  const auto var_it = vars.find("phys_crds_faces");
  const auto crd_var = var_it->second;
  return getCrdView(crd_var);
}

ko::View<Real**> PolyMeshReader::getFaceLagCrdView() const {
  const auto var_it = vars.find("lag_crds_faces");
  const auto crd_var = var_it->second;
  return getCrdView(crd_var);
}

ko::View<Real**> NcReader::getCrdView(const NcVar& crd_var) const {
  const Index ncrds = crd_var.getDim(0).getSize();
  const Index ndim = crd_var.getDim(1).getSize();

  ko::View<Real**> result("crds", ncrds, ndim);
  auto hcrds = ko::create_mirror_view(result);

  std::vector<size_t> ind(2,0), count(2);
  count[0] = 1;
  count[1] = ndim;
  for (Index i=0; i<ncrds; ++i) {
    ind[0] = i;
    Real crdvals[3];
    crd_var.getVar(ind, count, &crdvals[0]);
    for (Short j=0; j<ndim; ++j) {
      hcrds(i,j) = crdvals[j];
    }
  }
  ko::deep_copy(result, hcrds);
  return result;
}

Index PolyMeshReader::nEdges() const {
  const auto dim_it = dims.find("nedges");
  return dim_it->second.getSize();
}

Index PolyMeshReader::nFaces() const {
  const auto dim_it = dims.find("nfaces");
  return dim_it->second.getSize();
}

void NcReader::fill_host_index_view(host_index_view& hv,
  const netCDF::NcVar& ind_var) const {
  std::vector<size_t> ind(1,0);
  std::vector<size_t> count(1,hv.extent(0));
  ind_var.getVar(ind, count, hv.data());
}

void NcReader::fill_host_scalar_view(host_scalar_view& hv,
  const netCDF::NcVar& fvar) const {
  std::vector<size_t> ind(1,0);
  std::vector<size_t> count(1,hv.extent(0));
  fvar.getVar(ind, count, hv.data());
}

void NcReader::fill_host_vector_view(host_vector_view& hv,
  const netCDF::NcVar& fvar) const {
  const Int ndim = dims.find("crd_dim")->second.getSize();
  std::vector<size_t> ind(2,0);
  const std::vector<size_t> count = {1,ndim};
  for (Index i=0; i<hv.extent(0); ++i) {
    ind[0] = i;
    Real vvals[3];
    fvar.getVar(ind, count, &vvals[0]);
    for (Short j=0; j<ndim; ++j) {
      hv(i,j) = vvals[j];
    }
  }
}

void PolyMeshReader::fill_origs(host_index_view& hv) const {
  const auto var_it = vars.find("edge_origs");
  fill_host_index_view(hv, var_it->second);
}

void PolyMeshReader::fill_dests(host_index_view& hv) const {
  const auto var_it = vars.find("edge_dests");
  fill_host_index_view(hv, var_it->second);
}

void PolyMeshReader::fill_lefts(host_index_view& hv) const {
  const auto var_it = vars.find("edge_lefts");
  fill_host_index_view(hv, var_it->second);
}

void PolyMeshReader::fill_rights(host_index_view& hv) const {
  const auto var_it = vars.find("edge_rights");
  fill_host_index_view(hv, var_it->second);
}

void PolyMeshReader::fill_edge_tree(host_index_view& hv,
  typename ko::View<Index*[2]>::HostMirror& hk, Index& nleaves) const {
  const auto pvar_it = vars.find("edge_parents");
  const auto kvar_it = vars.find("edge_kids");
  fill_host_index_view(hv, pvar_it->second);

  nleaves = 0;
  std::vector<size_t> ind(2,0);
  const std::vector<size_t> count = {1,2};
  for (Index i=0; i<pvar_it->second.getDim(0).getSize(); ++i) {
    ind[0] = i;
    Index kids[2];
    kvar_it->second.getVar(ind, count, &kids[0]);
    for (Short j=0; j<2; ++j) {
      hk(i,j) = kids[j];
    }
    if (kids[0] == NULL_IND && kids[1] == NULL_IND) ++nleaves;
  }
}

Int PolyMeshReader::getTreeDepth() const {
  const auto att = ncfile->getAtt("baseTreeDepth");
  Int result;
  att.getValues(&result);
  return result;
}

void PolyMeshReader::fill_facemask(host_mask_view& hv) const {
  const auto var_it = vars.find("face_mask");
  const std::vector<size_t> start(1,0);
  const std::vector<size_t> count(1,hv.extent(0));
  var_it->second.getVar(start,count,hv.data());
}

void PolyMeshReader::fill_face_connectivity(host_topo_view_tri& faceverts,
  host_topo_view_tri& faceedges) const {
  const auto vit = vars.find("face_verts");
  const auto eit = vars.find("face_edges");
  const Int nfaceverts = dims.find("nfaceverts")->second.getSize();

  assert(nfaceverts == 3);

  std::vector<size_t> ind(2,0);
  const std::vector<size_t> count = {1,nfaceverts};
  const auto vert_var = vit->second;
  const auto edge_var = eit->second;
  for (Index i=0; i<faceverts.extent(0); ++i) {
    ind[0] = i;
    Index verts[nfaceverts];
    Index edges[nfaceverts];
    vert_var.getVar(ind, count, &verts[0]);
    edge_var.getVar(ind, count, &edges[0]);
    for (Short j=0; j<nfaceverts; ++j) {
      faceverts(i,j) = verts[j];
      faceedges(i,j) = edges[j];
    }
  }
}

void PolyMeshReader::fill_face_connectivity(host_topo_view_quad& faceverts,
  host_topo_view_quad& faceedges) const {
  const auto vit = vars.find("face_verts");
  const auto eit = vars.find("face_edges");
  const Int nfaceverts = dims.find("nfaceverts")->second.getSize();

  assert(nfaceverts == 4);

  std::vector<size_t> ind(2,0);
  const std::vector<size_t> count = {1,nfaceverts};
  const auto vert_var = vit->second;
  const auto edge_var = eit->second;
  for (Index i=0; i<faceverts.extent(0); ++i) {
    ind[0] = i;
    Index verts[nfaceverts];
    Index edges[nfaceverts];
    vert_var.getVar(ind, count, &verts[0]);
    edge_var.getVar(ind, count, &edges[0]);
    for (Short j=0; j<nfaceverts; ++j) {
      faceverts(i,j) = verts[j];
      faceedges(i,j) = edges[j];
    }
  }
}

void PolyMeshReader::fill_face_centers(host_index_view& hv) const {
  const auto var_it = vars.find("face_centers");
  fill_host_index_view(hv, var_it->second);
}

void PolyMeshReader::fill_face_levels(host_index_view& hv) const {
  const auto var_it = vars.find("face_tree_level");
  fill_host_index_view(hv, var_it->second);
}

void PolyMeshReader::fill_face_tree(host_index_view& hp,
  typename ko::View<Index*[4]>::HostMirror& hk, Index& nleaves) const {
  const auto pvar_it = vars.find("face_parents");
  const auto kvar_it = vars.find("face_kids");
  fill_host_index_view(hp, pvar_it->second);

  nleaves = 0;
  std::vector<size_t> ind(2,0);
  const std::vector<size_t> count = {1,4};
  const auto kvar = kvar_it->second;
  for (Index i=0; i<hp.extent(0); ++i) {
    ind[0] = i;
    Index kids[4];
    kvar.getVar(ind, count, &kids[0]);
    Short kid_counter = 0;
    for (Short j=0; j<4; ++j) {
      hk(i,j) = kids[j];
      if (kids[j] != NULL_IND) ++kid_counter;
    }
    if (kid_counter == 0) ++nleaves;
  }
}

void PolyMeshReader::fill_face_area(typename scalar_view_type::HostMirror& hv) const {
  const auto it = vars.find("face_area");
  fill_host_scalar_view(hv, it->second);
}

}
#endif
