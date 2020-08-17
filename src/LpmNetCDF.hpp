#ifndef LPM_NETCDF_HPP
#define LPM_NETCDF_HPP

#include "LpmConfig.h"
#ifdef LPM_HAVE_NETCDF

#include "LpmDefs.hpp"
#include "LpmPolyMesh2d.hpp"
#include "LpmEdges.hpp"
#include <netcdf>
#include <memory>
#include <map>

namespace Lpm {

bool has_nc_file_extension(const std::string& filename);

typedef std::conditional<std::is_same<int,Index>::value,
  netCDF::NcInt, netCDF::NcInt64>::type nc_index_type;
typedef std::conditional<std::is_same<double,Real>::value,
  netCDF::NcDouble, netCDF::NcFloat>::type nc_real_type;

class NcWriter {
  public:
    NcWriter(const std::string& filename);

    template <typename SeedType>
    void writePolymesh(const std::shared_ptr<PolyMesh2d<SeedType>>& mesh);

    template <typename ViewType>
    void writeScalarField(const ViewType& s,
      const FieldKind& fk, const std::string& name="", const std::string& units="null");

    template <typename ViewType>
    void writeVectorField(const ViewType& v,
      const FieldKind& fk, const std::string& name="", const std::string& units="null");
  protected:
    std::string fname;
    std::unique_ptr<netCDF::NcFile> ncfile;

    std::multimap<std::string,netCDF::NcDim> dims;
};

class NcReader {
  public:
    typedef typename ko::View<Index*>::HostMirror host_index_view;
    typedef typename scalar_view_type::HostMirror host_scalar_view;
    typedef typename ko::View<Real**>::HostMirror host_vector_view;

    NcReader(const std::string& filename);

  protected:
    std::string fname;
    std::unique_ptr<const netCDF::NcFile> ncfile;

    std::multimap<std::string, netCDF::NcDim> dims;
    std::multimap<std::string, netCDF::NcVar> vars;
    std::multimap<std::string, netCDF::NcGroupAtt> atts;

    ko::View<Real**> getCrdView(const netCDF::NcVar& crd_var) const;
    void fill_host_index_view(host_index_view& hv, const netCDF::NcVar& ind_var) const;
    void fill_host_scalar_view(host_scalar_view& hv, const netCDF::NcVar& fvar) const;
    void fill_host_vector_view(host_vector_view& hv, const netCDF::NcVar& fvar) const;
};

class PolyMeshReader : NcReader {
  public:
    typedef typename ko::View<Index*>::HostMirror host_index_view;
    typedef typename mask_view_type::HostMirror host_mask_view;
    typedef typename ko::View<Index*[3]>::HostMirror host_topo_view_tri;
    typedef typename ko::View<Index*[4]>::HostMirror host_topo_view_quad;

    PolyMeshReader(const std::string& filename) : NcReader(filename) {}

    Int getTreeDepth() const;

    ko::View<Real**> getVertPhysCrdView() const;
    ko::View<Real**> getVertLagCrdView() const;

    Index nEdges() const;
    void fill_origs(host_index_view& hv) const;
    void fill_dests(host_index_view& hv) const;
    void fill_lefts(host_index_view& hv) const;
    void fill_rights(host_index_view& hv) const;
    void fill_edge_tree(host_index_view& hv,
      typename ko::View<Index*[2]>::HostMirror& hk, Index& nleaves) const;


    Index nFaces() const;
    void fill_facemask(host_mask_view& hv) const;
    void fill_face_connectivity(host_topo_view_tri& faceverts,
      host_topo_view_tri& faceedges) const;
    void fill_face_connectivity(host_topo_view_quad& faceverts,
      host_topo_view_quad& faceedges) const;
    void fill_face_centers(host_index_view& hv) const;
    void fill_face_levels(host_index_view& hv) const;
    void fill_face_tree(host_index_view& hp,
      typename ko::View<Index*[4]>::HostMirror& hk, Index& nleaves) const;
    void fill_face_area(typename scalar_view_type::HostMirror& hv) const;

    ko::View<Real**> getFacePhysCrdView() const;
    ko::View<Real**> getFaceLagCrdView() const;



};

}
#endif
#endif
