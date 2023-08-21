#ifndef LPM_NETCDF_READER_HPP
#define LPM_NETCDF_READER_HPP

#include "LpmConfig.h"
#include "lpm_netcdf.hpp"

namespace Lpm {
// namespace netCDF {

template <typename Geo> class Coords; // fwd decl

class NcReader {
  public:

    typedef typename ko::View<Index*>::HostMirror host_index_view;
    typedef typename scalar_view_type::HostMirror host_scalar_view;
    typedef typename ko::View<Real**>::HostMirror host_vector_view;

    virtual ~NcReader() {nc_close(ncid);}

    Int n_timesteps() const;

    virtual std::string info_string(const int tab_level=0) const;

    virtual Index n_points() const = 0;

  protected:
    explicit NcReader(const std::string& full_filename) :
      fname(full_filename),
      name_varid_map(),
      name_dimid_map() {
      nc_open(full_filename.c_str(), NC_NOWRITE, &ncid);
      inq_dims();
      inq_vars();
    }

    void inq_dims();
    void inq_vars();

    int ncid;
    int ndims;
    int nvars;
    std::string fname;
    int time_dimid;
    int time_varid;

    std::map<std::string, int> name_dimid_map;
    std::map<std::string, int> name_varid_map;


//     ko::View<Real**> getCrdView(const netCDF::NcVar& crd_var) const;
//     void fill_host_index_view(host_index_view& hv, const netCDF::NcVar& ind_var) const;
//     void fill_host_scalar_view(host_scalar_view& hv, const netCDF::NcVar& fvar) const;
//     void fill_host_vector_view(host_vector_view& hv, const netCDF::NcVar& fvar) const;
};

template <typename Geo>
class UnstructuredNcReader : public NcReader {
  using Layout = DataLayoutTraits<1>;
  public:
    UnstructuredNcReader(const std::string& full_filename);

    Coords<Geo> create_coords() const;

    Index n_points() const {return n_nodes;}

    std::string info_string(const int tab_level=0) const override;

  protected:
    Index n_nodes;
    int nodes_dimid;
    bool unpacked_coords;
    bool is_lat_lon;
    std::string coord_var_name;

    void find_coord_var();
};

// } // namespace netCDF
} // namespace Lpm

#endif
