#ifndef LPM_NETCDF_READER_HPP
#define LPM_NETCDF_READER_HPP

#include "LpmConfig.h"
#include "lpm_netcdf.hpp"
#include "lpm_field.hpp"
#include "lpm_logger.hpp"

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

    metadata_type get_field_metadata(const std::string& field_name);

  protected:
    explicit NcReader(const std::string& full_filename, const Comm& comm = Comm());

    int ncid;
    int ndims;
    int nvars;
    int natts;
    std::string fname;
    int time_dimid;
    int time_varid;

    void inq_dims();
    void inq_vars();

    Logger<> logger;

    std::map<std::string, int> name_dimid_map;
    std::map<std::string, int> name_varid_map;
};

/** Unstructured points reader.

  Expects a single list of `n_nodes` points.

  Coordinates may be packed in a single (d x n_nodes) array or unpacked in
  individual arrays of size 1 x n_nodes.

*/
template <typename Geo>
class UnstructuredNcReader : public NcReader {
  using Layout = NcDataLayoutTraits<1>;
  public:
    UnstructuredNcReader(const std::string& full_filename);

    Coords<Geo> create_coords();

    ScalarField<ParticleField> create_scalar_field(const std::string& name);
    ScalarField<ParticleField> create_scalar_field(const std::string& name, const int time_idx);
//     VectorField<Geo,ParticleField> create_vector_Field(const std::string& name) const;

    Index n_points() const override {return n_nodes;}

    std::string info_string(const int tab_level=0) const override;


  protected:
    Index n_nodes;
    bool unpacked_coords;
    bool is_lat_lon;
    std::vector<std::string> coord_var_names;

    void find_coord_vars();
};

// } // namespace netCDF
} // namespace Lpm

#endif
