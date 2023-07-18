#ifndef LPM_FIELD_HPP
#define LPM_FIELD_HPP

#include <map>
#include <string>

#include "LpmConfig.h"
#include "lpm_assert.hpp"
#include "lpm_geometry.hpp"

namespace Lpm {

// define the locations where field data "live" in the sense of numerical
// methods
// ParticleField and VertexField imply that field data (e.g., velocity and vorticity) are
// colocated with grid points.
// EdgeFields are collocated with edge midpoints.
// FaceField can be used for cell-based quantities or for staggering.
enum FieldLocation { ParticleField, VertexField, EdgeField, FaceField };

// repeat the enums here to facilitate range-based iteration over the enum
static const FieldLocation AllFieldLocs[] = {ParticleField, VertexField,
                                             EdgeField, FaceField};

typedef std::map<std::string, std::string> metadata_type;

std::string field_loc_string(const FieldLocation& floc);

template <FieldLocation FL>
struct ScalarField {
  typedef scalar_view_type view_type;
  static constexpr FieldLocation field_loc = FL;
  static constexpr int ndim = 1;
  scalar_view_type view;
  typename scalar_view_type::HostMirror hview;

  ScalarField(
      const std::string& mname, const Index nmax,
      const std::string& u = "null_unit",
      const metadata_type& mdata = metadata_type())
      : name(mname), view(mname, nmax), units(u), metadata(mdata) {
    metadata.emplace("name", mname);
    metadata.emplace("location", field_loc_string(FL));
    metadata.emplace("units", u);
    hview = ko::create_mirror_view(view);
  }

  KOKKOS_INLINE_FUNCTION
  Real operator()(const Index i) const { return view(i); }

  std::string name;
  std::string units;
  metadata_type metadata;

  void update_device() const { ko::deep_copy(view, hview); }

  void update_host() const { ko::deep_copy(hview, view); }

  std::string info_string(const int tab_level = 0) const;
};

template <typename Geo, FieldLocation FL>
struct VectorField {
  typedef typename Geo::vec_view_type view_type;
  static constexpr FieldLocation field_loc = FL;
  static constexpr int ndim = Geo::ndim;
  typename Geo::vec_view_type view;
  typename Geo::vec_view_type::HostMirror hview;

  VectorField(
      const std::string& mname, const Index nmax,
      const std::string& u = "null_unit",
      const metadata_type& mdata = metadata_type())
      : name(mname), view(mname, nmax), units(u), metadata(mdata) {
    metadata.emplace("name", mname);
    metadata.emplace("location", field_loc_string(FL));
    metadata.emplace("units", u);
    hview = ko::create_mirror_view(view);
  }

  std::string name;
  std::string units;
  metadata_type metadata;

  KOKKOS_INLINE_FUNCTION
  Real operator()(const Index i, const Int j) const { return view(i, j); }

  void update_device() const { ko::deep_copy(view, hview); }

  void update_host() const { ko::deep_copy(hview, view); }

  std::string info_string(const int tab_level = 0) const;
};

}  // namespace Lpm

#endif
