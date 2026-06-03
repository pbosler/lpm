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
// ParticleField and VertexField imply that field data (e.g., velocity and
// vorticity) are colocated with grid points. EdgeFields are collocated with
// edge midpoints. FaceField can be used for cell-based quantities or for
// staggering.
enum FieldLocation { ParticleField, VertexField, EdgeField, FaceField };

// repeat the enums here to facilitate range-based iteration over the enum
static const FieldLocation AllFieldLocs[] = {ParticleField, VertexField,
                                             EdgeField, FaceField};

std::string field_loc_string(const FieldLocation& floc);

/**  Wraps scalar views with metadata to define scalar fields on
  particle sets and particle/panel meshes.
*/
template <FieldLocation FL>
struct ScalarField {
  typedef scalar_view_type view_type;
  static constexpr FieldLocation field_loc = FL;
  static constexpr int ndim                = 1;
  scalar_view_type view;
  typename scalar_view_type::HostMirror hview;

  /** Constructor.

    @param [in] mname name of scalar field
    @param [in] nmax number of values to allocate in memory
    @param [in] u unit string for scalar values
    @param [in] mdata additional metadata (key, value) string pairs
  */
  ScalarField(const std::string& mname, const Index nmax,
              const std::string& u       = "null_unit",
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

  /** Return a pair with the field's minimum and maximum values
    between indices (0, n-1).

    @param [in] n number of values to check
  */
  std::pair<Real, Real> range(const Index n) const;

  /** Count the number of INF or NAN values
    between indices (0, n-1).

    @param [in] n number of values to check
  */
  Index nan_count(const Index n) const;

  bool has_nan(const Index n) const;
};

/**  Wraps vector field views with metadata to define vector fields on
  particle sets and particle/panel meshes.
*/
template <typename Geo, FieldLocation FL>
struct VectorField {
  typedef typename Geo::vec_view_type view_type;
  static constexpr FieldLocation field_loc = FL;
  static constexpr int ndim                = Geo::ndim;
  typename Geo::vec_view_type view;
  typename Geo::vec_view_type::HostMirror hview;

  /** Constructor.

    @param [in] mname name of vector field
    @param [in] nmax number of values to allocate in memory
    @param [in] u unit string for vector values
    @param [in] mdata additional metadata (key, value) string pairs
  */
  VectorField(const std::string& mname, const Index nmax,
              const std::string& u       = "null_unit",
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

  /** Return a pair with the field's minimum and maximum magnitudes
    between indices (0, n-1).

    @param [in] n number of values to check
  */
  std::pair<Real, Real> range(const Index n) const;

  /** Count the number of INF or NAN values
    between indices (0, n-1).

    This function will count each component, so e.g., a
    vector v(i,:) = [0, inf, inf] will count as 2 nans.

    @param [in] n number of vector values to check
  */
  Index nan_count(const Index n) const;

  bool has_nan(const Index n) const;
};

}  // namespace Lpm

#endif
