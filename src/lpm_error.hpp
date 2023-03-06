#ifndef LPM_ERROR_NORMS_HPP
#define LPM_ERROR_NORMS_HPP

#include <vector>

#include "Kokkos_Core.hpp"
#include "LpmConfig.h"
#include "lpm_geometry.hpp"
#include "util/lpm_tuple.hpp"

namespace Lpm {

struct ENormScalar {
  Real l1num, l1denom;
  Real l2num, l2denom;
  Real linfnum, linfdenom;

  KOKKOS_INLINE_FUNCTION
  ENormScalar() { init(); }

  KOKKOS_INLINE_FUNCTION
  void init() {
    l1num = ko::reduction_identity<Real>::sum();
    l1denom = ko::reduction_identity<Real>::sum();
    l2num = ko::reduction_identity<Real>::sum();
    l2denom = ko::reduction_identity<Real>::sum();
    linfnum = ko::reduction_identity<Real>::max();
    linfdenom = ko::reduction_identity<Real>::max();
  }
};

template <typename Space = Dev>
struct ErrReducer {
 public:
  typedef ErrReducer reducer;
  typedef ENormScalar value_type;
  typedef ko::View<value_type, Space> result_view_type;

 private:
  result_view_type value;
  bool references_scalar_v;

 public:
  KOKKOS_INLINE_FUNCTION
  ErrReducer(value_type& val) : value(&val), references_scalar_v(true) {}

  KOKKOS_INLINE_FUNCTION
  ErrReducer(const result_view_type& val)
      : value(val), references_scalar_v(false) {}

  KOKKOS_INLINE_FUNCTION
  void join(value_type& dest, const value_type& src) const {
    dest.l1num += src.l1num;
    dest.l1denom += src.l1denom;
    dest.l2num += src.l2num;
    dest.l2denom += src.l2denom;
    if (src.linfnum > dest.linfnum) dest.linfnum = src.linfnum;
    if (src.linfdenom > dest.linfdenom) dest.linfdenom = src.linfdenom;
  }

  KOKKOS_INLINE_FUNCTION
  void join(volatile value_type& dest, const volatile value_type& src) const {
    dest.l1num += src.l1num;
    dest.l1denom += src.l1denom;
    dest.l2num += src.l2num;
    dest.l2denom += src.l2denom;
    if (src.linfnum > dest.linfnum) dest.linfnum = src.linfnum;
    if (src.linfdenom > dest.linfdenom) dest.linfdenom = src.linfdenom;
  }

  KOKKOS_INLINE_FUNCTION
  void init(value_type& val) const { val.init(); }

  KOKKOS_INLINE_FUNCTION
  result_view_type view() const { return value; }

  KOKKOS_INLINE_FUNCTION
  value_type& reference() const { return *value.data(); }
};

/**
    Computes error norms defined as:

    l1 = (\sum_{i=1}^N abs(appx(i) - exact(i))*weight(i)) / (\sum_{i=1}^N
   abs(exact(i))*weight(i))

    l2 = \sqrt( \sum_{i=1}^N square(appx(i) - exact(i))*weight(i) / \sum_{i=1}^N
   square(exact(i))*weight(i))

    linf = max_{i} abs(appx(i)- exact(i)) / max_{i}abs(exact(i))
*/
struct ErrNorms {
  Real l1;
  Real l2;
  Real linf;

  ErrNorms(const Real l_1, const Real l_2, const Real l_i)
      : l1(l_1), l2(l_2), linf(l_i) {}

  std::string info_string(const std::string& label = "",
                          const int tab_level = 0) const;

  /**
    constructor; assumes reductions have been pre-computed and stored in an
    ErrNormScalar.
  */
  ErrNorms(const ENormScalar& err)
      : l1(err.l1num / err.l1denom),
        l2(std::sqrt(err.l2num / err.l2denom)),
        linf(err.linfnum / err.linfdenom) {}

  template <typename V1, typename V2, typename V3>
  ErrNorms(const V1 err, const V2 appx, const V3 exact,
           const scalar_view_type wt);

  template <typename V1, typename V2, typename V3>
  ErrNorms(const V1 err, const V2 appx, const V3 exact,
           const scalar_view_type wt, const mask_view_type mask);

  template <typename V1, typename V2>
  ErrNorms(const V1 err, const V2 exact, const scalar_view_type wt);

};

std::vector<Real> convergence_rates(const std::vector<Real>& dx,
                                    const std::vector<Real>& ex);

std::string convergence_table(const std::string dxlabel,
                              const std::vector<Real>& dx,
                              const std::string exlabel,
                              const std::vector<Real>& ex,
                              const std::vector<Real>& rate);

template <typename V1, typename V2>
ENormScalar reduce_error(const V1 err, const V2 exact,
                         const scalar_view_type wt);

template <typename V1, typename V2, typename V3>
void compute_error(const V1 err, const V2 appx, const V3 exact);

template <typename V1, typename V2, typename V3>
void compute_error(const V1 err, const V2 appx, const V3 exact, const mask_view_type mask);

}  // namespace Lpm
#endif
