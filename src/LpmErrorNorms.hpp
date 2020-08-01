#ifndef LPM_ERROR_NORMS_HPP
#define LPM_ERROR_NORMS_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmKokkosUtil.hpp"
#include "LpmUtilities.hpp"
#include "LpmGeometry.hpp"
#include "Kokkos_Core.hpp"

namespace Lpm {

struct ENormScalar {
    Real l1num, l1denom;
    Real l2num, l2denom;
    Real linfnum, linfdenom;

    KOKKOS_INLINE_FUNCTION
    ENormScalar() {init();}

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

template <typename Space=Dev>
struct ErrReducer {
    public:
    typedef ErrReducer reducer;
    typedef ENormScalar value_type;
    typedef ko::View<value_type,Space> result_view_type;

    private:
        result_view_type value;
        bool references_scalar_v;

    public:
    KOKKOS_INLINE_FUNCTION
    ErrReducer(value_type& val) : value(&val), references_scalar_v(true) {}

    KOKKOS_INLINE_FUNCTION
    ErrReducer(const result_view_type& val) : value(val), references_scalar_v(false) {}

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
    void init(value_type& val) const {
        val.init();
    }

    KOKKOS_INLINE_FUNCTION
    result_view_type view() const {return value;}

    KOKKOS_INLINE_FUNCTION
    value_type& reference() const {return *value.data();}
};


/**
    Computes error norms defined as follows:

    l1 = (\sum_{i=1}^N abs(appx(i) - exact(i))*weight(i)) / (\sum_{i=1}^N abs(exact(i))*weight(i))

    l2 = \sqrt( \sum_{i=1}^N square(appx(i) - exact(i))*weight(i) / \sum_{i=1}^N square(exact(i))*weight(i))

    linf = max_{i} abs(appx(i)- exact(i)) / max_{i}abs(exact(i))
*/
template <typename Space=Host>
struct ErrNorms {
    Real l1;
    Real l2;
    Real linf;

    ErrNorms(const Real l_1, const Real l_2, const Real l_i) : l1(l_1), l2(l_2), linf(l_i) {}

    std::string infoString(const std::string& label="", const int tab_level=0) const;

    /**
      constructor; assumes reductions have been pre-computed and stored in an ErrNormScalar.
    */
    ErrNorms(const ENormScalar& err) : l1(err.l1num/err.l1denom), l2(std::sqrt(err.l2num/err.l2denom)),
      linf(err.linfnum/err.linfdenom) {}

    /**
        Scalar field constructor; performs reductions (error data is precomputed).
    */
    ErrNorms(const scalar_view_type& er, const scalar_view_type& ex, const scalar_view_type& wt) : l1(0), l2(0), linf(0) {
        compute(er, ex, wt);
    }

  /**
    vector field constructor; computes error, then performs reductions
  */
  ErrNorms(const SphereGeometry::vec_view_type& err, const SphereGeometry::vec_view_type& appx,
    const SphereGeometry::vec_view_type& exact, const scalar_view_type& wt) : l1(0), l2(0), linf(0) {
      compute(err, appx, exact, wt);
    }

  /**
    scalar field constructor; computes error, then performs reductions
  */
  ErrNorms(scalar_view_type& err, const scalar_view_type& appx, const scalar_view_type& exact, const scalar_view_type& wt):
    l1(0), l2(0), linf(0) {
    compute(err, appx, exact, wt);
  }

  void compute(const SphereGeometry::vec_view_type& err, const SphereGeometry::vec_view_type& appx,
    const SphereGeometry::vec_view_type& exct, const scalar_view_type& wt);

  void compute(scalar_view_type& err, const scalar_view_type& appx,
    const scalar_view_type& exact, const scalar_view_type& wt);

  void compute(const SphereGeometry::vec_view_type& err, const SphereGeometry::vec_view_type& exact,
    const scalar_view_type& wt);

  void compute(const scalar_view_type& err, const scalar_view_type& exact, const scalar_view_type& wt);
};



}
#endif
