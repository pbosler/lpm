#ifndef LPM_ERROR_IMPL_HPP
#define LPM_ERROR_IMPL_HPP

#include "lpm_error.hpp"
#include "lpm_geometry.hpp"

namespace Lpm {

template <typename V1, typename V2, typename V3, int Rank>
struct ComputeErrorFtor {
  V1 err;
  V2 appx;
  V3 exact;
  Int ndim;

  ComputeErrorFtor(const V1 er, const V2 ap, const V3 ex)
      : err(er), appx(ap), exact(ex), ndim(er.extent(1)) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index i) const {
    for (int j = 0; j < ndim; ++j) {
      err(i, j) = appx(i, j) - exact(i, j);
    }
  }
};

template <typename V1, typename V2, typename V3>
struct ComputeErrorFtor<V1, V2, V3, 1> {
  V1 err;
  V2 appx;
  V3 exact;
  Int ndim;

  ComputeErrorFtor(const V1 er, const V2 ap, const V3 ex)
      : err(er), appx(ap), exact(ex), ndim(1) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index i) const { err(i) = appx(i) - exact(i); }
};

template <typename V1, typename V2, int Rank>
struct ReduceErrorFtor {
  static_assert(V1::Rank == 1 or V1::Rank == 2, "rank 1 or rank 2 only.");
  V1 err;
  V2 exact;
  scalar_view_type weight;
  Int ndim;

  ReduceErrorFtor(const V1 er, const V2 ex, const scalar_view_type wt)
      : err(er), exact(ex), weight(wt), ndim(V1::Rank) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index i, ENormScalar& ll) const {
    const auto my_err = Kokkos::subview(err, i, Kokkos::ALL);
    const auto my_exact = Kokkos::subview(exact, i, Kokkos::ALL);
    const Real err_mag = (err.extent(1) == 2 ? PlaneGeometry::mag(my_err)
                                             : SphereGeometry::mag(my_err));
    const Real exact_mag = (err.extent(1) == 2 ? PlaneGeometry::mag(my_exact)
                                               : SphereGeometry::mag(my_exact));
    ll.l1num += err_mag * weight(i);
    ll.l1denom += exact_mag * weight(i);
    ll.l2num += square(err_mag) * weight(i);
    ll.l2denom += square(exact_mag) * weight(i);
    ll.linfnum = (err_mag > ll.linfnum ? err_mag : ll.linfnum);
    ll.linfdenom = (exact_mag > ll.linfdenom ? exact_mag : ll.linfdenom);
  }
};

template <typename V1, typename V2>
struct ReduceErrorFtor<V1, V2, 1> {
  V1 err;
  V2 exact;
  scalar_view_type weight;
  Int ndim;

  ReduceErrorFtor(const V1 er, const V2 ex, const scalar_view_type wt)
      : err(er), exact(ex), weight(wt), ndim(1) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const Index i, ENormScalar& ll) const {
    ll.l1num += abs(err(i)) * weight(i);
    ll.l1denom += abs(exact(i)) * weight(i);
    ll.l2num += square(err(i)) * weight(i);
    ll.l2denom += square(exact(i)) * weight(i);
    ll.linfnum = (abs(err(i)) > ll.linfnum ? abs(err(i)) : ll.linfnum);
    ll.linfdenom =
        (abs(exact(i)) > ll.linfdenom ? abs(exact(i)) : ll.linfdenom);
  }
};

template <typename V1, typename V2, typename V3>
void compute_error(const V1 err, const V2 appx, const V3 exact) {
  static_assert(V1::Rank == V2::Rank and V2::Rank == V3::Rank,
                "view ranks must match");
  LPM_REQUIRE(err.extent(0) == appx.extent(0) and
              appx.extent(0) == exact.extent(0));
  LPM_REQUIRE(err.extent(1) == appx.extent(1) and
              appx.extent(1) == exact.extent(1));

  Kokkos::parallel_for(
      err.extent(0), ComputeErrorFtor<V1, V2, V3, V1::Rank>(err, appx, exact));
}

template <typename V1, typename V2>
ENormScalar reduce_error(const V1 err, const V2 exact,
                         const scalar_view_type wt) {
  ENormScalar rval;
  static_assert(V1::Rank == V2::Rank, "view ranks must match");
  LPM_REQUIRE(err.extent(0) == exact.extent(0) and
              exact.extent(0) == wt.extent(0));
  LPM_REQUIRE(err.extent(1) == exact.extent(1));
  Kokkos::parallel_reduce("reduce error (vector)", err.extent(0),
                          ReduceErrorFtor<V1, V2, V1::Rank>(err, exact, wt),
                          ErrReducer<Host>(rval));
  return rval;
}

template <typename V1, typename V2, typename V3>
ErrNorms::ErrNorms(const V1 err, const V2 appx, const V3 exact,
                   const scalar_view_type wt) {
  compute_error(err, appx, exact);
  const auto rval = reduce_error(err, exact, wt);
  l1 = rval.l1num / rval.l1denom;
  l2 = sqrt(rval.l2num / rval.l2denom);
  linf = rval.linfnum / rval.linfdenom;
}

template <typename V1, typename V2>
ErrNorms::ErrNorms(const V1 err, const V2 exact, const scalar_view_type wt) {
  const auto rval = reduce_error(err, exact, wt);
  l1 = rval.l1num / rval.l1denom;
  l2 = sqrt(rval.l2num / rval.l2denom);
  linf = rval.linfnum / rval.linfdenom;
}

}  // namespace Lpm
#endif
