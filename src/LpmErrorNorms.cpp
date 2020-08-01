#include "LpmErrorNorms.hpp"
#include <sstream>

namespace Lpm {

template <typename Space>
std::string ErrNorms<Space>::infoString(const std::string& label, const int tab_level) const {
    std::ostringstream ss;
    std::string tabstr;
    for (int i=0; i<tab_level; ++i) tabstr += "\t";
    ss << tabstr << label << " ErrNorms: l1 = " << l1 << " l2 = " << l2 << " linf = " << linf << "\n";
    return ss.str();
}

template <typename Space>
void ErrNorms<Space>::compute(const SphereGeometry::crd_view_type& err,
  const SphereGeometry::crd_view_type& appx, const SphereGeometry::crd_view_type& exact, const scalar_view_type& wt) {
  ko::parallel_for(err.extent(0), KOKKOS_LAMBDA (const Index& i) {
    for (Short j=0; j<err.extent(1); ++j) {
      err(i,j) = appx(i,j) - exact(i,j);
    }
  });
  compute(err, exact, wt);
}

template <typename Space>
void ErrNorms<Space>::compute(const SphereGeometry::vec_view_type& err, const SphereGeometry::vec_view_type& exct,
   const scalar_view_type& wt) {
  ENormScalar rval;
  ko::parallel_reduce(err.extent(0), KOKKOS_LAMBDA (const Index& i, ENormScalar& ll) {
    auto my_err = ko::subview(err, i, ko::ALL());
    auto my_exact = ko::subview(exct, i, ko::ALL());
    const Real errmag = SphereGeometry::mag(my_err);
    const Real exactmag = SphereGeometry::mag(my_exact);
    ll.l1num += errmag * wt(i);
    ll.l1denom += exactmag *wt(i);
    ll.l2num += square(errmag)*wt(i);
    ll.l2denom += square(exactmag)*wt(i);
    if (errmag > ll.linfnum) ll.linfnum = errmag;
    if (exactmag > ll.linfdenom) ll.linfdenom = exactmag;
  }, ErrReducer<Space>(rval));
  l1 = rval.l1num / rval.l1denom;
  l2 = std::sqrt(rval.l2num / rval.l2denom);
  linf = rval.linfnum / rval.linfdenom;
}

template <typename Space>
void ErrNorms<Space>::compute(scalar_view_type& err, const scalar_view_type& appx,
  const scalar_view_type& exact, const scalar_view_type& wt) {
  ko::parallel_for(err.extent(0), KOKKOS_LAMBDA (const Index& i) {
    err(i) = appx(i) - exact(i);
  });
  compute(err, exact, wt);
}

template <typename Space>
void ErrNorms<Space>::compute(const scalar_view_type& err, const scalar_view_type& exact, const scalar_view_type& wt) {
  ENormScalar rval;
  ko::parallel_reduce(err.extent(0), KOKKOS_LAMBDA (const Index& i, ENormScalar& ll) {
    ll.l1num += std::abs(err(i))*wt(i);
    ll.l1denom += std::abs(exact(i))*wt(i);
    ll.l2num += square(err(i))*wt(i);
    ll.l2denom += square(exact(i))*wt(i);
    ll.linfnum = (err(i) > ll.linfnum ? err(i) : ll.linfnum);
    ll.linfdenom = (exact(i) > ll.linfdenom ? exact(i) : ll.linfdenom);
  }, ErrReducer<Space>(rval));
  l1 = rval.l1num / rval.l1denom;
  l2 = std::sqrt(rval.l2num / rval.l2denom);
  linf = rval.linfnum / rval.linfdenom;
}

template struct ErrNorms<Host>;

}
