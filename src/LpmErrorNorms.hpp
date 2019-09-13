#ifndef LPM_ERROR_NORMS_HPP
#define LPM_ERROR_NORMS_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmKokkosUtil.hpp"
#include "LpmUtilities.hpp"
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

template <typename Space=Host>
struct ErrNorms {
    Real l1;
    Real l2;
    Real linf;
    
    ErrNorms(const Real l_1, const Real l_2, const Real l_i) : l1(l_1), l2(l_2), linf(l_i) {}

    std::string infoString(const std::string& label="", const int tab_level=0) const;
    
    ErrNorms(const scalar_view_type& er, const scalar_view_type& ex, const scalar_view_type& wt) : l1(0), l2(0), linf(0) {
        compute(er, ex, wt);
    }
    
    void compute(const scalar_view_type& er, const scalar_view_type& ex, const scalar_view_type& wt) {
        ENormScalar rval;
        ko::parallel_reduce(er.extent(0), KOKKOS_LAMBDA (const Index& i, ENormScalar& ll) {
            ll.l1num += std::abs(er(i))*wt(i);
            ll.l1denom += std::abs(ex(i))*wt(i);
            ll.l2num += square(er(i))*wt(i);
            ll.l2denom += square(ex(i))*wt(i);
            if (std::abs(er(i)) > ll.linfnum) ll.linfnum = std::abs(er(i));
            if (std::abs(ex(i)) > ll.linfdenom) ll.linfdenom = std::abs(ex(i));
        }, ErrReducer<Space>(rval));
        l1 = rval.l1num/rval.l1denom;
        l2 = std::sqrt(rval.l2num/rval.l2denom);
        linf = rval.linfnum/rval.linfdenom;
    }
    
};



}
#endif