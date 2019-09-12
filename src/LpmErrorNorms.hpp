#ifndef LPM_ERROR_NORMS_HPP
#define LPM_ERROR_NORMS_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmKokkosUtil.hpp"
#include "LpmUtilities.hpp"
#include "Kokkos_Core.hpp"

namespace Lpm {

template <typename Space=Dev>
struct ErrReducer {
    public:
    typedef ErrReducer reducer;
    typedef ko::Tuple<Real,6> value_type;
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
        for (int i=0; i<4; ++i) {
            dest[i] += src[i];
        }
        for (int i=4; i<6; ++i) {
            if (abs(src[i]) > dest[i]) dest[i] = abs(src[i]);
        }
    }
    
    KOKKOS_INLINE_FUNCTION
    void join(volatile value_type& dest, const volatile value_type& src) const {
        for (int i=0; i<4; ++i) {
            dest[i] += src[i];
        }
        for (int i=4; i<6; ++i) {
            if (abs(src[i]) > dest[i]) dest[i] = abs(src[i]);
        }
    }
    
    KOKKOS_INLINE_FUNCTION
    void init(value_type& val) const {
        for (int i=0; i<4; ++i) {
            val[i] = ko::reduction_identity<Real>::sum();
        }
        for (int i=4; i<6; ++i) {
            val[i] = ko::reduction_identity<Real>::max();
        }
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
    
    ErrNorms(const ko::Tuple<Real,6>& tup) : l1(tup[0]/tup[1]), l2(tup[2]/tup[3]), linf(tup[4]/tup[5]) {}
    
    ErrNorms(const ko::View<Real[6],Space>& v) : l1(v[0]/v[1]), l2(v[2]/v[3]), linf(v[4]/v[5]) {}
    
    ErrNorms(const scalar_view_type& er, const scalar_view_type& ex, const scalar_view_type& wt) : l1(0), l2(0), linf(0) {
        compute(er, ex, wt);
    }
    
    void compute(const scalar_view_type& er, const scalar_view_type& ex, const scalar_view_type& wt) {
        ko::View<ko::Tuple<Real,6>> tview("tuple_view");
        ko::parallel_reduce(er.extent(0), KOKKOS_LAMBDA (const Index& i, ko::Tuple<Real,6>& eup) {
            eup[0] += std::abs(er(i))*wt(i);
            eup[1] += std::abs(ex(i))*wt(i);
            eup[2] += square(er(i))*wt(i);
            eup[3] += square(ex(i))*wt(i);
            eup[4] = max(std::abs(er(i)), eup[4]);
            eup[5] = max(std::abs(ex(i)), eup[5]);
        }, ErrReducer<Dev>(tview));
        auto host_tv = ko::create_mirror_view(tview);
        ko::deep_copy(host_tv, tview);
        auto host_tup = *host_tv.data();
        l1 = host_tup[0]/host_tup[1];
        l2 = std::sqrt(host_tup[2]/host_tup[3]);
        linf = host_tup[4]/host_tup[5];
    }
    
};



}
#endif