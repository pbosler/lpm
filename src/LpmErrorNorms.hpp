#ifndef LPM_ERROR_NORMS_HPP
#define LPM_ERROR_NORMS_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmKokkosUtil.hpp"
#include "LpmUtilities.hpp"
#include "Kokkos_Core.hpp"

namespace Lpm {

struct ErrNorms {
    Real l1;
    Real l2;
    Real linf;
    
    ErrNorms(const Real l_1, const Real l_2, const Real l_i) : l1(l_1), l2(l_2), linf(l_i) {}

    std::string infoString(const std::string& label="", const int tab_level=0) const;
};



}
#endif