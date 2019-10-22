#include "LpmGaussGrid.hpp"

#ifdef LPM_HAVE_SPHEREPACK
#include <sstream>
#include <string>
#include <iostream>

namespace Lpm {

template <typename MemSpace>
std::string GaussGrid<MemSpace>::infoString(const int tab_level) const {
    std::ostringstream ss;
    std::string tabstr;
    for (int i=0; i<tab_level; ++i) tabstr += "\t";
    ss << "GaussGrid info:\n";
    ss << tabstr << "\tnlat = " << nlat << "\n";
    ss << tabstr << "\tcolatitudes = [";
    for (Int i=0; i<nlat; ++i) {
        ss << colatitudes(i) << (i<nlat-1 ? " ": "]\n");
    }
    ss << "\tweights = [";
    for (Int i=0; i<nlat; ++i) {
        ss << weights(i) << (i<nlat-1 ? " " : "]\n");
    }
    return ss.str();
}


template struct GaussGrid<DevMem>;
template void legendre_polynomial<ko::View<Real*>>(const Int& n, const Real& theta, const Real& constcoeff, 
    const ko::View<Real*> coeffs, const ko::View<Real*> dcoeffs, Real& polyval, Real& dpolyval);
template void fourier_coeff_legendre_poly<ko::View<Real*>>(const Int& n, Real& constcoeff, ko::View<Real*> coeffs, ko::View<Real*> dcoeffs);

}

#endif
