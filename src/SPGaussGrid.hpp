#ifndef SPHERICAL_GAUSSIAN_GRID_HPP
#define SPHERICAL_GAUSSIAN_GRID_HPP

#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "Kokkos_Core.hpp"
#include "LpmUtilities.hpp"
#include <cmath>
#include <limits>

namespace Lpm {
// namespace Spherepack {

/// Reference: Spherepack file gaqd.f

static constexpr Real oor3 = 0.577350269189626; /// 1/sqrt(3)
static constexpr Real pio2 = 0.5*PI;

template <typename VT=ko::View<Real*>> KOKKOS_INLINE_FUNCTION
void fourier_coeff_legendre_poly(const Int& n, Real& constcoeff, VT coeffs, VT dcoeffs);

template <typename CVT=ko::View<const Real*>>
void legendre_polynomial(const Int& n, const Real& theta, const Real& constcoeff, 
    const CVT coeffs, const CVT dcoeffs, Real& polyval, Real& dpolyval);

template <typename MemSpace=DevMem>
struct GaussGrid {
    ko::View<Real*, MemSpace> colatitudes;
    ko::View<Real*, MemSpace> weights;
    Int nlat;
    
    GaussGrid(const Int nl) : nlat(nl), colatitudes("colat", nl), weights("weights", nl) {
        if (nl < 1) ko::abort("GaussGrid error : nlat must be >= 1\n");
        init();
    }
    
    std::string infoString(const int tab_level=0) const;
    
    void init();
};

template <typename MemSpace>
void GaussGrid<MemSpace>::init() {
    switch (nlat) {
        case (1) : {
            colatitudes(0) = 0.5*PI;
            weights(0) = 2.0;
            break;
        }
        case (2) : {
            colatitudes(0) = std::acos(oor3);
            colatitudes(1) = std::acos(-oor3);
            weights(0) = 1.0;
            weights(1) = 1.0;
            break;
        default : {
            const Real eps_roundoff = std::numeric_limits<Real>::epsilon();
            const Int nhalf = (nlat+1)/2;
            const Real dtheta = pio2/nhalf;
            const Int no2 = nlat/2;
            
            Real const_coeff;
            ko::View<Real*, MemSpace> coeffs("fourier_coeffs",nlat);
            ko::View<Real*, MemSpace> deriv_coeffs("fourier_deriv_coeffs",nlat);
            fourier_coeff_legendre_poly<ko::View<Real*,MemSpace>>(nlat, const_coeff, 
                coeffs, deriv_coeffs);
            std::cout << "const_coeff = " << const_coeff << "\n";
            std::cout << "coeffs = (";
            for (Int i=0; i<nlat; ++i) {
                std::cout << coeffs(i) << (i<nlat-1 ? " " : ")\n");
            }    
            std::cout << "deriv_coeffs = (";
            for (Int i=0; i<nlat; ++i) {
                std::cout << deriv_coeffs(i) << (i<nlat-1 ? " " : ")\n");
            }            
            const Real max_corr = 0.2*dtheta;
            
            /// Start at equator, theta = pi/2
            Real root_current, root_prev, root_last;
            Int root_ind;   
            if (nlat%2!=0) {
                root_current = pio2 - dtheta;
                root_prev = pio2;
                root_ind = nhalf-1;
            }
            else {
                root_current = pio2 - 0.5*dtheta;
                root_ind = nhalf;
            }
            Int max_iters = 0;
            Int iter = 0;
            const Int iter_limit = 4;
            Real polyval, derivval;
            while (root_ind >= 0) {
                if (iter > max_iters) max_iters = iter;
                iter=0;
                bool keepGoing = true;
                while (keepGoing) {
                    iter++;
                    root_last = root_current;
                    /// Newton iterations
                    Real polyval, derivval;
                    legendre_polynomial<ko::View<Real*,MemSpace>>(nlat, root_current, const_coeff, 
                        coeffs, deriv_coeffs, polyval, derivval);
                    Real corr = polyval/derivval;
                    Real sgn = 1.0;
                    if (corr != 0.0) sgn = corr/std::abs(corr);
                    corr = sgn*std::min(std::abs(corr), max_corr);
                    std::cout << "iter " << iter << " p_n^0 = " << polyval << " dp_n^0 = " << derivval << " corr = " << corr << "\n";
                    root_current -= corr;
                    std::cout << "rc = " << root_current << " rl = " << root_last << " rp = " << root_prev << " ri = " << root_ind << "\n";
                    if (std::abs(root_current-root_last) <= eps_roundoff*std::abs(root_current)) {
                        std::cout << "converged!!!\n";
                        keepGoing = false;
                    }
                    if (iter == iter_limit) keepGoing = false;
                }
                colatitudes(root_ind) = root_current;
                Real root_temp = root_current;
                weights(root_ind) = (2*nlat+1)/(square(derivval+polyval*std::cos(root_last)/std::sin(root_last)));
                root_ind--;
                if (root_ind == nhalf-1) root_current = 3*root_current - PI;
                if (root_ind < nhalf-1) root_current = 2*root_current - root_prev;
                root_prev = root_temp;
            }
            /// Apply symmetry
            if (nlat%2 != 0) {
                colatitudes(nhalf) = pio2;
                legendre_polynomial<ko::View<Real*,MemSpace>>(nlat, pio2, const_coeff, 
                    coeffs, deriv_coeffs, polyval, derivval);
                weights(nhalf) = (2*nlat + 1)/square(derivval);
            }
            for (Int i=0; i<no2; ++i) {
                weights(nlat-i) = weights(i);
                colatitudes(nlat-i) = PI - colatitudes(i);
            }
            Real sum = 0;
            for (Int i=0; i<nlat; ++i) {
                sum += weights(i); 
            }
            for (Int i=0; i<nlat; ++i) {
                weights(i) *= (2.0/sum);
            }
            break;
        }
    }
}
}


/**
    Computes the Fourier coefficients a_{nk} of the Legendre polynomial P_n^0 and its derivative, where
    
    P_n^0(\theta) = \sum_{k=0}^n a_{nk}\cos(k\theta).
    
    See subroutine `cpdp` in `gaqd.f`.
    
    a_{11} = \sqrt{2}
    a_{jj} = \sqrt\left(1 - \frac{1}{4n^2}a_{n-1,n-1}\right)
    
    
    - or - 
    
    a{nn} = \sqrt{\frac{2n+1}}{2} \cdot \frac{\Gamma(2n+1)}{2^{2n-1}\Gamma^2(n+1)}
    
    a_{nk} (n /= k):
    (n+k)(2n-(n+k)+1)a_{nk} - ((n+k)-1)(2n-(n+k)+2)a_{n,k+2}=0
*/
template <typename VT> KOKKOS_INLINE_FUNCTION
void fourier_coeff_legendre_poly(const Int& n, Real& constcoeff, VT coeffs, VT dcoeffs) {
    /**
        n = degree of polynomial
        if n is even, n/2 coefficients are returned in cp and dcp
        if n is odd, (n+1)/2 coefficients are returned in cp and dcp
        if n is even, the constant coefficient is returned in cz
        cz is unused if n is odd
        
        Output:
            cp : coefficients
            dcp : derivative coefficients
    */
    const Int equator_index = (n+1)/2 - 1; // -1 for 0-based indexing
    coeffs(equator_index) = 1.0;
    Real t1, t2, t3, t4;
    t1 = -1.0;
    t2 = n + 1.0;
    t3 = 0.0;
    t4 = 2*n + 1.0;
    if (n%2 == 0) {
        for (Int j=equator_index; j>0; --j) {
            t1 += 2.0;
            t2 -= 1.0;
            t3 += 1.0;
            t4 -= 2.0;
            coeffs(j-1) = (t1*t2)/(t3*t4)*coeffs(j);
        }
        t1 += 2.0;
        t2 -= 1.0;
        t3 += 1.0;
        t4 -= 2.0;
        constcoeff = (t1*t2)/(t3*t4)*coeffs(0);
        for (Int j=0; j<=equator_index; ++j) {
            dcoeffs(j) = 2*j*coeffs(j);
        }
    }
    else {
        for (Int j=equator_index-1; j>=0; --j) {
            t1 += 2.0;
            t2 -= 1.0;
            t3 += 1.0;
            t4 -= 2.0;
            coeffs(j) = (t1*t2)/(t3*t4)*coeffs(j+1);
        }
        for (Int j=0; j<=equator_index; ++j) {
            dcoeffs(j) = (2*j-1)*coeffs(j);
        }
    }
}

/**
    Computes the value of the Legendre polynomial P_n(theta), where theta is colatitude, and its derivative
*/
template <typename CVT>
void legendre_polynomial(const Int& n, const Real& theta, const Real& constcoeff, 
    const CVT coeffs, const CVT dcoeffs, Real& polyval, Real& dpolyval) {
    const Real cos2 = std::cos(2*theta);
    const Real sin2 = std::sin(2*theta);
    if (n%2 == 0) {
        const Int kstop = n/2;
        polyval = 0.5*constcoeff;
        dpolyval = 0.0;
        if (n>0) {
            Real cth = cos2;
            Real sth = sin2;
            Real chh;
            for (Int k=0; k<kstop; ++k) {
                polyval += coeffs(k)*cth;
                dpolyval -= dcoeffs(k)*sth;
                chh = cos2*cth - sin2*sth;
                sth = sin2*cth + cos2*sth;
                cth = chh;
            }
        }
    }
    else {
        const Int kstop = (n+1)/2;
        polyval = 0.0;
        dpolyval = 0.0;
        Real cth = std::cos(theta);
        Real sth = std::sin(theta);
        Real chh;
        for (Int k=0; k<kstop; ++k) {
            polyval += coeffs(k)*cth;
            dpolyval -= dcoeffs(k)*sth;
            chh = cos2*cth - sin2*sth;
            sth = sin2*cth + cos2*sth;
            cth = chh;
        }
    }
}

}
#endif
