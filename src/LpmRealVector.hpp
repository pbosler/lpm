#ifndef LPM_REAL_REALVEC_HPP
#define LPM_REAL_REALVEC_HPP

#include <cmath>
#include <iostream>
#include <vector>
#include <exception>
#include <array>
#include "LpmConfig.h"
#include "LpmTypeDefs.hpp"
#include "LpmUtilities.hpp"
#include "Kokkos_Core.hpp"
#include "Kokkos_View.hpp"

namespace Lpm {

/*
	Struct for handling RealVector computations in R^d, where typically d = 2,3.
*/
template<int ndim=3> struct RealVec {
    Real x[ndim];

	/// Constructor, initialize to 0.
	KOKKOS_INLINE_FUNCTION
    RealVec() {
        for (int i=0; i<ndim; ++i)
            this->x[i] = 0.0;
    }
    
    /// Copy constructor
    template <int ndim2> KOKKOS_INLINE_FUNCTION
    RealVec(const RealVec<ndim2>& other){
        if (ndim < ndim2) {
            throw std::runtime_error("RealVec<ndim>(RealVec<ndim2>&) : cannot convert to smaller type.");
        }
        for (int i=0; i<std::min(ndim, ndim2); ++i) {
            this->x[i] = other.x[i];
        }
        for (int i=std::min(ndim, ndim2)-1; i<ndim; ++i) {
            this->x[i] = 0.0;
        }
    }

	/// Copy constructor
	KOKKOS_INLINE_FUNCTION
    RealVec(const RealVec<ndim>& other) {
        for (int i=0; i<ndim; ++i)
            this->x[i] = other.x[i];
    }

	/// Constructor, initialize first two components
	KOKKOS_INLINE_FUNCTION
    RealVec(const Real xx, const Real yy) {
        this->x[0] = xx;
        this->x[1] = yy;
    }

	/// Constructor, initialize 3 components.
	KOKKOS_INLINE_FUNCTION
    RealVec(const Real xx, const Real yy, const Real zz) {
        if (ndim == 2) {
            this->x[0] = xx;
            this->x[1] = yy;
        }
        else if (ndim == 3) {
            this->x[0] = xx;
            this->x[1] = yy;
            this->x[2] = zz;
        }
    }

	/// Constructor, initialize from std::array<Real, ndim>
    inline RealVec(const std::array<Real, ndim>& arr) {
        for (int i=0; i<ndim; ++i)
            this->x[i] = arr[i];
    }
    
	/// Constructor, initialize from std::vector<Real>
    inline RealVec(const std::vector<Real>& xx) {
        for (int i=0; i<ndim; ++i)
            this->x[i] = xx[i];
    }
    
    /// Constructor, initialize from Kokkos::View
    KOKKOS_INLINE_FUNCTION
    RealVec(Kokkos::View<const Real[3]> v) {
        for (int i=0; i<ndim; ++i) {
            this->x[i] = v(i);
        }
    }
    
	/// Constructor, intialize from Real[]
	KOKKOS_INLINE_FUNCTION
    RealVec(const Real* xx) {
        for (int i=0; i<ndim; ++i)
            this->x[i] = xx[i];
    }    
    /// Basic assignment operator.
    KOKKOS_INLINE_FUNCTION
    RealVec& operator= (const RealVec<ndim>& other) {
        if (this != &other) {
            for (int i=0; i<ndim; ++i)
                this->x[i] = other.x[i];
        }
        return *this;
    }

	/// Convert to std::array<Real, ndim>
    inline std::array<Real, ndim> toArray() const {
        std::array<Real, ndim> result;
        for (int i=0; i<ndim; ++i)
            result[i] = this->x[i];
        return result;
    }

	/// Convert to std::vector<Real>
    inline std::vector<Real> toStdRealVec() const {
        std::vector<Real> result(ndim);
        for (int i=0; i<ndim; ++i)
            result[i] = this->x[i];
        return result;
    }
    
    KOKKOS_INLINE_FUNCTION
    void toKokkosView(Kokkos::View<Real[3]> v) const {
        for (int i=0; i<ndim; ++i) 
            v(i) = this->x[i];
    }
    
    /// [] operator
    KOKKOS_INLINE_FUNCTION
    const Real& operator [] (const Index ind) const {return this->x[ind];}
    
    KOKKOS_INLINE_FUNCTION
    Real& operator [] (const Index ind) {return this->x[ind];}
	
	/// += operator (elemental)
    KOKKOS_INLINE_FUNCTION
    RealVec<ndim>& operator += (const RealVec<ndim>& other) {
        for (int i=0; i<ndim; ++i)
            this->x[i] += other.x[i];
        return *this;
    }

	/// -= operator (elemental)
    KOKKOS_INLINE_FUNCTION 
    RealVec<ndim>& operator -= (const RealVec<ndim>& other) {
        for (int i=0; i<ndim; ++i)
            this->x[i] -= other.x[i];
        return *this;
    }

	/// *= operator (elemental)
   KOKKOS_INLINE_FUNCTION
   RealVec<ndim>& operator *= (const RealVec<ndim>& other) {
        for (int i=0; i<ndim; ++i)
            this->x[i] *= other.x[i];
        return *this;
    }

	/// /= operator (elemental)
    KOKKOS_INLINE_FUNCTION 
    RealVec<ndim>& operator /= (const RealVec<ndim>& other) {
        for (int i=0; i<ndim; ++i)
            this->x[i] /= other.x[i];
        return *this;
    }

	/// + operator member function
    KOKKOS_INLINE_FUNCTION
    RealVec<ndim> operator + (const RealVec<ndim>& other) const {
        return RealVec<ndim>(*this) += other;
    }

	/// - operator member function
    KOKKOS_INLINE_FUNCTION
	 RealVec<ndim> operator - (const RealVec<ndim>& other) const {
        return RealVec<ndim>(*this) -= other;
    }

	/// * operator member function
    KOKKOS_INLINE_FUNCTION
	 RealVec<ndim> operator * (const RealVec<ndim>& other) const {
        return RealVec<ndim>(*this) *= other;
    }

	/// / operator member function
    KOKKOS_INLINE_FUNCTION
	 RealVec<ndim> operator / (const RealVec<ndim>& other) const {
        return RealVec<ndim>(*this) /= other;
    }

	/// compute & return the square of the RealVector's magnitude
    KOKKOS_INLINE_FUNCTION
	 Real magSq() const {
        Real sumsq(0.0);
        for (int i=0; i<ndim; ++i)
            sumsq += this->x[i]*this->x[i];
        return sumsq;
    }

	/// compute and return the RealVector's magnitude
    KOKKOS_INLINE_FUNCTION
	 Real mag() const {
        return std::sqrt(this->magSq());
    }

	/// RealVector dot product
    KOKKOS_INLINE_FUNCTION
	 Real dotProd(const RealVec<ndim>& other) const {
        Real dp(0.0);
        for (int i=0; i<ndim; ++i)
            dp += this->x[i]*other.x[i];
        return dp;
    }

	/// 2d cross product (returns the only nonzero component as a scalar)
    KOKKOS_INLINE_FUNCTION
	 Real crossProdComp3(const RealVec<2>& other) const {
        return this->x[0]*other.x[1] - this->x[1]*other.x[0];
    }

	/// RealVector cross product
    KOKKOS_INLINE_FUNCTION
	 RealVec<ndim> crossProd(const RealVec<ndim>& other) const {
        const Real cp[3] = {this->x[1]*other.x[2] - this->x[2]*other.x[1],
                                   this->x[2]*other.x[0] - this->x[0]*other.x[2],
                                   this->x[0]*other.x[1] - this->x[1]*other.x[0]};
        return RealVec<ndim>(cp);
    }

	/// Scalar multiply without copy and no return 
    KOKKOS_INLINE_FUNCTION
	 void scaleInPlace(const Real mult) {
        for (int i=0; i<ndim; ++i)
            this->x[i] *= mult;
    }

	/// Scalar multiply with copy, return new RealVec
    KOKKOS_INLINE_FUNCTION
	 RealVec<ndim> scale(const Real mult) const {
        Real sm[ndim];
        for (int i=0; i<ndim; ++i)
            sm[i] = this->x[i]*mult;
        return RealVec<ndim>(sm);
    }

	/// Normalize RealVector without copy, no return
    KOKKOS_INLINE_FUNCTION
	 void normalizeInPlace() {
        const Real len = this->mag();
        this->scaleInPlace(1.0/len);
    }

	/// Normalize RealVector with copy, return new RealVec
    KOKKOS_INLINE_FUNCTION
	 RealVec<ndim> normalize() const {
        const Real len = this->mag();
        return this->scale(1.0/len);
    }

	/// Compute & return the longitude of a RealVector in R^3
    KOKKOS_FUNCTION
	 Real longitude() const {return atan4(this->x[1], this->x[0]);}


	/// Compute and return the latitude of a RealVector in R^3
    KOKKOS_INLINE_FUNCTION
	 Real latitude() const {
        const Real xy2 = this->x[0]*this->x[0] + this->x[1]*this->x[1];
        return std::atan2(this->x[2] , std::sqrt(xy2));
    }

	/// Compute the midpoint of *this and another RealVec<ndim>
    KOKKOS_INLINE_FUNCTION
	 RealVec<ndim> midpoint(const RealVec<ndim>& other) const {
        RealVec<ndim> result = *this + other;
        result.scaleInPlace(0.5);
        return result;
    }

	/// Compute the great-circle midpoint of *this and another RealVec<ndim>
    KOKKOS_INLINE_FUNCTION
	 RealVec<ndim> sphereMidpoint(const RealVec<ndim>& other, const Real radius = 1.0) const {
        RealVec<ndim> result = this->midpoint(other);
        result.normalizeInPlace();
        result.scaleInPlace(radius);
        return result;
    }

	/// Compute the Euclidean distance between *this and another RealVec<ndim>
    KOKKOS_INLINE_FUNCTION
	 Real dist(const RealVec<ndim>& other) const {
        return (*this - other).mag();
    }

    KOKKOS_INLINE_FUNCTION
	 Real sphereDist(const RealVec<ndim>& other, const Real radius=1.0) const {
        const RealVec<ndim> cp = this->crossProd(other);
        const Real dp = this->dotProd(other);
        return std::atan2(cp.mag(), dp) * radius;
    }

	/// equivalence operator
    KOKKOS_INLINE_FUNCTION
	 bool operator == (const RealVec<ndim>& other) const {
        return this->dist(other) < ZERO_TOL;
    }
    
    KOKKOS_INLINE_FUNCTION
	 bool operator != (const RealVec<ndim>& other) const {
        return !(*this == other);
    }
};

/// Return the RealVec<ndim> along a chord between RealVec<ndim> a and RealVec<ndim> b.
/**	Chord is parameterized by s \in [-1,1]
	
	@param a origin of chord RealVector
	@param b destination of chord RealVector
	@param s parameterization variable
**/
template <int ndim> KOKKOS_FUNCTION
RealVec<ndim> pointAlongChord(RealVec<ndim> a, RealVec<ndim> b, const Real s) {
    a.scaleInPlace(1.0-s);
    b.scaleInPlace(1.0+s);
    RealVec<ndim> result = a + b;
    result.scaleInPlace(0.5);
    return result;
}

/// Return the RealVec<ndim> along a great-circle arc (in R^3) between RealVec<ndim> a and RealVec<ndim> b.
/**	Chord is parameterized by s \in [-1,1]
	
	@param a origin of chord RealVector
	@param b destination of chord RealVector
	@param s parameterization variable
**/
template <int ndim> KOKKOS_FUNCTION
RealVec<ndim> pointAlongCircle(const RealVec<ndim>& a, const RealVec<ndim>& b, const Real s, 
    const Real radius=1.0) {
    RealVec<ndim> result = pointAlongChord(a,b,s);
    result.normalizeInPlace();
    result.scaleInPlace(radius);
    return result;
}

/// Return the Euclidean barycenter of a collection of RealVec<ndim>s
template <int ndim> RealVec<ndim> barycenter(const std::vector<RealVec<ndim>>& RealVecs) {
    RealVec<ndim> result;
    for (int i=0; i<RealVecs.size(); ++i)
        result += RealVecs[i];
    result.scaleInPlace(1.0/RealVecs.size());
    return result;
}

/// Return the Euclidean barycenter from a Kokkos::View
template <int ndim> KOKKOS_FUNCTION
RealVec<ndim> barycenter(Kokkos::View<Real*[ndim]> crds) {
    RealVec<ndim> result;
    for (int i=0; i<crds.extent(0); ++i) {
        for (int j=0; j<ndim; ++j) 
            result[j] += crds(i,j);
    }
    result.scaleInPlace(1.0/crds.extent(0));
    return result;
}

/// Return the barycenter on the spherical surface  of a collection of RealVec<ndim>s (using radial projection)
template <int ndim> RealVec<ndim> sphereBarycenter(const std::vector<RealVec<ndim>>& RealVecs, const Real radius = 1.0) {
    RealVec<3> result;
    for (int i=0; i<RealVecs.size(); ++i)
        result += RealVecs[i];
    result.scaleInPlace(1.0/RealVecs.size());
    result.normalizeInPlace();
    result.scaleInPlace(radius);
    return result;
}

template <int ndim> KOKKOS_FUNCTION 
RealVec<ndim> sphereBarycenter(Kokkos::View<Real*[ndim]> crds, const Real radius=1.0) {
    RealVec<ndim> result;
    for (int i=0; i<crds.extent(0); ++i) {
        for (int j=0; j<ndim; ++j) 
            result[j] += crds(i,j);
    }
    result.scaleInPlace(1.0/crds.extent(0));
    result.normalizeInPlace();
    result.scaleInPlace(radius);
    return result;
}

/// Return the area of a planar triangle defined by 3 vertices
KOKKOS_INLINE_FUNCTION
Real triArea(const RealVec<3>& RealVecA, const RealVec<3>& RealVecB, const RealVec<3>& RealVecC) {
    const RealVec<3> s1 = RealVecB - RealVecA;
    const RealVec<3> s2 = RealVecC - RealVecA;
    return 0.5*s1.crossProd(s2).mag();
}

/// Return the area of a planar triangle defined by 3 vertices
KOKKOS_INLINE_FUNCTION
Real triArea(const RealVec<2>& RealVecA, const RealVec<2>& RealVecB, const RealVec<2>& RealVecC) {
    const RealVec<2> s1 = RealVecB - RealVecA;
    const RealVec<2> s2 = RealVecC - RealVecA;
    return std::abs(0.5*s1.crossProdComp3(s2));
}

/// Return the area of a planar triangle defined by 3 vertices
template <int ndim> Real triArea(const std::vector<RealVec<ndim>>& RealVecs) {
    return triArea(RealVecs[0], RealVecs[1], RealVecs[2]);
}

/// Return the area of a spherical triangle defined by 3 vertices on the surface of a sphere
template <int ndim> KOKKOS_FUNCTION
Real sphereTriArea(const RealVec<ndim>& a, const RealVec<ndim>& b, const RealVec<ndim>& c, const Real radius=1.0) {
    const Real s1 = a.sphereDist(b, radius);
    const Real s2 = b.sphereDist(c, radius);
    const Real s3 = c.sphereDist(a, radius);
    const Real halfPerim = 0.5*(s1 + s2 + s3);
    const Real zz = std::tan(0.5*halfPerim) * std::tan(0.5*(halfPerim-s1)) * std::tan(0.5*(halfPerim-s2)) *
        std::tan(0.5*(halfPerim-s3));
    return 4.0 * std::atan(std::sqrt(zz)) * radius*radius;
}

/// Return the area of a spherical triangle defined by 3 vertices on the surface of a sphere
inline Real sphereTriArea(const std::vector<RealVec<3>>& RealVecs) {
    return sphereTriArea(RealVecs[0], RealVecs[1], RealVecs[2]);
}

template <int ndim> Real polygonArea(const RealVec<ndim>& ctr, const std::vector<RealVec<ndim>>& ccwcorners){
	Real result = 0.0;
	const int nverts = ccwcorners.size();
	for (int i=0; i<nverts; ++i) {
		result += triArea(ccwcorners[i], ctr, ccwcorners[(i+1)%nverts]);
	}
	return result;
}

template <int ndim> KOKKOS_FUNCTION
Real polygonArea(Kokkos::View<Real[3]> ctr, Kokkos::View<Real*[3]> ccwcorners) {
    Real result = 0.0;
    const RealVec<ndim> ctrvec(ctr);
    const int nverts = ccwcorners.extent(0);
    for (int i=0; i<nverts; ++i) {
        RealVec<ndim> v1;
        RealVec<ndim> v2;
        for (int j=0; j<ndim; ++j) {
            v1[j] = ccwcorners(i,j);
            v2[j] = ccwcorners((i+1)%nverts, j);
        }
        result += triArea(v1, ctrvec, v2);
    }
    return result;
}

template <int ndim> KOKKOS_FUNCTION
Real spherePolygonArea(Kokkos::View<Real[3]> ctr, Kokkos::View<Real*[3]> ccwcorners) {
    Real result = 0.0;
    const RealVec<ndim> ctrvec(ctr);
    const int nverts = ccwcorners.extent(0);
    for (int i=0; i<nverts; ++i) {
        RealVec<ndim> v1;
        RealVec<ndim> v2;
        for (int j=0; j<ndim; ++j) {
            v1[j] = ccwcorners(i,j);
            v2[j] = ccwcorners((i+1)%nverts, j);
        }
        result += sphereTriArea(v1, ctrvec, v2);
    }
    return result;
}

template <int ndim> Real spherePolygonArea(const RealVec<ndim>& ctr, const std::vector<RealVec<ndim>>& ccwcorners, 
	const Real radius) {
	Real result = 0.0;
	const int nverts = ccwcorners.size();
	for (int i=0; i<nverts; ++i) {
		result += sphereTriArea(ccwcorners[i], ctr, ccwcorners[(i+1)%nverts], radius);
	}
	return result;
}


/// Basic output to console
std::ostream& operator << (std::ostream& os, const RealVec<1>& RealVec);
/// Basic output to console
std::ostream& operator << (std::ostream& os, const RealVec<2>& RealVec);
/// Basic output to console
std::ostream& operator << (std::ostream& os, const RealVec<3>& RealVec);

}
#endif
