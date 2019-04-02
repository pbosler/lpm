#ifndef LPM_AOS_TYPES_HPP
#define LPM_AOS_TYPES_HPP

#include <cmath>
#include <iostream>
#include <vector>
#include <exception>
#include <array>
#include "LpmConfig.h"
#include "LpmTypeDefs.h"
#include "LpmUtilities.h"

namespace Lpm {
namespace Aos {

// scalar_type atan4(const scalar_type y, const scalar_type x);

typedef std::vector<index_type> ind_vec;

/*
	Struct for handling vector computations in R^d, where typically d = 2,3.
*/
template<int ndim=3> struct Vec {
    scalar_type x[ndim];

	/// Constructor, initialize to 0.
    Vec() {
        for (int i=0; i<ndim; ++i)
            this->x[i] = 0.0;
    }
    
    /// Copy constructor
    template <int ndim2> Vec(const Vec<ndim2>& other){
        if (ndim < ndim2) {
            throw std::runtime_error("Vec<ndim>(Vec<ndim2>&) : cannot convert to smaller type.");
        }
        for (int i=0; i<std::min(ndim, ndim2); ++i) {
            this->x[i] = other.x[i];
        }
        for (int i=std::min(ndim, ndim2)-1; i<ndim; ++i) {
            this->x[i] = 0.0;
        }
    }

	/// Copy constructor
    Vec(const Vec<ndim>& other) {
        for (int i=0; i<ndim; ++i)
            this->x[i] = other.x[i];
    }

	/// Constructor, initialize first two components
    Vec(const scalar_type xx, const scalar_type yy) {
        this->x[0] = xx;
        this->x[1] = yy;
    }

	/// Constructor, initialize 3 components.
    Vec(const scalar_type xx, const scalar_type yy, const scalar_type zz) {
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

	/// Constructor, initialize from std::array<scalar_type, ndim>
    Vec(const std::array<scalar_type, ndim>& arr) {
        for (int i=0; i<ndim; ++i)
            this->x[i] = arr[i];
    }
    
	/// Constructor, initialize from std::vector<scalar_type>
    inline Vec(const std::vector<scalar_type>& xx) {
        for (int i=0; i<ndim; ++i)
            this->x[i] = xx[i];
    }
    
	/// Constructor, intialize from scalar_type[]
    inline Vec(const scalar_type* xx) {
        for (int i=0; i<ndim; ++i)
            this->x[i] = xx[i];
    }    
    /// Basic assignment operator.
    Vec& operator= (const Vec<ndim>& other) {
        if (this != &other) {
            for (int i=0; i<ndim; ++i)
                this->x[i] = other.x[i];
        }
        return *this;
    }

	/// Convert to std::array<scalar_type, ndim>
    inline std::array<scalar_type, ndim> toArray() const {
        std::array<scalar_type, ndim> result;
        for (int i=0; i<ndim; ++i)
            result[i] = this->x[i];
        return result;
    }

	/// Convert to std::vector<scalar_type>
    inline std::vector<scalar_type> toStdVec() const {
        std::vector<scalar_type> result(ndim);
        for (int i=0; i<ndim; ++i)
            result[i] = this->x[i];
        return result;
    }
    
    /// [] operator
    inline const scalar_type& operator [] (const index_type ind) const {return this->x[ind];}
    
    inline scalar_type& operator [] (const index_type ind) {return this->x[ind];}
	
	/// += operator (elemental)
    inline Vec<ndim>& operator += (const Vec<ndim>& other) {
        for (int i=0; i<ndim; ++i)
            this->x[i] += other.x[i];
        return *this;
    }

	/// -= operator (elemental)
    inline Vec<ndim>& operator -= (const Vec<ndim>& other) {
        for (int i=0; i<ndim; ++i)
            this->x[i] -= other.x[i];
        return *this;
    }

	/// *= operator (elemental)
   inline  Vec<ndim>& operator *= (const Vec<ndim>& other) {
        for (int i=0; i<ndim; ++i)
            this->x[i] *= other.x[i];
        return *this;
    }

	/// /= operator (elemental)
    inline Vec<ndim>& operator /= (const Vec<ndim>& other) {
        for (int i=0; i<ndim; ++i)
            this->x[i] /= other.x[i];
        return *this;
    }

	/// + operator member function
    inline const Vec<ndim> operator + (const Vec<ndim>& other) const {
        return Vec<ndim>(*this) += other;
    }

	/// - operator member function
    inline const Vec<ndim> operator - (const Vec<ndim>& other) const {
        return Vec<ndim>(*this) -= other;
    }

	/// * operator member function
    inline const Vec<ndim> operator * (const Vec<ndim>& other) const {
        return Vec<ndim>(*this) *= other;
    }

	/// / operator member function
    inline const Vec<ndim> operator / (const Vec<ndim>& other) const {
        return Vec<ndim>(*this) /= other;
    }

	/// compute & return the square of the vector's magnitude
    inline scalar_type magSq() const {
        scalar_type sumsq(0.0);
        for (int i=0; i<ndim; ++i)
            sumsq += this->x[i]*this->x[i];
        return sumsq;
    }

	/// compute and return the vector's magnitude
    inline scalar_type mag() const {
        return std::sqrt(this->magSq());
    }

	/// vector dot product
    inline scalar_type dotProd(const Vec<ndim>& other) const {
        scalar_type dp(0.0);
        for (int i=0; i<ndim; ++i)
            dp += this->x[i]*other.x[i];
        return dp;
    }

	/// 2d cross product (returns the only nonzero component as a scalar)
    inline scalar_type crossProdComp3(const Vec<2>& other) const {
        return this->x[0]*other.x[1] - this->x[1]*other.x[0];
    }

	/// Vector cross product
    inline const Vec<ndim> crossProd(const Vec<ndim>& other) const {
        const scalar_type cp[3] = {this->x[1]*other.x[2] - this->x[2]*other.x[1],
                                   this->x[2]*other.x[0] - this->x[0]*other.x[2],
                                   this->x[0]*other.x[1] - this->x[1]*other.x[0]};
        return Vec<ndim>(cp);
    }

	/// Scalar multiply without copy and no return 
    inline void scaleInPlace(const scalar_type mult) {
        for (int i=0; i<ndim; ++i)
            this->x[i] *= mult;
    }

	/// Scalar multiply with copy, return new Vec
    inline const Vec<ndim> scale(const scalar_type mult) const {
        scalar_type sm[ndim];
        for (int i=0; i<ndim; ++i)
            sm[i] = this->x[i]*mult;
        return Vec<ndim>(sm);
    }

	/// Normalize vector without copy, no return
    inline void normalizeInPlace() {
        const scalar_type len = this->mag();
        this->scaleInPlace(1.0/len);
    }

	/// Normalize vector with copy, return new Vec
    inline const Vec<ndim> normalize() const {
        const scalar_type len = this->mag();
        return this->scale(1.0/len);
    }

	/// Compute & return the longitude of a vector in R^3
    inline scalar_type longitude() const {return atan4(this->x[1], this->x[0]);}


	/// Compute and return the latitude of a vector in R^3
    inline scalar_type latitude() const {
        const scalar_type xy2 = this->x[0]*this->x[0] + this->x[1]*this->x[1];
        return std::atan2(this->x[2] , std::sqrt(xy2));
    }

	/// Compute the midpoint of *this and another Vec<ndim>
    inline const Vec<ndim> midpoint(const Vec<ndim>& other) const {
        Vec<ndim> result = *this + other;
        result.scaleInPlace(0.5);
        return result;
    }

	/// Compute the great-circle midpoint of *this and another Vec<ndim>
    inline const Vec<ndim> sphereMidpoint(const Vec<ndim>& other, const scalar_type radius = 1.0) const {
        Vec<ndim> result = this->midpoint(other);
        result.normalizeInPlace();
        result.scaleInPlace(radius);
        return result;
    }

	/// Compute the Euclidean distance between *this and another Vec<ndim>
    inline scalar_type dist(const Vec<ndim>& other) const {
        return (*this - other).mag();
    }

    inline scalar_type sphereDist(const Vec<ndim>& other, const scalar_type radius=1.0) const {
        const Vec<ndim> cp = this->crossProd(other);
        const scalar_type dp = this->dotProd(other);
        return std::atan2(cp.mag(), dp) * radius;
    }

	/// Compute the great-circle distance between *this and another Vec<ndim>
    inline const bool operator == (const Vec<ndim>& other) const {
        return this->dist(other) < ZERO_TOL;
    }
};

/// Return the Vec<ndim> along a chord between Vec<ndim> a and Vec<ndim> b.
/**	Chord is parameterized by s \in [-1,1]
	
	@param a origin of chord vector
	@param b destination of chord vector
	@param s parameterization variable
**/
template <int ndim> Vec<ndim> pointAlongChord(Vec<ndim> a, Vec<ndim> b, const scalar_type s) {
    a.scaleInPlace(1.0-s);
    b.scaleInPlace(1.0+s);
    Vec<ndim> result = a + b;
    result.scaleInPlace(0.5);
    return result;
}

/// Return the Vec<ndim> along a great-circle arc (in R^3) between Vec<ndim> a and Vec<ndim> b.
/**	Chord is parameterized by s \in [-1,1]
	
	@param a origin of chord vector
	@param b destination of chord vector
	@param s parameterization variable
**/
template <int ndim> Vec<ndim> pointAlongCircle(const Vec<ndim>& a, const Vec<ndim>& b, const scalar_type s, 
    const scalar_type radius=1.0) {
    Vec<ndim> result = pointAlongChord(a,b,s);
    result.normalizeInPlace();
    result.scaleInPlace(radius);
    return result;
}

/// Return the Euclidean barycenter of a collection of Vec<ndim>s
template <int ndim> const Vec<ndim> baryCenter(const std::vector<Vec<ndim>>& vecs) {
    Vec<ndim> result;
    for (int i=0; i<vecs.size(); ++i)
        result += vecs[i];
    result.scaleInPlace(1.0/vecs.size());
    return result;
}

/// Return the barycenter on the spherical surface  of a collection of Vec<ndim>s (using radial projection)
template <int ndim> const Vec<ndim> sphereBaryCenter(const std::vector<Vec<ndim>>& vecs, const scalar_type radius = 1.0) {
    Vec<3> result;
    for (int i=0; i<vecs.size(); ++i)
        result += vecs[i];
    result.scaleInPlace(1.0/vecs.size());
    result.normalizeInPlace();
    result.scaleInPlace(radius);
    return result;
}

/// Return the area of a planar triangle defined by 3 vertices
inline scalar_type triArea(const Vec<3>& vecA, const Vec<3>& vecB, const Vec<3>& vecC) {
    const Vec<3> s1 = vecB - vecA;
    const Vec<3> s2 = vecC - vecA;
    return 0.5*s1.crossProd(s2).mag();
}

/// Return the area of a planar triangle defined by 3 vertices
inline scalar_type triArea(const Vec<2>& vecA, const Vec<2>& vecB, const Vec<2>& vecC) {
    const Vec<2> s1 = vecB - vecA;
    const Vec<2> s2 = vecC - vecA;
    return std::abs(0.5*s1.crossProdComp3(s2));
}

/// Return the area of a planar triangle defined by 3 vertices
template <int ndim> scalar_type triArea(const std::vector<Vec<ndim>>& vecs) {
    return triArea(vecs[0], vecs[1], vecs[2]);
}

/// Return the area of a spherical triangle defined by 3 vertices on the surface of a sphere
scalar_type sphereTriArea(const Vec<3>& a, const Vec<3>& b, const Vec<3>& c, const scalar_type radius = 1.0);

/// Return the area of a spherical triangle defined by 3 vertices on the surface of a sphere
inline scalar_type sphereTriArea(const std::vector<Vec<3>>& vecs) {
    return sphereTriArea(vecs[0], vecs[1], vecs[2]);
}

template <int ndim> scalar_type polygonArea(const Vec<ndim>& ctr, const std::vector<Vec<ndim>>& ccwcorners){
	scalar_type result = 0.0;
	const int nverts = ccwcorners.size();
	for (int i=0; i<nverts; ++i) {
		result += triArea(ccwcorners[i], ctr, ccwcorners[(i+1)%nverts]);
	}
	return result;
}

template <int ndim> scalar_type spherePolygonArea(const Vec<ndim>& ctr, const std::vector<Vec<ndim>>& ccwcorners, 
	const scalar_type radius) {
	scalar_type result = 0.0;
	const int nverts = ccwcorners.size();
	for (int i=0; i<nverts; ++i) {
		result += sphereTriArea(ccwcorners[i], ctr, ccwcorners[(i+1)%nverts], radius);
	}
	return result;
}

/// Inverse tangent with quadrant information, but with output range in [0, 2*pi) instead of (-pi, pi]
scalar_type atan4(const scalar_type y, const scalar_type x);

/// Basic output to console
std::ostream& operator << (std::ostream& os, const Vec<1>& vec);
/// Basic output to console
std::ostream& operator << (std::ostream& os, const Vec<2>& vec);
/// Basic output to console
std::ostream& operator << (std::ostream& os, const Vec<3>& vec);

}
}
#endif
