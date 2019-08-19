#include "LpmUtilities.hpp"
#include <cmath>
#include <limits>

namespace Lpm {
  void quadraticRoots(Real& r1, Real& r2, const Real a, const Real b, const Real c) {
     const Real apa = 2.0 * a;
     Real disc = b*b - 4.0*a*c;
     if (std::abs(disc) < ZERO_TOL) {
        disc = 0.0;
     }
     else if (disc < -ZERO_TOL) {
        r1 = std::numeric_limits<Real>::max();
        r2 = std::numeric_limits<Real>::max();
     }
     else {
        const Real rdisc = std::sqrt(disc);
        r1 = (-b + rdisc)/apa;
        r2 = (-b - rdisc)/apa;
     }
  }


std::string weightName(const int ndim) {
    std::string result;
    switch(ndim) {
        case (1) : {result = "length"; break;}
        case (2) : {result = "area"; break;}
        case (3) : {result = "volume"; break;}
    }
    return result;
}

}
