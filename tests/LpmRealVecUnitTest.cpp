#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <exception>
#include "LpmConfig.h"
#include "LpmTypeDefs.hpp"
#include "LpmUtilities.hpp"
#include "LpmRealVector.hpp"

using namespace Lpm;

int main (int argc, char* argv[]) {
    
    RealVec<3> v3def;
    std::cout << "default constructor: " <<  v3def << std::endl;
    
    const Real xxa[3] = {0.0, 1.0, 2.0};
    const Real xxb[3] = {1.0, -1.0, 5.0};
    const std::vector<Real> xxc = {-10.0, -100.0, -1000.0};
    
    RealVec<2> v2a(xxa);
    RealVec<2> v2b(xxb);
    RealVec<3> v3a(xxa);
    RealVec<3> v3b(xxb);
    RealVec<3> v3copy(v3a);
    RealVec<2> v2fromvec(xxc);
    RealVec<2> v2assign = v2a;
    if (v3a == v3copy) {
        std::cout << "copy constructor and == operator test passed." << std::endl;
    }
    else {
        throw std::logic_error("either copy constructor or == operator test failed.");
    }
    
    if (v2a.mag() != 1.0) {
        throw std::logic_error("magnitude error.");
    }
    
    if (v2a.crossProdComp3(v2b) != -1) {
        throw std::logic_error("cp component 3 error.");
    }
    
    if ( (v3a + v3b) != RealVec<3>(1,0,7)) {
        throw std::logic_error("RealVec sum error.");
    }
    
    if (v2assign.normalize().mag() != 1) {
        throw std::logic_error("Normalize error.");
    }
    
    std::cout << "v3a.midpoint(v3b) = " << v3a.midpoint(v3b) << std::endl;
    std::cout << "v3a.dotProd(v3b) = " << v3a.dotProd(v3b) << std::endl;
    std::cout << "v3a / v3c = " << v3a / RealVec<3>(xxc) << std::endl;
    std::cout << "v3a * v3c = " << v3a * RealVec<3>(xxc) << std::endl;
    std::cout << "v3a = v3copy : " << (v3a == v3copy ? "True" : "False") << std::endl;
    
    RealVec<3> v3spha = v3a.normalize();
    RealVec<3> v3sphb = v3b.normalize();
    if (std::abs(v3spha.longitude() - 0.5*PI) > ZERO_TOL ) {
        throw std::logic_error("longitude error.");
    }
    std::cout << "v3a.normalize() " << v3a.normalize() << std::endl;
    std::cout << "v3a.normalize().magSq() = " << v3a.normalize().magSq() << std::endl;
    std::cout << "v3spha.longitude(), latitude() = (" << v3spha.longitude() << ", " 
        << v3spha.latitude() << ")" << std::endl;
    std::cout << "v3spha.sphereMidpoint(v3sphb), magSq = " << v3spha.sphereMidpoint(v3sphb) << ", " 
        << v3spha.sphereMidpoint(v3sphb).magSq() << std::endl;
    std::cout << "v3spha.sphereDist(v3sphB) = " << v3spha.sphereDist(v3sphb) << std::endl;
    std::cout << "v3spha.dist(v3sphB) = " << v3spha.dist(v3sphb) << std::endl;
    
    std::vector<RealVec<3>> vecs = {v3a, v3b, RealVec<3>(xxc)};
    std::cout << "barycenter = " << barycenter(vecs) << std::endl;
    std::cout << "triArea = " << triArea(vecs) << std::endl;
    
    for (int i=0; i<3; ++i) 
        vecs[i].normalizeInPlace();
    std::cout << "sphereBarycetner = " << sphereBarycenter(vecs) << std::endl;
    std::cout << "sphereTriArea = " << sphereTriArea(vecs) << std::endl;
return 0;
}
