#include <iostream>
#include <sstream>
#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmCoords.hpp"

using namespace Lpm;

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {

    Coords<PlaneGeometry> pc4(4);
    std::cout << "pc4.nMax() = " << pc4.nMax() << ", pc4.nh() = " << pc4.nh() << std::endl;
    Real a[2] = {-1, 1};
    Real b[2] = {-1, 0};
    Real c[2] = {-1, -1};
    Real d[2] = {0, -1};
    pc4.insertHost(a);
    pc4.insertHost(b);
    pc4.insertHost(c);
    pc4.insertHost(d);
    
    pc4.updateDevice();
    std::cout << "pc4.nMax() = " << pc4.nMax() << ", pc4.nh() = " << pc4.nh() << std::endl;
    pc4.printcrds("pc4");
    
    Coords<PlaneGeometry> pcr(20);
    pcr.initRandom(3.0);
    pcr.printcrds("pcr");
    
    
    Coords<SphereGeometry> sc4(4);
    std::cout << "sc4.nMax() = " << sc4.nMax() << ", sc4.nh() = " << sc4.nh() << std::endl;
    
    const Real p0[3] = {0.57735026918962584,  -0.57735026918962584,  0.57735026918962584};
    const Real p1[3] = {0.57735026918962584,  -0.57735026918962584,  -0.57735026918962584};
    const Real p2[3] = {0.57735026918962584,  0.57735026918962584,  -0.57735026918962584};
    const Real p3[3] = {0.57735026918962584,  0.57735026918962584, 0.57735026918962584};
    sc4.insertHost(p0);
    sc4.insertHost(p1);
    sc4.insertHost(p2);
    sc4.insertHost(p3);
    std::cout << "updating sc4." << std::endl;
    sc4.updateDevice();
    std::cout << "sc4.nMax() = " << sc4.nMax() << ", sc4.nh() = " << sc4.nh() << std::endl;
    sc4.printcrds("sc4");
    
    Coords<SphereGeometry> scr(20);
    scr.initRandom();
    std::cout<<"scr done." << std::endl;
    scr.printcrds("scr");
    
    }
    std::cout << "tests pass." << std::endl;
    Kokkos::finalize();
return 0;
}