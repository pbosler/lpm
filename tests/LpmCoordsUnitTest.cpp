#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <exception>
#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include "LpmCoords.hpp"

using namespace Lpm;

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {

    Coords<PlaneGeometry> pc(20);
    std::cout << "pc.nMax() = " << pc.nMax() << ", pc.n() = " << pc.n() << std::endl;
    Real a[2] = {-1, 1};
    Real b[2] = {-1, 0};
    Real c[2] = {-1, -1};
    Real d[2] = {0, -1};
    pc.insertHost(a);
    pc.insertHost(b);
    pc.insertHost(c);
    pc.insertHost(d);
    
    pc.updateDevice();
    std::cout << "pc.nMax() = " << pc.nMax() << ", pc.n() = " << pc.n() << std::endl;
    
    
    Coords<SphereGeometry> sc(20);
    std::cout << "sc.nMax() = " << sc.nMax() << ", sc.n() = " << sc.n() << std::endl;
    
    Real p0[3] = {0.57735026918962584,  -0.57735026918962584,  0.57735026918962584};
    Real p1[3] = {0.57735026918962584,  -0.57735026918962584,  -0.57735026918962584};
    Real p2[3] = {0.57735026918962584,  0.57735026918962584,  -0.57735026918962584};
    Real p3[3] = {0.57735026918962584,  0.57735026918962584, 0.57735026918962584};
    sc.insertHost(p0);
    sc.insertHost(p1);
    sc.insertHost(p2);
    sc.insertHost(p3);
    
    sc.updateDevice();
    std::cout << "sc.nMax() = " << sc.nMax() << ", sc.n() = " << sc.n() << std::endl;
    }
    std::cout << "tests pass." << std::endl;
    Kokkos::finalize();
return 0;
}