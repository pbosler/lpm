#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <exception>
#include "LpmConfig.h"
#include "LpmTypeDefs.hpp"
#include "LpmCoords.hpp"
#include "LpmRealVector.hpp"

using namespace Lpm;

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {

    Coords<2, PLANAR_GEOMETRY> pc(20);
    std::cout << "pc.nMax() = " << pc.nMax() << ", pc.n() = " << pc.n() << std::endl;
    pc.initRandom(2.0);
    std::cout << "pc.n() = " << pc.n() << std::endl;
    std::cout << "pc.dist(5,11) = " << pc.distance(5, 11) << std::endl;
    
    IndexArray inds("inds", 3);
    IndexArray::HostMirror hinds = Kokkos::create_mirror_view(inds);
    for (int i=0; i<3; ++i) {
        hinds(i) = 3*i;
    }
    Kokkos::deep_copy(inds, hinds);
    
    Kokkos::View<Real[2]> result_view("device_result");
    RealVec<2> bc = pc.barycenter(inds);
    result_view(0) = bc[0];
    result_view(1) = bc[1];
    Kokkos::View<Real[2]>::HostMirror host_result = Kokkos::create_mirror_view(result_view);
    Kokkos::deep_copy(host_result, result_view);
    RealVec<2> bchost;
    bchost[0] = host_result(0);
    bchost[1] = host_result(1);
    std::cout << "pc.barycenter(inds) = " << bchost << std::endl;
    
    Kokkos::View<Real> dev_tri("device_tri_area");
    dev_tri(0) = pc.triArea(inds(0), inds(1), inds(2));
    Kokkos::View<Real>::HostMirror host_tri = Kokkos::create_mirror_view(dev_tri);
    Kokkos::deep_copy(host_tri, dev_tri);
    std::cout << "host tri_area = " << host_tri(0) << std::endl;
    
    Coords<3, SPHERICAL_SURFACE_GEOMETRY> sc(20);
    sc.initRandom(1.0,0);
    const RealVec<3> mp = sc.midpoint(1,2);
    std::cout << "sc.midpoint(1,2) = " << mp << ", len = " << mp.mag() << std::endl;
    std::cout << "sc.distance(5, 11) = " << sc.distance(5,11) << std::endl;

    dev_tri(0) = sc.triArea(inds(0), inds(1), inds(2));
    Kokkos::deep_copy(host_tri, dev_tri);
    std::cout << "host_tri_area (sphere) = " << host_tri(0) << std::endl;
    
    }
    Kokkos::finalize();
return 0;
}