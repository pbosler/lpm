#include "LpmConfig.h"
#include "LpmDefs.hpp"
#include "LpmGeometry.hpp"
#include <typeinfo>

using namespace Lpm;

int main(int argc, char* argv[]) {
ko::initialize(argc, argv); 
{

    const Real r2a[2] = {1.0, 0.0};
    const Real r2b[2] = {2.0, 1.0};
    const Real r2c[2] = {-1.0, -10.0};
    
    typedef ko::DefaultExecutionSpace ExeSpace;
    // allocate a Vec on the device
    Vec<2, ExeSpace> r2device_a("a");
    // allocate a mirror on host
    Vec<2, ExeSpace>::HostMirror r2h = ko::create_mirror_view(r2device_a);
    // initialize on host
    r2h[0] = r2a[0];
    r2h[1] = r2a[1];
    // copy to device
    ko::deep_copy(r2device_a, r2h);
    
    Vec<2, ExeSpace> r2device_b("b");
    r2h[0] = r2b[0];
    r2h[1] = r2b[1];
    ko::deep_copy(r2device_b, r2h);
    
    Vec<2, ExeSpace> r2device_c("c");
    r2h[0] = r2c[0];
    r2h[1] = r2c[1];
    ko::deep_copy(r2device_c, r2h);
    
    // allocate a scalar on the device and a host mirror
    ko::View<Real,Layout,ExeSpace> scalar_result("res");
    ko::View<Real>::HostMirror host_scalar = ko::create_mirror_view(scalar_result);
    Vec<2, ExeSpace> result("res");
    Vec<2, ExeSpace>::HostMirror host_result = ko::create_mirror_view(result);
    VecArr<2, ExeSpace> vecs("vecs", 3);
    
    
    // execute functions on device
    ko::parallel_for(1, KOKKOS_LAMBDA (int i) {
        scalar_result(0) = PlaneGeometry::triArea(r2device_a, r2device_b, r2device_c);
        if (scalar_result(0) != 4) error("triArea error");
        for (int j=0; j<2; ++j) {
            vecs(0,j) = r2device_a(j);
            vecs(1,j) = r2device_b(j);
            vecs(2,j) = r2device_c(j);
        }
        PlaneGeometry::midpoint(result, slice(vecs, 0), slice(vecs, 1));
        printf("midpt = (%f,%f)\n", result(0), result(1));
        PlaneGeometry::barycenter(result, vecs, 3);
        scalar_result(0) = PlaneGeometry::distance(slice(vecs, 0), slice(vecs,2));
        if ( std::abs(scalar_result(0) - 2*std::sqrt(26.0)) > ZERO_TOL) error("distance error");
        scalar_result(0) = PlaneGeometry::dot(r2device_a, r2device_b);
        }); 
    // copy results to host
    ko::deep_copy(host_result, result);
    prarr("barycenter", host_result.data(), 2);
    ko::deep_copy(host_scalar, scalar_result);
    LPM_THROW_IF( host_scalar(0) != 2, "dot product error.");
    
}
ko::finalize();
return 0;
}
