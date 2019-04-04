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
    Vec<2, ExeSpace>::HostMirror ha = ko::create_mirror_view(r2device_a);
    // initialize on host
    ha[0] = r2a[0];
    ha[1] = r2a[1];
    // copy to device
    ko::deep_copy(r2device_a, ha);
    
    Vec<2, ExeSpace> r2device_b("b");
    Vec<2, ExeSpace>::HostMirror hb = ko::create_mirror_view(r2device_b);
    hb[0] = r2b[0];
    hb[1] = r2b[1];
    ko::deep_copy(r2device_b, hb);
    
    Vec<2, ExeSpace> r2device_c("c");
    Vec<2, ExeSpace>::HostMirror hc = ko::create_mirror_view(r2device_c);
    hc[0] = r2c[0];
    hc[1] = r2c[1];
    ko::deep_copy(r2device_c, hc);
    
    // allocate a scalar on the device and a host mirror
    ko::View<Real,Layout,ExeSpace> scalar_result("res");
    ko::View<Real>::HostMirror host_scalar = ko::create_mirror_view(scalar_result);
    Vec<2, ExeSpace> result("res");
    
    
    Vec<2, ExeSpace>::HostMirror host_result = ko::create_mirror_view(result);
    VecArr<2, ExeSpace> vecs("vecs", 3);
    std::cout << "shape(vecs) = (" << vecs.extent(0) << ", " << vecs.extent(1) << ")" << std::endl;
    
    std::cout << typeid(vecs).name() << std::endl;
    
    // execute functions on device
    ko::parallel_for(1, KOKKOS_LAMBDA (int i) {
        scalar_result(0) = PlaneGeometry::triArea<Vec<2,ExeSpace>>(r2device_a, r2device_b, r2device_c);
        if (scalar_result(0) != 4) printf("triArea error %f\n", scalar_result(0));
        for (int j=0; j<2; ++j) {
            vecs(0,j) = r2device_a(j);
            vecs(1,j) = r2device_b(j);
            vecs(2,j) = r2device_c(j);
        }
        PlaneGeometry::midpoint(result, slice(vecs, 0), slice(vecs, 1));
        if (result(0) != 1.5 || result(1) != 0.5) printf("midpt error (%f,%f)\n", result(0), result(1));
        PlaneGeometry::barycenter(result, vecs, 3);
        if (result(0) != 2.0/3.0 || result(1) != -3) printf("barycenter error (%f,%f)\n", result(0), result(1));
        scalar_result(0) = PlaneGeometry::distance(slice(vecs, 0), slice(vecs,2));
        if ( std::abs(scalar_result(0) - 2*std::sqrt(26.0)) > ZERO_TOL) printf("distance error: %f\n", scalar_result(0));
        scalar_result(0) = PlaneGeometry::dot(r2device_a, r2device_b);
        }); 
    // copy results to host
    ko::deep_copy(host_result, result);
    prarr("barycenter", host_result.data(), 2);
    LPM_THROW_IF( (host_result(0) != 2.0/3.0 || host_result(1) != -3), "barycenter error\n");
    ko::deep_copy(host_scalar, scalar_result);
    LPM_THROW_IF( host_scalar(0) != 2, "dot product error.\n");
    
}
std::cout << "tests pass." << std::endl;
ko::finalize();
return 0;
}
