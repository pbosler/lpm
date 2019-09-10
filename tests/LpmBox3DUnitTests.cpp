#include "LpmConfig.h"

#include "LpmDefs.hpp"
#include "LpmBox3d.hpp"
#include "Kokkos_Core.hpp"
#include <iostream>
#include <string>

using namespace Lpm;

int main(int argc, char* argv[]) {
ko::initialize(argc, argv); 
{
    typedef ko::View<Real[6]> box_type;
    typedef typename box_type::HostMirror box_host;
    typedef ko::View<Real*[6]> boxes_type;
    typedef typename boxes_type::HostMirror boxes_host;
    typedef ko::View<Real[3]> pt_type;
    typedef typename pt_type::HostMirror host_pt;
    typedef ko::View<Real*[3]> pts_type;
    typedef typename pts_type::HostMirror host_pts;
    typedef ko::View<bool> bool_view;
    typedef typename bool_view::HostMirror host_bool;
    
    box_type box_std("unit_box");
    box_host host_std = ko::create_mirror_view(host_std);
    
    for (short i=0; i<6; ++i)
        host_std(i) = (i%2 == 0 ? -1.0 : 1.0);
    ko::deep_copy(box_std, host_std);
        
    pt_type origin("origin");
    host_pt ohost = ko::create_mirror_view(origin);
    ohost(0) = 0;
    ohost(1) = 0;
    ohost(2) = 0;
    ko::deep_copy(origin, ohost);
    
    pt_type far("far");
    host_pt fhost = ko::create_mirror_view(far);
    far(0) = 10;
    far(1) = 10;
    far(2) = 10;
    ko::deep_copy(far, fhost);
    
    bool_view origin_inside("origin_inside");
    host_bool oi_host = ko::create_mirror_view(origin_inside);
    bool_view far_outside("far_outside");
    host_bool fo_host = ko::create_mirror_view(far_outside);
    ko::parallel_for(1, KOKKOS_LAMBDA (const int& i) {
        origin_inside() = boxContainsPoint(box_std, origin);
        far_outside() = !boxContainsPoint(box_std, far);
    });
    
    ko::deep_copy(oi_host, origin_inside);
    ko::deep_copy(fo_host, far_outside);
    if (!oi_host()) {
        std::cout << "ERROR: origin not found inside root box.\n";
    }
    if (!fo_host()) {
        std::cout << "ERROR: (10,10,10) found inside root box.\n";
    }
    
    boxes_type kid_boxes("kid_boxes",8);
    boxes_host kids_host = ko::create_mirror_view(kid_boxes);

    
}
ko::finalize();    
}
