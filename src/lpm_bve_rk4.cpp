#include "lpm_bve_rk4.hpp"

namespace Lpm {

void BVERK4::init(const Index nv, const Index nf) {

  if (nv != nverts or !is_initialized) {
    vertx1 = crd_view("vertex_xyz_stage1", nv);
    vertx2 = crd_view("vertex_xyz_stage2", nv);
    vertx3 = crd_view("vertex_xyz_stage3", nv);
    vertx4 = crd_view("vertex_xyz_stage4", nv);
    vertxwork = crd_view("vertex_xyz_workspace", nv);

    vertvort1 = scalar_view_type("vertex_vorticity_stage1", nv);
    vertvort2 = scalar_view_type("vertex_vorticity_stage2", nv);
    vertvort3 = scalar_view_type("vertex_vorticity_stage3", nv);
    vertvort4 = scalar_view_type("vertex_vorticity_stage4", nv);
    vertvortwork = scalar_view_type("vertex_vorticity_workspace", nv);

    nverts = nv;
  }

  if (nf != nfaces or !is_initialized) {
    facex1 = crd_view("face_xyz_stage1", nf);
    facex2 = crd_view("face_xyz_stage2", nf);
    facex3 = crd_view("face_xyz_stage3", nf);
    facex4 = crd_view("face_xyz_stage4", nf);
    facexwork = crd_view("face_xyz_workspace", nf);

    facevort1 = scalar_view_type("face_vorticity_stage1",nf);
    facevort2 = scalar_view_type("face_vorticity_stage2",nf);
    facevort3 = scalar_view_type("face_vorticity_stage3",nf);
    facevort4 = scalar_view_type("face_vorticity_stage4",nf);
    facevortwork = scalar_view_type("face_vorticity_workspace", nf);

    nfaces = nf;
  }

  is_initialized = true;
}

};
