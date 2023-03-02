#ifndef LPM_2DTRANSPORT_RK4_HPP
#define LPM_2DTRANSPORT_RK4_HPP

#include "LpmConfig.h"
#include "lpm_2d_transport_mesh.hpp"

namespace Lpm {

template <typename SeedType>
class Transport2dRK4 {
 public:
  typedef typename SeedType::geo::crd_view_type crd_view;
  typedef typename SeedType::geo::vec_view_type vec_view;

  crd_view vertx;
  vec_view vertvel;

  crd_view facex;
  vec_view facevel;

  Real dt;

  Index nverts;
  Index nfaces;

  Transport2dRK4(const Real timestep,
                 std::shared_ptr<TransportMesh2d<SeedType>> m)
      : tmesh(m),
        dt(timestep),
        nverts(m->n_vertices_host()),
        nfaces(m->n_faces_host()) {
    init();
  }

  template <typename VelocityFtor>
  void advance_timestep();

 protected:
  std::shared_ptr<TransportMesh2d<SeedType>> tmesh;

  void init();

  crd_view vertx1;
  crd_view vertx2;
  crd_view vertx3;
  crd_view vertx4;
  crd_view vertxwork;

  crd_view facex1;
  crd_view facex2;
  crd_view facex3;
  crd_view facex4;
  crd_view facexwork;
};

}  // namespace Lpm

#endif
