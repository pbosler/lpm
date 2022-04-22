#include "lpm_2d_transport_rk4.hpp"
#include "util/lpm_tuple.hpp"
#include "KokkosBlas.hpp"

namespace Lpm {

struct RK4Update {
  Kokkos::View<Real**> xcrds;
  Kokkos::View<Real**> x1;
  Kokkos::View<Real**> x2;
  Kokkos::View<Real**> x3;
  Kokkos::View<Real**> x4;
  Real dt;

  RK4Update(Kokkos::View<Real**> x,
            const Kokkos::View<Real**> k1,
            const Kokkos::View<Real**> k2,
            const Kokkos::View<Real**> k3,
            const Kokkos::View<Real**> k4,
            const Real ddt) :
            xcrds(x),
            x1(k1),
            x2(k2),
            x3(k3),
            x4(k4),
            dt(ddt) {}

  KOKKOS_INLINE_FUNCTION
  void operator() (const Index i) const {
    for (int j=0; j<xcrds.extent(1); ++j) {
      xcrds(i,j) += dt * (x1(i,j) + 2*x2(i,j) + 2*x3(i,j) + x4(i,j))/6;
    }
  }
};

template <typename SeedType>
void Transport2dRK4<SeedType>::init() {
  vertx = Kokkos::subview(tmesh->vertices.phys_crds->crds, std::make_pair(0, nverts), Kokkos::ALL);
  vertvel = Kokkos::subview(tmesh->velocity_verts.view, std::make_pair(0, nverts), Kokkos::ALL);
  facex = Kokkos::subview(tmesh->faces.phys_crds->crds, std::make_pair(0, nfaces), Kokkos::ALL);
  facevel = Kokkos::subview(tmesh->velocity_faces.view, std::make_pair(0, nfaces), Kokkos::ALL);

  vertx1 = crd_view("vertx1", nverts);
  vertx2 = crd_view("vertx2", nverts);
  vertx3 = crd_view("vertx3", nverts);
  vertx4 = crd_view("vertx4", nverts);
  vertxwork = crd_view("vertex_work", nverts);

  facex1 = crd_view("facex1", nfaces);
  facex2 = crd_view("facex2", nfaces);
  facex3 = crd_view("facex3", nfaces);
  facex4 = crd_view("facex4", nfaces);
  facexwork = crd_view("facexwork", nfaces);

}

template <typename SeedType> template <typename VelocityFtor>
void Transport2dRK4<SeedType>::advance_timestep() {
  const Real t = tmesh->t;

  // RK Stage 1: velocity is already stored
  vertx1 = vertvel;
  facex1 = facevel;
  // RK Stage 2
  //  setup input in work arrays
  KokkosBlas::update(1.0, vertx, 0.5*dt, vertx1, 0, vertxwork);
  KokkosBlas::update(1.0, facex, 0.5*dt, facex1, 0, facexwork);
  //  compute updated velocity
  Kokkos::parallel_for(nverts, VelocityKernel<VelocityFtor>(vertx2, vertxwork, t+0.5*dt));
  Kokkos::parallel_for(nfaces, VelocityKernel<VelocityFtor>(facex2, facexwork, t+0.5*dt));
  // RK Stage 3
  //    setup input
  KokkosBlas::update(1.0, vertx, 0.5*dt, vertx2, 0, vertxwork);
  KokkosBlas::update(1.0, facex, 0.5*dt, facex2, 0, facexwork);
  //    compute velocity
  Kokkos::parallel_for(nverts, VelocityKernel<VelocityFtor>(vertx3, vertxwork, t + 0.5*dt));
  Kokkos::parallel_for(nfaces, VelocityKernel<VelocityFtor>(facex3, facexwork, t + 0.5*dt));
  //  RK Stage 4
  //    input
  KokkosBlas::update(1.0, vertx, dt, vertx3, 0, vertxwork);
  KokkosBlas::update(1.0, facex, dt, facex3, 0, facexwork);
  //    velocity
  Kokkos::parallel_for(nverts, VelocityKernel<VelocityFtor>(vertx4, vertxwork, t+dt));
  Kokkos::parallel_for(nfaces, VelocityKernel<VelocityFtor>(facex4, facexwork, t+dt));
  //  Update positions
  Kokkos::parallel_for(nverts, RK4Update(vertx, vertx1, vertx2, vertx3, vertx4, dt));
  Kokkos::parallel_for(nfaces, RK4Update(facex, facex1, facex2, facex3, facex4, dt));

  tmesh->t_idx++;
  tmesh->t = tmesh->t_idx * dt;
  //  Set velocity
  Kokkos::parallel_for(nverts, VelocityKernel<VelocityFtor>(vertvel, vertx, tmesh->t));
  Kokkos::parallel_for(nfaces, VelocityKernel<VelocityFtor>(facevel, facex, tmesh->t));
}

}
