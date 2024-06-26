#ifndef LPM_SWERK4_IMPL_HPP
#define LPM_SWERK4_IMPL_HPP

#include "KokkosBlas.hpp"
#include "LpmSWEKernels.hpp"
#include "LpmSWERK4.hpp"
#include "LpmUtilities.hpp"

namespace Lpm {

template <typename SeedType, typename ProblemType>
std::string SWERK4<SeedType, ProblemType>::infoString(
    const std::string& label, const int& tab_level) const {
  std::ostringstream ss;
  ss << (label.empty() ? "SWERK4 info:\n" : label + "\n");
  const std::string tabstr = indentString(tab_level + 1);
  ss << "vertx: (label, ext(0), ext(1)) = (" << vertx.label() << ", "
     << vertx.extent(0) << ", " << vertx.extent(1) << ")\n";
  ss << "vertx1: (label, ext(0), ext(1)) = (" << vertx1.label() << ", "
     << vertx1.extent(0) << ", " << vertx1.extent(1) << ")\n";
  ss << "vertx2: (label, ext(0), ext(1)) = (" << vertx2.label() << ", "
     << vertx2.extent(0) << ", " << vertx2.extent(1) << ")\n";
  ss << "vertx3: (label, ext(0), ext(1)) = (" << vertx3.label() << ", "
     << vertx3.extent(0) << ", " << vertx3.extent(1) << ")\n";
  ss << "vertx4: (label, ext(0), ext(1)) = (" << vertx4.label() << ", "
     << vertx4.extent(0) << ", " << vertx4.extent(1) << ")\n";
  ss << "vertvel: (label, ext(0), ext(1)) = (" << vertvel.label() << ", "
     << vertvel.extent(0) << ", " << vertvel.extent(1) << ")\n";
  ss << "facex: (label, ext(0), ext(1)) = (" << facex.label() << ", "
     << facex.extent(0) << ", " << facex.extent(1) << ")\n";
  ss << "facex1: (label, ext(0), ext(1)) = (" << facex1.label() << ", "
     << facex1.extent(0) << ", " << facex1.extent(1) << ")\n";
  ss << "facex2: (label, ext(0), ext(1)) = (" << facex2.label() << ", "
     << facex2.extent(0) << ", " << facex2.extent(1) << ")\n";
  ss << "facex3: (label, ext(0), ext(1)) = (" << facex3.label() << ", "
     << facex3.extent(0) << ", " << facex3.extent(1) << ")\n";
  ss << "facex4: (label, ext(0), ext(1)) = (" << facex4.label() << ", "
     << facex4.extent(0) << ", " << facex4.extent(1) << ")\n";
  ss << "facevel: (label, ext(0), ext(1)) = (" << facevel.label() << ", "
     << facevel.extent(0) << ", " << facevel.extent(1) << ")\n";
  return ss.str();
}

template <typename SeedType, typename ProblemType>
void SWERK4<SeedType, ProblemType>::init() {
  ;
  vertex_policy = std::unique_ptr<ko::TeamPolicy<>>(
      new ko::TeamPolicy<>(nverts, ko::AUTO()));
  face_policy = std::unique_ptr<ko::TeamPolicy<>>(
      new ko::TeamPolicy<>(nfaces, ko::AUTO()));

  vertx1 = crd_view("vertx1", nverts);
  vertx2 = crd_view("vertx2", nverts);
  vertx3 = crd_view("vertx3", nverts);
  vertx4 = crd_view("vertx4", nverts);
  vertxwork = crd_view("vertxwork", nverts);

  vertvort1 = scalar_view_type("vertvort1", nverts);
  vertvort2 = scalar_view_type("vertvort2", nverts);
  vertvort3 = scalar_view_type("vertvort3", nverts);
  vertvort4 = scalar_view_type("vertvort4", nverts);
  vertvortwork = scalar_view_type("vertvortwork", nverts);

  vertdiv1 = scalar_view_type("vertdiv1", nverts);
  vertdiv2 = scalar_view_type("vertdiv2", nverts);
  vertdiv3 = scalar_view_type("vertdiv3", nverts);
  vertdiv4 = scalar_view_type("vertdiv4", nverts);
  vertdivwork = scalar_view_type("vertdivwork", nverts);

  verth1 = scalar_view_type("verth1", nverts);
  verth2 = scalar_view_type("verth2", nverts);
  verth3 = scalar_view_type("verth3", nverts);
  verth4 = scalar_view_type("verth4", nverts);
  verthwork = scalar_view_type("verthwork", nverts);

  vertddot = scalar_view_type("vertddot", nverts);
  vertlaps = scalar_view_type("vertlaps", nverts);

  facex1 = crd_view("facex1", nfaces);
  facex2 = crd_view("facex2", nfaces);
  facex3 = crd_view("facex3", nfaces);
  facex4 = crd_view("facex4", nfaces);
  facexwork = crd_view("facexwork", nfaces);

  facevort1 = scalar_view_type("facevort1", nfaces);
  facevort2 = scalar_view_type("facevort2", nfaces);
  facevort3 = scalar_view_type("facevort3", nfaces);
  facevort4 = scalar_view_type("facevort4", nfaces);
  facevortwork = scalar_view_type("facevortwork", nfaces);

  facediv1 = scalar_view_type("facediv1", nfaces);
  facediv2 = scalar_view_type("facediv2", nfaces);
  facediv3 = scalar_view_type("facediv3", nfaces);
  facediv4 = scalar_view_type("facediv4", nfaces);
  facedivwork = scalar_view_type("facedivwork", nfaces);

  faceddot = scalar_view_type("faceddot", nfaces);
  facelaps = scalar_view_type("facelaps", nfaces);

  facearea1 = scalar_view_type("facearea1", nfaces);
  facearea2 = scalar_view_type("facearea2", nfaces);
  facearea3 = scalar_view_type("facearea3", nfaces);
  facearea4 = scalar_view_type("facearea4", nfaces);
  faceareawork = scalar_view_type("faceareawork", nfaces);
}

template <typename SeedType, typename ProblemType>
void SWERK4<SeedType, ProblemType>::advance_timestep() {
  /// RK Stage 1
  ko::parallel_for(
      "VertexSums-RK1", *vertex_policy,
      PlanarSWEVertexSums(vertvel, vertddot, vertlaps, vertx, vertsfc, facex,
                          facevort, facediv, facearea, facesfc, eps_pse));
  ko::parallel_for(
      "PlanarSWEFaceSums-RK1", *face_policy,
      PlanarSWEFaceSums(facevel, faceddot, facelaps, facex, facevort, facediv,
                        facearea, facesfc, eps_pse));

  ko::parallel_for(
      "VertexRHS-RK1", nverts,
      PlanarSWEVertexRHS(vertx1, vertvort1, vertdiv1, verth1, vertx, vertvel,
                         vertvort, vertdiv, vertddot, vertlaps, vertdepth, f0,
                         beta, g, dt));
  ko::parallel_for(
      "PlanarSWEFaceRHS", nfaces,
      PlanarSWEFaceRHS(facex1, facevort1, facediv1, facearea1, facex, facevel,
                       facevort, facediv, faceddot, facelaps, facearea,
                       facemask, f0, beta, g, dt));

  /// RK Stage 2
  KokkosBlas::update(1.0, vertx, 0.5, vertx1, 0.0, vertxwork);
  KokkosBlas::update(1.0, vertvort, 0.5, vertvort1, 0.0, vertvortwork);
  KokkosBlas::update(1.0, vertdiv, 0.5, vertdiv1, 0.0, vertdivwork);
  KokkosBlas::update(1.0, vertdepth, 0.5, verth1, 0.0, verthwork);

  KokkosBlas::update(1.0, facex, 0.5, facex1, 0.0, facexwork);
  KokkosBlas::update(1.0, facevort, 0.5, facevort1, 0.0, facevortwork);
  KokkosBlas::update(1.0, facediv, 0.5, facediv1, 0.0, facedivwork);
  KokkosBlas::update(1.0, facearea, 0.5, facearea1, 0.0, faceareawork);

  update_sfc();
  compute_direct_sums();

  ko::parallel_for(
      "VertexRHS-RK2", nverts,
      PlanarSWEVertexRHS(vertx2, vertvort2, vertdiv2, verth2, vertxwork,
                         vertvel, vertvortwork, vertdivwork, vertddot, vertlaps,
                         verthwork, f0, beta, g, dt));
  ko::parallel_for(
      "FaceRHS-RK2", nfaces,
      PlanarSWEFaceRHS(facex2, facevort2, facediv2, facearea2, facexwork,
                       facevel, facevortwork, facedivwork, faceddot, facelaps,
                       faceareawork, facemask, f0, beta, g, dt));

  KokkosBlas::update(1.0, vertx, 0.5, vertx2, 0.0, vertxwork);
  KokkosBlas::update(1.0, vertvort, 0.5, vertvort2, 0.0, vertvortwork);
  KokkosBlas::update(1.0, vertdiv, 0.5, vertdiv2, 0.0, vertdivwork);
  KokkosBlas::update(1.0, vertdepth, 0.5, verth2, 0.0, verthwork);

  KokkosBlas::update(1.0, facex, 0.5, facex2, 0.0, facexwork);
  KokkosBlas::update(1.0, facevort, 0.5, facevort2, 0.0, facevortwork);
  KokkosBlas::update(1.0, facediv, 0.5, facediv2, 0.0, facedivwork);
  KokkosBlas::update(1.0, facearea, 0.5, facearea2, 0.0, faceareawork);

  update_sfc();
  compute_direct_sums();

  ko::parallel_for(
      "VertexRHS-RK3", nverts,
      PlanarSWEVertexRHS(vertx3, vertvort3, vertdiv3, verth3, vertxwork,
                         vertvel, vertvortwork, vertdivwork, vertddot, vertlaps,
                         verthwork, f0, beta, g, dt));
  ko::parallel_for(
      "FaceRHS-RK3", nfaces,
      PlanarSWEFaceRHS(facex3, facevort3, facediv3, facearea3, facexwork,
                       facevel, facevortwork, facedivwork, faceddot, facelaps,
                       faceareawork, facemask, f0, beta, g, dt));

  KokkosBlas::update(1.0, vertx, 1.0, vertx3, 0.0, vertxwork);
  KokkosBlas::update(1.0, vertvort, 1.0, vertvort3, 0.0, vertvortwork);
  KokkosBlas::update(1.0, vertdiv, 1.0, vertdiv3, 0.0, vertdivwork);
  KokkosBlas::update(1.0, vertdepth, 1.0, verth3, 0.0, verthwork);

  KokkosBlas::update(1.0, facex, 1.0, facex3, 0.0, facexwork);
  KokkosBlas::update(1.0, facevort, 1.0, facevort3, 0.0, facevortwork);
  KokkosBlas::update(1.0, facediv, 1.0, facediv3, 0.0, facedivwork);
  KokkosBlas::update(1.0, facearea, 1.0, facearea3, 0.0, faceareawork);

  update_sfc();
  compute_direct_sums();

  ko::parallel_for(
      "VertexRHS-RK4", nverts,
      PlanarSWEVertexRHS(vertx4, vertvort4, vertdiv4, verth4, vertxwork,
                         vertvel, vertvortwork, vertdivwork, vertddot, vertlaps,
                         verthwork, f0, beta, g, dt));
  ko::parallel_for(
      "FaceRHS-RK4", nfaces,
      PlanarSWEFaceRHS(facex4, facevort4, facediv4, facearea4, facexwork,
                       facevel, facevortwork, facedivwork, faceddot, facelaps,
                       faceareawork, facemask, f0, beta, g, dt));

  ko::parallel_for("VertPositionUpdate",
                   ko::MDRangePolicy<ko::Rank<2>>({0, 0}, {nverts, 2}),
                   PositionUpdate(vertx, vertx1, vertx2, vertx3, vertx4));
  ko::parallel_for(
      "VertVortUpdate", nverts,
      ScalarUpdate(vertvort, vertvort1, vertvort2, vertvort3, vertvort4));
  ko::parallel_for(
      "VertDivUpdate", nverts,
      ScalarUpdate(vertdiv, vertdiv1, vertdiv2, vertdiv3, vertdiv4));
  ko::parallel_for("VertHUpdate", nverts,
                   ScalarUpdate(vertdepth, verth1, verth2, verth3, verth4));
  ko::parallel_for(
      "VertSfcUpdate", nverts,
      PlanarSWESetVertexSfc<ProblemType>(vertsfc, verttopo, vertdepth, vertx));

  ko::parallel_for("FacePositionUpdate",
                   ko::MDRangePolicy<ko::Rank<2>>({0, 0}, {nfaces, 2}),
                   PositionUpdate(facex, facex1, facex2, facex3, facex4));
  ko::parallel_for(
      "FaceVortUpdate", nfaces,
      ScalarUpdate(facevort, facevort1, facevort2, facevort3, facevort4));
  ko::parallel_for(
      "FaceDivUpdate", nfaces,
      ScalarUpdate(facediv, facediv1, facediv2, facediv3, facediv4));
  ko::parallel_for(
      "FaceAreaUpdate", nfaces,
      ScalarUpdate(facearea, facearea1, facearea2, facearea3, facearea4));
  ko::parallel_for(
      "FaceSfcUpdate", nfaces,
      PlanarSWESetFaceSfc<ProblemType>(facesfc, facedepth, facetopo, facemass,
                                       facearea, facemask, facex));
}

template <typename SeedType, typename ProblemType>
void SWERK4<SeedType, ProblemType>::compute_direct_sums() {
  ko::parallel_for(
      "VertexSums", *vertex_policy,
      PlanarSWEVertexSums(vertvel, vertddot, vertlaps, vertxwork, vertsfc,
                          facexwork, facevortwork, facedivwork, faceareawork,
                          facesfc, eps_pse));
  ko::parallel_for(
      "FaceSums", *face_policy,
      PlanarSWEFaceSums(facevel, faceddot, facelaps, facexwork, facevortwork,
                        facedivwork, faceareawork, facesfc, eps_pse));
}

template <typename SeedType, typename ProblemType>
void SWERK4<SeedType, ProblemType>::update_sfc() {
  ko::parallel_for("VertexSfcUpdate", nverts,
                   PlanarSWESetVertexSfc<ProblemType>(vertsfc, verttopo,
                                                      verthwork, vertxwork));
  ko::parallel_for(
      "FaceSfcUpdate", nfaces,
      PlanarSWESetFaceSfc<ProblemType>(facesfc, facedepth, facetopo, facemass,
                                       facearea, facemask, facexwork));
}

}  // namespace Lpm
#endif
