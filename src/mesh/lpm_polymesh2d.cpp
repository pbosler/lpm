#include "mesh/lpm_polymesh2d.hpp"

#include "lpm_kokkos_defs.hpp"
#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#endif
#include <iostream>

#include "lpm_constants.hpp"
#include "util/lpm_floating_point.hpp"

namespace Lpm {

template <typename SeedType>
void PolyMesh2d<SeedType>::seed_init(const MeshSeed<SeedType>& seed) {
  vertices.phys_crds->init_vert_crds_from_seed(seed);
  vertices.lag_crds->init_vert_crds_from_seed(seed);
  for (int i = 0; i < SeedType::nverts; ++i) {
    vertices.insert_host(i);
  }
  edges.init_from_seed(seed);
  faces.init_from_seed(seed);
  faces.phys_crds->init_interior_crds_from_seed(seed);
  faces.lag_crds->init_interior_crds_from_seed(seed);
}

template <typename SeedType>
void PolyMesh2d<SeedType>::tree_init(const Int initDepth,
                                     const MeshSeed<SeedType>& seed) {
  seed_init(seed);
  base_tree_depth = initDepth;

  Index startInd = 0;
  for (int i = 0; i < initDepth; ++i) {
    Index stopInd = faces.nh();
    for (Index j = startInd; j < stopInd; ++j) {
      if (!faces.has_kids_host(j)) {
        divider::divide(j, vertices, edges, faces);
      }
    }
    startInd = stopInd - 1;
  }
  update_device();
}

template <typename SeedType>
template <typename LoggerType>
void PolyMesh2d<SeedType>::divide_face(const Index face_idx,
                                       LoggerType& logger) {
  if (faces.has_kids_host(face_idx)) {
    logger.warn("divide_face: face {} has already been divided.", face_idx);
  } else {
    divider::divide(face_idx, vertices, edges, faces);
  }
}

template <typename SeedType>
void PolyMesh2d<SeedType>::reset_face_centroids() {
  ko::parallel_for(n_faces_host(), FaceCentroidFunctor<SeedType>(
                                       faces.phys_crds->crds, faces.crd_inds,
                                       vertices.phys_crds->crds,
                                       vertices.crd_inds, faces.verts));
}

#ifdef LPM_USE_VTK
template <typename SeedType>
void PolyMesh2d<SeedType>::output_vtk(const std::string& fname) const {
  VtkInterface<Geo, FaceType> vtk;
  auto cd = vtkSmartPointer<vtkCellData>::New();
  vtk.addScalarToCellData(cd, faces.area_host(), "area", faces);
  vtkSmartPointer<vtkPolyData> pd =
      vtk.toVtkPolyData(faces, edges, vertices, NULL, cd);
  vtk.writePolyData(fname, pd);
}
#endif

template <typename SeedType>
void PolyMesh2d<SeedType>::update_device() const {
  vertices.update_device();
  edges.update_device();
  faces.update_device();
}

template <typename SeedType>
void PolyMesh2d<SeedType>::update_host() const {
  vertices.update_host();
  edges.update_host();
  faces.update_host();
}

template <typename SeedType>
template <typename VT>
void PolyMesh2d<SeedType>::get_leaf_face_crds(VT leaf_crds) const {
  faces.leaf_crd_view(leaf_crds);
}

template <typename SeedType>
typename SeedType::geo::crd_view_type PolyMesh2d<SeedType>::get_leaf_face_crds()
    const {
  typename SeedType::geo::crd_view_type result("face_leaf_crds",
                                               n_faces_host());
  faces.leaf_crd_view(result);
  return result;
}

template <typename SeedType>
std::string PolyMesh2d<SeedType>::info_string(const std::string& label,
                                              const int& tab_level,
                                              const bool& dump_all) const {
  std::ostringstream ss;
  ss << "\nPolyMesh2d<" << SeedType::id_string() << "> " << label << " info:\n";
  ss << vertices.info_string(label, tab_level + 1, dump_all);
  ss << edges.info_string(label, tab_level + 1, dump_all);
  ss << faces.info_string(label, tab_level + 1, dump_all);
  return ss.str();
}

/// ETI
template class PolyMesh2d<TriHexSeed>;
template class PolyMesh2d<QuadRectSeed>;
template class PolyMesh2d<IcosTriSphereSeed>;
template class PolyMesh2d<CubedSphereSeed>;
}  // namespace Lpm
