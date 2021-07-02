#include "mesh/lpm_polymesh2d.hpp"
#include "lpm_kokkos_defs.hpp"
#include "vtk/lpm_vtk_io.hpp"
#include "util/lpm_floating_point_util.hpp"
#include "lpm_constants.hpp"
#include <iostream>

namespace Lpm {

template <typename SeedType>
void PolyMesh2d<SeedType>::seed_init(const MeshSeed<SeedType>& seed) {
    vertices.phys_crds->init_vert_crds_from_seed(seed);
    vertices.lag_crds->init_vert_crds_from_seed(seed);
    for (int i=0; i<SeedType::nverts; ++i) {
      vertices.insert_host(i);
    }
    edges.init_from_seed(seed);
    faces.init_from_seed(seed);
    faces.phys_crds->init_interior_crds_from_seed(seed);
    faces.lag_crds->init_interior_crds_from_seed(seed);
}

template <typename SeedType>
void PolyMesh2d<SeedType>::tree_init(const Int initDepth, const MeshSeed<SeedType>& seed) {
    seed_init(seed);
    base_tree_depth=initDepth;

    for (int i=0; i<initDepth; ++i) {
        Index startInd = 0;
        Index stopInd = faces.nh();
        for (Index j=startInd; j<stopInd; ++j) {
            if (!faces.has_kids_host(j)) {
                divider::divide(j, vertices, edges, faces);
            }
        }
    }
    update_device();
}

template <typename SeedType>
void PolyMesh2d<SeedType>::reset_face_centroids() {
  ko::parallel_for(n_faces_host(),
    FaceCentroidFunctor<SeedType>(faces.phys_crds->crds,
        faces.crd_inds,
        vertices.phys_crds->crds,
        vertices.crd_inds,
        faces.verts)
        );
}

template <typename SeedType>
void PolyMesh2d<SeedType>::output_vtk(const std::string& fname) const {
    VtkInterface<Geo,FaceType> vtk;
    auto cd = vtkSmartPointer<vtkCellData>::New();
    vtk.addScalarToCellData(cd, faces.area_host(), "area", faces);
    vtkSmartPointer<vtkPolyData> pd = vtk.toVtkPolyData(faces, edges, vertices, NULL, cd);
    vtk.writePolyData(fname, pd);
}

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
std::string PolyMesh2d<SeedType>::info_string(const std::string& label, const int& tab_level, const bool& dump_all) const {
  std::ostringstream ss;
  ss << "PolyMesh2d " << label << " info:\n";
  ss << vertices.info_string(label, tab_level+1, dump_all);
  ss << edges.info_string(label, tab_level+1, dump_all);
  ss << faces.info_string(label, tab_level+1, dump_all);
  return ss.str();
}

/// ETI
template class PolyMesh2d<TriHexSeed>;
template class PolyMesh2d<QuadRectSeed>;
template class PolyMesh2d<IcosTriSphereSeed>;
template class PolyMesh2d<CubedSphereSeed>;
}
