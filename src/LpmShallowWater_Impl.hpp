#ifndef LPM_SWE_IMPL_HPP
#define LPM_SWE_IMPL_HPP

#include <sstream>

#include "LpmShallowWater.hpp"
#include "LpmUtilities.hpp"

namespace Lpm {

template <typename SeedType>
std::string ShallowWater<SeedType>::infoString(const std::string& label,
                                               const int& tab_level,
                                               const bool& dump_all) const {
  std::ostringstream ss;
  ss << PolyMesh2d<SeedType>::infoString(label, tab_level, dump_all);
  ko::MinMaxScalar<Real> mm;
  const auto vzeta = relVortVerts;
  ko::parallel_reduce(
      this->nvertsHost(),
      KOKKOS_LAMBDA(const Index& i, ko::MinMaxScalar<Real>& rr) {
        if (vzeta(i) < rr.min_val) rr.min_val = vzeta(i);
        if (vzeta(i) > rr.max_val) rr.max_val = vzeta(i);
      },
      ko::MinMax<Real>(mm));
  ss << indentString(tab_level) << "relvortVerts (min,max) = (" << mm.min_val
     << ", " << mm.max_val << ")\n";
  return ss.str();
}

template <typename SeedType>
ShallowWater<SeedType>::ShallowWater(const Index nmaxverts,
                                     const Index nmaxedges,
                                     const Index nmaxfaces,
                                     const Int& nq_scaler, const Int& nq_vector)
    : PolyMesh2d<SeedType>(nmaxverts, nmaxedges, nmaxfaces),

      relVortVerts("relative_vorticity_vertices", nmaxverts),
      potVortVerts("potential_vorticity_vertices", nmaxverts),
      divVerts("divergence_vertices", nmaxverts),
      surfaceHeightVerts("fluid_sfc_hght_vertices", nmaxverts),
      depthVerts("fluid_depth_vertices", nmaxverts),
      topoVerts("bottom_topography_vertices", nmaxverts),
      velocityVerts("velocity_vertices", nmaxverts),

      relVortFaces("relative_vorticity_faces", nmaxfaces),
      potVortFaces("potential_vorticity_faces", nmaxfaces),
      divFaces("divergence_faces", nmaxfaces),
      surfaceHeightFaces("fluid_sfc_hgt_faces", nmaxfaces),
      depthFaces("fluid_depth_faces", nmaxfaces),
      topoFaces("bottom_topography_faces", nmaxfaces),
      massFaces("mass_faces", nmaxfaces),
      velocityFaces("velocity_faces", nmaxfaces),

      scalar_tracer_verts(nq_scaler),
      scalar_tracer_faces(nq_scaler),
      vector_tracer_verts(nq_vector),
      vector_tracer_faces(nq_vector),

      f0(0),
      beta(0),
      Omega(0) {
  host_relVortVerts = ko::create_mirror_view(relVortVerts);
  host_potVortVerts = ko::create_mirror_view(potVortVerts);
  host_divVerts = ko::create_mirror_view(divVerts);
  host_sfcVerts = ko::create_mirror_view(surfaceHeightVerts);
  host_depthVerts = ko::create_mirror_view(depthVerts);
  host_topoVerts = ko::create_mirror_view(topoVerts);
  host_velocityVerts = ko::create_mirror_view(velocityVerts);

  host_relVortFaces = ko::create_mirror_view(relVortFaces);
  host_potVortFaces = ko::create_mirror_view(potVortFaces);
  host_divFaces = ko::create_mirror_view(divFaces);
  host_sfcFaces = ko::create_mirror_view(surfaceHeightFaces);
  host_depthFaces = ko::create_mirror_view(depthFaces);
  host_topoFaces = ko::create_mirror_view(topoFaces);
  host_massFaces = ko::create_mirror_view(massFaces);
  host_velocityFaces = ko::create_mirror_view(velocityFaces);

  std::ostringstream ss;
  for (Int i = 0; i < nq_scaler; ++i) {
    ss << "tracer" << i;
    scalar_tracer_verts[i] = scalar_view_type(ss.str(), nmaxverts);
    host_scalar_tracer_verts.push_back(
        ko::create_mirror_view(scalar_tracer_verts[i]));

    scalar_tracer_faces[i] = scalar_view_type(ss.str(), nmaxfaces);
    host_scalar_tracer_faces.push_back(
        ko::create_mirror_view(scalar_tracer_faces[i]));
    ss.str("");
  }

  for (Int i = 0; i < nq_vector; ++i) {
    ss << "tracer" << i;
    vector_tracer_verts[i] = vector_field(ss.str(), nmaxverts);
    host_vector_tracer_verts.push_back(
        ko::create_mirror_view(vector_tracer_verts[i]));

    vector_tracer_faces[i] = vector_field(ss.str(), nmaxfaces);
    host_vector_tracer_faces.push_back(
        ko::create_mirror_view(vector_tracer_faces[i]));
    ss.str("");
  }
}

template <typename SeedType>
template <typename ProblemType>
void ShallowWater<SeedType>::set_bottom_topography() {
  auto tvcopy = this->topoVerts;
  auto tfcopy = this->topoFaces;
  auto vccopy = this->physVerts.crds;
  auto fccopy = this->physFaces.crds;
  ko::parallel_for(
      this->nvertsHost(), KOKKOS_LAMBDA(const Index& i) {
        const auto mycrd = ko::subview(vccopy, i, ko::ALL);
        tvcopy(i) = ProblemType::bottom_height(mycrd);
      });

  ko::parallel_for(
      this->nfacesHost(), KOKKOS_LAMBDA(const Index& i) {
        const auto mycrd = ko::subview(fccopy, i, ko::ALL);
        tfcopy(i) = ProblemType::bottom_height(mycrd);
      });

  //   ko::deep_copy(host_topoVerts, topoVerts);
  //   ko::deep_copy(host_topoFaces, topoFaces);
}

template <typename SeedType>
template <typename ProblemType>
void ShallowWater<SeedType>::init_problem() {
  this->f0 = ProblemType::f0;
  this->beta = ProblemType::beta;
  this->Omega = ProblemType::OMEGA;

  auto zetav = this->relVortVerts;
  auto Qv = this->potVortVerts;
  auto sigmav = this->divVerts;
  auto sv = this->surfaceHeightVerts;
  auto sbv = this->topoVerts;
  auto vv = this->velocityVerts;
  auto hv = this->depthVerts;
  const auto vcrds = this->physVerts.crds;
  ko::parallel_for(
      this->nvertsHost(), KOKKOS_LAMBDA(const Index& i) {
        const auto mcrd = ko::subview(vcrds, i, ko::ALL);
        zetav(i) = ProblemType::zeta0(mcrd);
        Qv(i) =
            zetav(i) + (SeedType::geo::ndim == 3
                            ? 2 * ProblemType::OMEGA * mcrd(2)
                            : ProblemType::f0 + ProblemType::beta * mcrd(1));

        sigmav(i) = ProblemType::sigma0(mcrd);
        sv(i) = ProblemType::sfc0(mcrd);
        sbv(i) = ProblemType::bottom_height(mcrd);
        auto uv = ko::subview(vv, i, ko::ALL);
        ProblemType::u0(uv, mcrd);
        const Real h = sv(i) - sbv(i);
        hv(i) = h;
        if (h < 0) {
          sv(i) = sbv(i);
          zetav(i) = 0;
          Qv(i) = 0;
          sigmav(i) = 0;
          for (Short j = 0; j < vv.extent(1); ++j) {
            uv(j) = 0;
          }
          hv(i) = 0;
        }
      });

  auto zetaf = this->relVortFaces;
  auto Qf = this->potVortFaces;
  auto sigmaf = this->divFaces;
  auto sf = this->surfaceHeightFaces;
  auto sbf = this->topoFaces;
  auto vf = this->velocityFaces;
  auto mass = this->massFaces;
  auto hf = this->depthFaces;
  const auto fcrds = this->physFaces.crds;
  const auto fmask = this->faces.mask;
  const auto farea = this->faces.area;
  ko::parallel_for(
      this->nfacesHost(), KOKKOS_LAMBDA(const Index& i) {
        if (fmask(i)) {
          zetaf(i) = 0;
          Qf(i) = 0;
          sigmaf(i) = 0;
          sf(i) = 0;
          sbf(i) = 0;
          mass(i) = 0;
          hf(i) = 0;
          for (Short j = 0; j < vf.extent(1); ++j) {
            vf(i, j) = 0;
          }
        } else {
          const auto mcrd = ko::subview(fcrds, i, ko::ALL);
          zetaf(i) = ProblemType::zeta0(mcrd);
          Qf(i) =
              zetaf(i) + (SeedType::geo::ndim == 3
                              ? 2 * ProblemType::OMEGA * mcrd(2)
                              : ProblemType::f0 + ProblemType::beta * mcrd(1));
          sigmaf(i) = ProblemType::sigma0(mcrd);
          sf(i) = ProblemType::sfc0(mcrd);
          sbf(i) = ProblemType::bottom_height(mcrd);
          const Real h = sf(i) - sbf(i);
          hf(i) = h;
          mass(i) = (h > 0 ? h * farea(i) : 0);
          if (h < 0) {
            zetaf(i) = 0;
            Qf(i) = 0;
            sigmaf(i) = 0;
            sf(i) = sbf(i);
            hf(i) = 0;
          }
          auto uv = ko::subview(vf, i, ko::ALL);
          ProblemType::u0(uv, mcrd);
          if (h < 0) {
            for (Short j = 0; j < vf.extent(1); ++j) {
              uv(j) = 0;
            }
          }
        }
      });

  ko::deep_copy(host_relVortVerts, relVortVerts);
  ko::deep_copy(host_potVortVerts, potVortVerts);
  ko::deep_copy(host_divVerts, divVerts);
  ko::deep_copy(host_sfcVerts, surfaceHeightVerts);
  ko::deep_copy(host_depthVerts, depthVerts);
  ko::deep_copy(host_topoVerts, topoVerts);
  ko::deep_copy(host_velocityVerts, velocityVerts);

  ko::deep_copy(host_relVortFaces, relVortFaces);
  ko::deep_copy(host_potVortFaces, potVortFaces);
  ko::deep_copy(host_divFaces, divFaces);
  ko::deep_copy(host_sfcFaces, surfaceHeightFaces);
  ko::deep_copy(host_depthFaces, depthFaces);
  ko::deep_copy(host_topoFaces, topoFaces);
  ko::deep_copy(host_velocityFaces, velocityFaces);
  ko::deep_copy(host_massFaces, massFaces);
}

template <typename SeedType>
void ShallowWater<SeedType>::addFieldsToVtk(
    Polymesh2dVtkInterface<SeedType>& vtk) const {
  vtk.addScalarPointData(relVortVerts, "relative_vorticity");
  vtk.addScalarPointData(potVortVerts, "potential_vorticity");
  vtk.addScalarPointData(divVerts, "divergence");
  vtk.addScalarPointData(surfaceHeightVerts, "surface_height");
  vtk.addScalarPointData(topoVerts, "bottom_height");
  vtk.addVectorPointData(velocityVerts, "velocity");
  vtk.addScalarPointData(depthVerts, "depth");

  vtk.addScalarCellData(relVortFaces, "relative_vorticity");
  vtk.addScalarCellData(potVortFaces, "potential_vorticity");
  vtk.addScalarCellData(divFaces, "divergence");
  vtk.addScalarCellData(surfaceHeightFaces, "surface_height");
  vtk.addScalarCellData(massFaces, "mass");
  vtk.addScalarCellData(topoFaces, "bottom_height");
  vtk.addVectorCellData(velocityFaces, "velocity");
  vtk.addScalarCellData(depthFaces, "depth");

  //   for (Int k=0; k<tracer_verts.size(); ++k) {
  //     vtk.addScalarPointData(tracer_verts[k]);
  //     vtk.addScalarCellData(tracer_faces[k]);
  //   }
}

template <typename SeedType>
Real ShallowWater<SeedType>::total_mass() const {
  Real m;
  const auto local_mass = massFaces;
  ko::parallel_reduce(
      this->nfacesHost(),
      KOKKOS_LAMBDA(const Index& i, Real& sum) { sum += local_mass(i); }, m);
  return m;
}

template <typename SeedType>
Real ShallowWater<SeedType>::total_mass_integral() const {
  const auto local_h = depthFaces;
  const auto local_a = this->faces.area;
  const auto local_mask = this->faces.mask;
  Real m;
  ko::parallel_reduce(
      this->nfacesHost(),
      KOKKOS_LAMBDA(const Index& i, Real& sum) {
        if (!local_mask(i)) {
          sum += local_h(i) * local_a(i);
        }
      },
      m);
  return m;
}

}  // namespace Lpm
#endif
