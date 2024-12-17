#ifndef LPM_COLLOCATED_SWE_IMPL_HPP
#define LPM_COLLOCATED_SWE_IMPL_HPP

#include "lpm_assert.hpp"
#include "lpm_collocated_swe.hpp"
#include "lpm_coords_impl.hpp"
#include "lpm_field_impl.hpp"
#include "lpm_kokkos_defs.hpp"
#include "mesh/lpm_polymesh2d_impl.hpp"
#include "lpm_regularized_kernels_2d.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"

namespace Lpm {
namespace colloc {

namespace impl {
  template <typename Geom, typename ConstViewType>
  struct IntegralFunctor {
    ConstViewType values;
    const_scalar_view area;
    const_mask_view mask;
    Index n;
    bool abs_val;

    IntegralFunctor(const ConstViewType values, const const_scalar_view area,
      const const_mask_view mask, const Index n, const bool abs_val = false) :
      values(values),
      area(area),
      mask(mask),
      n(n),
      abs_val(abs_val) {}

    KOKKOS_INLINE_FUNCTION
    void operator() (const Index i, Real& s) const {
      if (!mask(i)) {
       s += (abs_val ? abs(values(i))*area(i) : values(i)*area(i));
      }
    }
  };

  template <typename Geom, typename ConstViewType>
  struct SquaredIntegralFunctor {
    ConstViewType values;
    const_scalar_view area;
    const_mask_view mask;
    Index n;

    SquaredIntegralFunctor(const ConstViewType values, const const_scalar_view area,
      const const_mask_view mask, const Index n) :
      values(values),
      area(area),
      mask(mask),
      n(n) {}

    KOKKOS_INLINE_FUNCTION
    void operator() (const Index i, Real& s) const {
      if constexpr (std::is_same<ConstViewType, scalar_view_type>::value or
                    std::is_same<ConstViewType, const_scalar_view>::value) {
        if (!mask(i)) {
         s += square(values(i))*area(i);
        }
      }
      else {
        const auto vec_i = Kokkos::subview(values, i, Kokkos::ALL);
        if (!mask(i)) {
          s += Geom::norm2(vec_i)*area(i);
        }
      }
    }
  };
} // namespace colloc::impl

template <typename SeedType>
CollocatedSWE<SeedType>::CollocatedSWE(const PolyMeshParameters<SeedType>& mesh_params, const Coriolis& coriolis) :
  rel_vort("relative_vorticity", mesh_params.nmaxfaces),
  pot_vort("potential_vorticity", mesh_params.nmaxfaces),
  divergence("divergence", mesh_params.nmaxfaces),
  surface("surface_height", mesh_params.nmaxfaces),
  bottom("bottom_height", mesh_params.nmaxfaces),
  surface_lap("surface_laplacian", mesh_params.nmaxfaces),
  depth("depth", mesh_params.nmaxfaces),
  double_dot("double_dot", mesh_params.nmaxfaces),
  ftle("ftle", mesh_params.nmaxfaces),
  du1dx1("du1dx1", mesh_params.nmaxfaces), // TODO: remove these after double dot is verified
  du1dx2("du1dx2", mesh_params.nmaxfaces), // TODO: remove these after double dot is verified
  du2dx1("du2dx1", mesh_params.nmaxfaces), // TODO: remove these after double dot is verified
  du2dx2("du2dx2", mesh_params.nmaxfaces), // TODO: remove these after double dot is verified
  velocity_active("velocity", mesh_params.nmaxfaces),
  velocity_passive("velocity", mesh_params.nmaxverts),
  mass("mass", mesh_params.nmaxfaces),
  mesh(mesh_params),
  ref_crds_active(mesh_params.nmaxfaces),
  ref_crds_passive(mesh_params.nmaxverts),
  g(1),
  t(0),
  eps(0)
{
  Kokkos::deep_copy(ref_crds_passive.view, mesh.vertices.lag_crds.view);
  Kokkos::deep_copy(ref_crds_active.view, mesh.faces.lag_crds.view);
}

template <typename SeedType>
void CollocatedSWE<SeedType>::set_kernel_width_from_power(const Real power) {
  LPM_ASSERT( (power > 0 and power < 1) );
  const Real dx = mesh.appx_mesh_size();
  if (dx < 1) {
    eps = pow(dx, power);
  }
  else {
    eps = pow(dx, 1/power);
  }
}

template <typename SeedType>
void CollocatedSWE<SeedType>::update_host() {
  rel_vort.update_host();
  divergence.update_host();
  pot_vort.update_host();
  bottom.update_host();
  surface.update_host();
  surface_lap.update_host();
  depth.update_host();
  mass.update_host();
  velocity_active.update_host();
  velocity_passive.update_host();
  double_dot.update_host();
  du1dx1.update_host();
  du1dx2.update_host();
  du2dx1.update_host();
  du2dx2.update_host();
  ftle.update_host();
  mesh.update_host();
}

template <typename SeedType>
void CollocatedSWE<SeedType>::update_device() {
  rel_vort.update_device();
  divergence.update_device();
  pot_vort.update_device();
  bottom.update_device();
  surface.update_device();
  surface_lap.update_device();
  depth.update_device();
  mass.update_device();
  velocity_active.update_device();
  velocity_passive.update_device();
  double_dot.update_device();
  du1dx1.update_device();
  du1dx2.update_device();
  du2dx1.update_device();
  du2dx2.update_device();
  ftle.update_device();
  mesh.update_device();
}

template <typename SeedType>
void CollocatedSWE<SeedType>::set_kernel_width_from_multiplier(const Real multiplier) {
  LPM_ASSERT(multiplier > 1);
  const Real dx = mesh.appx_mesh_size();
  eps = dx * multiplier;
}

template <typename SeedType>
Real CollocatedSWE<SeedType>::total_mass() const {
  Real sum;
  Kokkos::parallel_reduce(mesh.n_faces_host(),
    impl::IntegralFunctor<geometry_t, const_scalar_view>(
      mass.view, mesh.faces.area, mesh.faces.mask, mesh.n_faces_host()),
    sum);
  return sum;
}

template <typename SeedType>
Real CollocatedSWE<SeedType>::total_vorticity() const {
  Real sum;
  Kokkos::parallel_reduce(mesh.n_faces_host(),
    impl::IntegralFunctor<geometry_t, const_scalar_view>(rel_vort.view, mesh.faces.area, mesh.faces.mask, mesh.n_faces_host()),
    sum);
  return sum;
}

template <typename SeedType>
Real CollocatedSWE<SeedType>::total_divergence() const {
  Real sum;
  Kokkos::parallel_reduce(mesh.n_faces_host(),
    impl::IntegralFunctor<geometry_t, const_scalar_view>(divergence.view, mesh.faces.area, mesh.faces.mask, mesh.n_faces_host()),
    sum);
  return sum;
}

template <typename SeedType>
Real CollocatedSWE<SeedType>::total_enstrophy() const {
  Real sum;
  Kokkos::parallel_reduce(mesh.n_faces_host(),
    impl::SquaredIntegralFunctor<geometry_t, const_mask_view>(rel_vort.view, mesh.faces.area, mesh.faces.mask, mesh.n_faces_host()),
    sum);
  return 0.5*sum;
}

template <typename SeedType>
Real CollocatedSWE<SeedType>::total_kinetic_energy() const {
  Real sum;
  const auto h = depth.view;
  const auto u = velocity_active.view;
  const auto m = mesh.faces.mask;
  const auto a = mesh.faces.area;
  Kokkos::parallel_reduce(mesh.n_faces_host(),
    KOKKOS_LAMBDA (const Index i, Real& s) {
      if (!m(i)) {
        const auto u_i = Kokkos::subview(u, i, Kokkos::ALL);
        s += h(i) * geometry_t::norm2(u_i)*a(i);
      }
    },
    sum);
  return 0.5*sum;
}

template <typename SeedType>
Real CollocatedSWE<SeedType>::total_potential_energy() const {
  Real sum;
  const auto h = depth.view;
  const auto m = mesh.faces.mask;
  const auto a = mesh.faces.area;
  const auto local_g = g;
  Kokkos::parallel_reduce(mesh.n_faces_host(),
    KOKKOS_LAMBDA (const Index i, Real& s) {
      if (!m(i)) {
        s += local_g * square(h(i)) * a(i);
      }
    },
    sum);
  return 0.5*sum;
}

template <typename SeedType>
Real CollocatedSWE<SeedType>::total_energy() const {
  return total_potential_energy() + total_kinetic_energy();
}

template <typename SeedType> template <typename TopographyType, typename SurfaceType>
void CollocatedSWE<SeedType>::init_surface_and_depth(const TopographyType& topo, const SurfaceType& sfc) {
  const auto crds = mesh.faces.lag_crds.view;
  auto topo_view = bottom.view;
  auto sfc_view = surface.view;
  auto depth_view = depth.view;
  auto mass_view = mass.view;
  const auto area = mesh.faces.area;
  Kokkos::parallel_for(mesh.n_faces_host(),
    KOKKOS_LAMBDA (const Index i) {
      const auto x_i = Kokkos::subview(crds, i, Kokkos::ALL);
      const Real b = topo(x_i);
      const Real s = sfc(x_i);
      topo_view(i) = b;
      sfc_view(i) = s;
      depth_view(i) = s - b;
      mass_view(i) = (s - b) * area(i);
    });
}

template <typename SeedType> template <typename VorticityType, typename DivergenceType>
void CollocatedSWE<SeedType>::init_vorticity_and_divergence(const VorticityType& zeta, const DivergenceType& delta) {
  const auto crds = mesh.faces.lag_crds.view;
  const auto depth_view = depth.view;
  auto zeta_view = rel_vort.view;
  auto delta_view = divergence.view;
  auto q_view = pot_vort.view;
  auto cor = coriolis;
  Kokkos::parallel_for(mesh.n_faces_host(),
    KOKKOS_LAMBDA (const Index i) {
      const auto x_i = Kokkos::subview(crds, i, Kokkos::ALL);
      zeta_view(i) = zeta(x_i);
      delta_view(i) = delta(x_i);
      q_view(i) = (zeta_view(i) + cor.f(x_i)) / depth_view(i) ;
    });
}

template <typename SeedType> template <typename KernelType>
void CollocatedSWE<SeedType>::compute_surface_laplacian_pse(const KernelType& kernels) {
  using reducer_type = PlaneLaplacianReducer<KernelType>;

  Kokkos::TeamPolicy<> policy(mesh.n_faces_host(), Kokkos::AUTO());
  Kokkos::parallel_for(policy,
    DirectSum<reducer_type>(surface_lap.view,
      mesh.faces.phys_crds.view,
      surface.view,
      mesh.faces.phys_crds.view,
      surface.view,
      kernels,
      mesh.faces.area,
      mesh.faces.mask,
      mesh.n_faces_host())
  );
}

template <typename SeedType>
void CollocatedSWE<SeedType>::allocate_scalar_tracer(const std::string& name) {
  tracers.emplace(name, ScalarField<FaceField>(name, rel_vort.view.extent(0)));
}

template <typename SeedType>
template <typename TopographyType, typename SurfaceType, typename VorticityType, typename DivergenceType, typename KernelType>
void CollocatedSWE<SeedType>::init_swe_problem(const TopographyType& topo, const SurfaceType& sfc, const VorticityType& zeta, const DivergenceType& delta, const KernelType& kernels) {
  init_surface_and_depth(topo, sfc);
  init_vorticity_and_divergence(zeta, delta);
  compute_surface_laplacian_pse(kernels);
  compute_velocity_direct_sum(kernels);
}

template <typename SeedType> template <typename KernelType>
void CollocatedSWE<SeedType>::compute_velocity_direct_sum(const KernelType& kernels) {
  Kokkos::TeamPolicy<> active_policy(mesh.n_faces_host(), Kokkos::AUTO());
  Kokkos::parallel_for(active_policy,
    VelocityDirectSum(velocity_active.view,
      double_dot.view,
      du1dx1.view, // TODO: remove once double dot is verified
      du1dx2.view,
      du2dx1.view,
      du2dx2.view,
      mesh.faces.phys_crds.view,
      mesh.faces.phys_crds.view,
      kernels,
      rel_vort.view,
      divergence.view,
      mesh.faces.area,
      mesh.faces.mask,
      mesh.n_faces_host())
  );

  scalar_view_type dummy_view("dummy", mesh.n_vertices_host());
  Kokkos::TeamPolicy<> passive_policy(mesh.n_vertices_host(), Kokkos::AUTO());
  Kokkos::parallel_for(passive_policy,
    VelocityDirectSum(velocity_passive.view,
      dummy_view,
      dummy_view,
      dummy_view,
      dummy_view,
      dummy_view,
      mesh.vertices.phys_crds.view,
      mesh.faces.phys_crds.view,
      kernels,
      rel_vort.view,
      divergence.view,
      mesh.faces.area,
      mesh.faces.mask,
      mesh.n_faces_host())
  );

}

template <typename SeedType>
VtkPolymeshInterface<SeedType> vtk_mesh_interface(const CollocatedSWE<SeedType>& swe) {
  VtkPolymeshInterface<SeedType> vtk(swe.mesh);
  vtk.add_scalar_cell_data(swe.rel_vort.view);
  vtk.add_scalar_cell_data(swe.divergence.view);
  vtk.add_scalar_cell_data(swe.pot_vort.view);
  vtk.add_scalar_cell_data(swe.depth.view);
  vtk.add_scalar_cell_data(swe.surface.view);
  vtk.add_scalar_cell_data(swe.surface_lap.view);
  vtk.add_scalar_cell_data(swe.double_dot.view);
  vtk.add_scalar_cell_data(swe.du1dx1.view); // TODO: remove once double dot is verified
  vtk.add_scalar_cell_data(swe.du1dx2.view); // TODO: remove once double dot is verified
  vtk.add_scalar_cell_data(swe.du2dx1.view); // TODO: remove once double dot is verified
  vtk.add_scalar_cell_data(swe.du2dx2.view); // TODO: remove once double dot is verified
  vtk.add_scalar_cell_data(swe.bottom.view);
  vtk.add_vector_cell_data(swe.velocity_active.view);
  vtk.add_vector_point_data(swe.velocity_passive.view);
  vtk.add_scalar_cell_data(swe.mass.view);
  for (const auto& t : swe.tracers) {
    vtk.add_scalar_cell_data(t.second.view);
  }
  return vtk;
}

} // namespace colloc
} // namespace Lpm
#endif
