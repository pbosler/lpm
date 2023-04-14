#ifndef LPM_F_INTERP_IMPL_HPP
#define LPM_F_INTERP_IMPL_HPP

#include "fortran/lpm_ssrfpack_interface.hpp"
#include "lpm_fortran_c.h"
#include "util/lpm_string_util.hpp"

#ifndef NDEBUG
#include <iostream>
#include <sstream>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/// stripack.f subroutine trmesh
/**

SUBROUTINE TRMESH (N,X,Y,Z, LIST,LPTR,LEND,LNEW,NEAR,NEXT,DIST,IER)
*/
extern void trmesh(int*, double*, double*, double*, int*, int*, int*, int*,
                   int*, int*, double*, int*);

/// ssrfpack.f subroutine gradl
/**
 */
extern void gradl(int*, int*, double*, double*, double*, double*, int*, int*,
                  int*, double*, int*);

/// ssrfpack.f subroutine getsig
/**

*/
extern void getsig(int*, double*, double*, double*, double*, int*, int*, int*,
                   double*, double*, double*, double*, int*);

/// ssrfpack.f subroutine intrc1
/**
 */
extern void intrc1(int*, double*, double*, double*, double*, double*, double*,
                   int*, int*, int*, int*, double*, int*, double*, int*,
                   double*, int*);

#ifdef __cplusplus
}
#endif

namespace Lpm {

Real left_val(double x1, double y1, double z1,
          double x2, double y2, double z2,
          double x0, double y0, double z0) {
          return x0*(y1*z2-y2*z1) - y0*(x1*z2-x2*z1) + z0*(x1*y2-x2*y1);
          }

/**  @brief c-interface to stripack subroutine trmesh

  generates a Delaunay triangulation on the surface of the sphere.

  @params [in] n number of source points
  @params [in] x array of points' x-coordinates
  @params [in] y array of points' y-coordinates
  @params [in] z array of points' z-coordinates
  @params [in/out] list triangulation data structure (with lptr and lend)
  @params [in/out] lptr triangulation data structure (with list and lend)
  @params [in/out] lend triangulation data structure (with list and lptr)
  @params [in/out] lnew index of first empty spot in triangulation
  @params [in/out] near work array
  @params [in/out] next work array
  @params [in/out] dist work array
  @return ier error code
*/
int c_trmesh(int n, double* x, double* y, double* z, int* list, int* lptr,
             int* lend, int lnew, int* near, int* next, double* dist) {
  int ier;
  trmesh(&n, x, y, z, list, lptr, lend, &lnew, near, next, dist, &ier);
#ifndef NDEBUG
  if (ier != 0) {
    std::ostringstream ss;
    ss << "stripack trmesh error: ";
    if (ier == -1) {
      ss << " at least 3 nodes are required; received n < 3.\n";
    } else if (ier == -2) {
      ss << " the first 3 nodes are colinear.\n";
      for (int i=0; i<3; ++i) {
        ss << "\t(x y z) = (" << x[i] << " " << y[i] << " " << z[i] << ")\n";
      }
      const Real l1 = left_val(x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2]);
      const Real l2 = left_val(x[1], y[1], z[1], x[0], y[0], z[0], x[2], y[2], z[2]);
      ss << "l1 = " << l1 << " l2 = " << l2 << "\n";
    } else if (ier > 0) {
      ss << " duplicate of node " << ier << " detected.\n";
    }
    std::cout << ss.str();
  }
#endif
  return ier;
}

/** @brief c-interface to ssrfpack subroutine gradl.

  estimates the gradient vector at a location for a function on the sphere
  using the following steps:
    1. Rotate the point (node k) so that it is at the north pole.
    2. Project nearby points into the x-y plane centered at the pole.
    3. Fit a bivariate quadratic polynomial to the projected data
    4. Differentiate the bivariate polynomial to get its 2d gradient
    5. Rotate the gradient back to node k, where it will be tangent to the
          sphere, and orthogonal to the kth node's position vector.

  @param [in] n number of points on the sphere
  @param [in] k index of point whose gradient needs estimation
  @params [in] x array of points' x-coordinates
  @params [in] y array of points' y-coordinates
  @params [in] z array of points' z-coordinates
  @params [in/out] list triangulation data structure (with lptr and lend)
  @params [in/out] lptr triangulation data structure (with list and lend)
  @params [in/out] lend triangulation data structure (with list and lptr)
  @params [out] gk gradient vector
  @return ier error code
*/
int c_gradl(int n, int k, double* x, double* y, double* z, double* w, int* list,
            int* lptr, int* lend, double* gk) {
  int ier;
  int kp1 = k + 1; /* convert to fortran base-1 idx */
  gradl(&n, &kp1, x, y, z, w, list, lptr, lend, gk, &ier);
#ifndef NDEBUG
  if (ier < 6) {
    std::ostringstream ss;
    ss << "ssrfpack gradl error: ";
    if (ier == -1) {
      ss << " index out of range. n = " << n << " k = " << k << "\n";
    } else if (ier == -2) {
      ss << " least squares system has no solution at node " << k << "\n";
    } else {
      ss << " unspecified error.\n";
    }
    std::cout << ss.str();
  }
#endif
  return ier;
}

int c_getsig(int n, double* x, double* y, double* z, double* h, int* list,
             int* lptr, int* lend, double* grad, double sigma_tol,
             double* sigma, double& dsmax) {
  int ier;
  getsig(&n, x, y, z, h, list, lptr, lend, grad, &sigma_tol, sigma, &dsmax,
         &ier);
#ifndef NDEBUG
  if (ier < 0) {
    std::ostringstream ss;
    ss << "ssrfpack getsig error: ";
    if (ier == -1) {
      ss << " at least 3 nodes are required.\n";
    } else if (ier == -2) {
      ss << " duplicate node found.\n";
    }
    std::cout << ss.str();
  }
#endif
  return ier;
}

double c_intrc1(int n, double plat, double plon, double* x, double* y,
                double* z, double* f, int* list, int* lptr, int* lend,
                int sigma_flag, double* sigma, int grad_flag, double* grad,
                int& tri_idx) {
  double result;
  int ier;
  intrc1(&n, &plat, &plon, x, y, z, f, list, lptr, lend, &sigma_flag, sigma,
         &grad_flag, grad, &tri_idx, &result, &ier);
#ifndef NDEBUG
  if (ier != 0) {
    std::ostringstream ss;
    ss << "ssrfpack intrc1 error: ";
    if (ier == 1) {
      ss << " extrapolation was required.\n";
    } else if (ier == -1) {
      ss << " index out of range; n = " << n << " tri_idx = " << tri_idx
         << "\n";
    } else if (ier == -2) {
      ss << " colinear nodes found.\n";
    } else if (ier == -3) {
      ss << " point at (lon, lat) = (" << plat << ", " << plon
         << ") falls outside the triangulation.\n";
    }
    std::cout << ss.str();
  }
#endif
  LPM_ASSERT(ier == 0);
  return result;
}

template <typename SeedType>
STRIPACKInterface::STRIPACKInterface(const GatherMeshData<SeedType>& input)
    : list("sphere_deltri_list", 6 * input.n() - 12),
      lptr("sphere_deltri_lptr", 6 * input.n() - 12),
      lend("sphere_deltri_lend", input.n()),
      n(input.n()) {
  static_assert(std::is_same<typename SeedType::geo, SphereGeometry>::value,
                "STRIPACKInterface:: sphere geometry required.");
  LPM_REQUIRE_MSG(input.unpacked, "STRIPACKInterface requires unpacked coordinates.");
  build_triangulation(input);
}

template <typename SeedType>
void STRIPACKInterface::build_triangulation(
    const GatherMeshData<SeedType>& input) {
  Kokkos::View<double*, Host> dist("trmesh_dist", n);
  Kokkos::View<int*, Host> near("trmesh_near", n);
  Kokkos::View<int*, Host> next("trmesh_next", n);
  int lnew;
  int err = c_trmesh(n, input.h_x.data(), input.h_y.data(), input.h_z.data(),
                     list.data(), lptr.data(), lend.data(), lnew, near.data(),
                     next.data(), dist.data());
  if (err == -1) {
    LPM_REQUIRE_MSG(false, "STRIPACKInterface error: less than 3 points.");
  } else if (err == -2) {
    LPM_REQUIRE_MSG(false,
                    "STRIPACKInterface error: first 3 points are colinear.");
  } else if (err > 0) {
    std::ostringstream ss;
    ss << "STRIPACKInterface error: node " << err << " is a duplicate.";
    LPM_REQUIRE_MSG(false, ss.str());
  }
}

template <typename SeedType>
SSRFPackInterface<SeedType>::SSRFPackInterface(
    const GatherMeshData<SeedType>& in,
    const std::map<std::string, std::string>& s_in_out,
    const std::map<std::string, std::string>& v_in_out)
    : input(in),
      scalar_in_out_map(s_in_out),
      vector_in_out_map(v_in_out),
      comp1("ssrfpack_comp1", input.n()),
      comp2("ssrfpack_comp2", input.n()),
      comp3("ssrfpack_comp3", input.n()),
      grad1("ssrfpack_grad1", 3, input.n()),
      grad2("ssrfpack_grad2", 3, input.n()),
      grad3("ssrfpack_grad3", 3, input.n()),
      sigma1("ssrfpack_sigma1", 6 * input.n() - 12),
      sigma2("ssrfpack_sigma2", 6 * input.n() - 12),
      sigma3("ssrfpack_sigma3", 6 * input.n() - 12),
      del_tri(input),
      sigma_flag(0),
      grad_flag(1),
      sigma_tol(0.01) {
      LPM_REQUIRE_MSG(in.unpacked, "SSRFPackInterface requires unpacked coordinates.");
    }

template <typename SeedType>
template <typename PolyMeshType>
void SSRFPackInterface<SeedType>::interpolate(
    PolyMeshType& pm,
    std::map<std::string, ScalarField<VertexField>> vert_scalar_fields,
    std::map<std::string, ScalarField<FaceField>> face_scalar_fields,
    std::map<std::string, VectorField<SphereGeometry, VertexField>>
        vert_vector_fields,
    std::map<std::string, VectorField<SphereGeometry, FaceField>>
        face_vector_fields) {

  const auto h_vert_crds = pm.vertices.phys_crds.get_host_crd_view();
  const auto h_face_crds = pm.faces.phys_crds.get_host_crd_view();

  // Loop over scalar fields
  for (const auto& sio : scalar_in_out_map) {
    LPM_REQUIRE_MSG(
        input.h_scalar_fields.count(sio.first) == 1,
        "SSRFPACK interpolate : input scalar not found.");
    LPM_REQUIRE_MSG(
        vert_scalar_fields.count(sio.second) == 1,
        "SSRFPACK interpolate : vert output scalar not found.");
    LPM_REQUIRE_MSG(
        face_scalar_fields.count(sio.second) == 1,
        "SSRFPACK intperolate : face output scalar not found.");

    set_scalar_source_data(sio.first);
    const auto vert_out_view = vert_scalar_fields.at(sio.second).hview;
    const auto face_out_view = face_scalar_fields.at(sio.second).hview;

    int tri_idx = 1;

    for (int i = 0; i < pm.n_vertices_host(); ++i) {
      const auto xyz = Kokkos::subview(h_vert_crds, i, Kokkos::ALL);
      const Real lon = SphereGeometry::longitude(xyz);
      const Real lat = SphereGeometry::latitude(xyz);

      vert_out_view(i) = c_intrc1(
          input.n(), lat, lon, input.h_x.data(), input.h_y.data(),
          input.h_z.data(), input.h_scalar_fields.at(sio.first).data(),
          del_tri.list.data(), del_tri.lptr.data(), del_tri.lend.data(),
          sigma_flag, sigma1.data(), grad_flag, grad1.data(), tri_idx);
    }
    for (int i = 0; i < pm.n_faces_host(); ++i) {
      const auto xyz = Kokkos::subview(h_face_crds, i, Kokkos::ALL);
      const Real lon = SphereGeometry::longitude(xyz);
      const Real lat = SphereGeometry::latitude(xyz);

      face_out_view(i) = c_intrc1(
          input.n(), lat, lon, input.h_x.data(), input.h_y.data(),
          input.h_z.data(), input.h_scalar_fields.at(sio.first).data(),
          del_tri.list.data(), del_tri.lptr.data(), del_tri.lend.data(),
          sigma_flag, sigma1.data(), grad_flag, grad1.data(), tri_idx);
    }

    vert_scalar_fields.at(sio.second).update_device();
    face_scalar_fields.at(sio.second).update_device();
  } // scalar field loop

  // loop over vector fields
  for (const auto& vio : vector_in_out_map) {
    LPM_REQUIRE_MSG(
        input.h_vector_fields.count(vio.first) == 1,
        "SSRFPACK interpolate : input vector not found.");
    LPM_REQUIRE_MSG(
        vert_vector_fields.count(vio.second) == 1,
        "SSRFPACK interpolate : vert output vector not found.");
    LPM_REQUIRE_MSG(
        face_vector_fields.count(vio.second) == 1,
        "SSRFPACK intperolate : face output vector not found.");

    set_vector_source_data(vio.first);

    const auto vert_out_view = vert_vector_fields.at(vio.second).hview;
    const auto face_out_view = face_vector_fields.at(vio.second).hview;
    int tri_idx = 1;
    for (int i = 0; i < pm.n_vertices_host(); ++i) {
      const auto xyz = Kokkos::subview(h_vert_crds, i, Kokkos::ALL);
      const Real lon = SphereGeometry::longitude(xyz);
      const Real lat = SphereGeometry::latitude(xyz);

      vert_out_view(i, 0) =
          c_intrc1(input.n(), lat, lon, input.h_x.data(), input.h_y.data(),
                   input.h_z.data(), comp1.data(), del_tri.list.data(),
                   del_tri.lptr.data(), del_tri.lend.data(), sigma_flag,
                   sigma1.data(), grad_flag, grad1.data(), tri_idx);
      vert_out_view(i, 1) =
          c_intrc1(input.n(), lat, lon, input.h_x.data(), input.h_y.data(),
                   input.h_z.data(), comp2.data(), del_tri.list.data(),
                   del_tri.lptr.data(), del_tri.lend.data(), sigma_flag,
                   sigma2.data(), grad_flag, grad2.data(), tri_idx);
      vert_out_view(i, 2) =
          c_intrc1(input.n(), lat, lon, input.h_x.data(), input.h_y.data(),
                   input.h_z.data(), comp3.data(), del_tri.list.data(),
                   del_tri.lptr.data(), del_tri.lend.data(), sigma_flag,
                   sigma3.data(), grad_flag, grad3.data(), tri_idx);
    }
    for (int i = 0; i < pm.n_faces_host(); ++i) {
      const auto xyz = Kokkos::subview(h_face_crds, i, Kokkos::ALL);
      const Real lon = SphereGeometry::longitude(xyz);
      const Real lat = SphereGeometry::latitude(xyz);

      face_out_view(i, 0) =
          c_intrc1(input.n(), lat, lon, input.h_x.data(), input.h_y.data(),
                   input.h_z.data(), comp1.data(), del_tri.list.data(),
                   del_tri.lptr.data(), del_tri.lend.data(), sigma_flag,
                   sigma1.data(), grad_flag, grad1.data(), tri_idx);
      face_out_view(i, 1) =
          c_intrc1(input.n(), lat, lon, input.h_x.data(), input.h_y.data(),
                   input.h_z.data(), comp2.data(), del_tri.list.data(),
                   del_tri.lptr.data(), del_tri.lend.data(), sigma_flag,
                   sigma2.data(), grad_flag, grad2.data(), tri_idx);
      face_out_view(i, 2) =
          c_intrc1(input.n(), lat, lon, input.h_x.data(), input.h_y.data(),
                   input.h_z.data(), comp3.data(), del_tri.list.data(),
                   del_tri.lptr.data(), del_tri.lend.data(), sigma_flag,
                   sigma3.data(), grad_flag, grad3.data(), tri_idx);
    }

    vert_vector_fields.at(vio.second).update_device();
    face_vector_fields.at(vio.second).update_device();
  } // vector field loop
}

template <typename SeedType> template <typename PolyMeshType>
void SSRFPackInterface<SeedType>::interpolate_lag_crds(PolyMeshType& mesh_out, const Index vert_start_idx, const Index face_start_idx) {

  static_assert(std::is_same<typename PolyMeshType::Geo, SphereGeometry>::value,
    "sphere geometry required");

  set_lag_crd_source_data();

  const auto h_vert_crds = mesh_out.vertices.phys_crds.get_host_crd_view();
  const auto h_face_crds = mesh_out.faces.phys_crds.get_host_crd_view();
  auto h_vert_lag_crds = mesh_out.vertices.lag_crds.get_host_crd_view();
  auto h_face_lag_crds = mesh_out.faces.lag_crds.get_host_crd_view();
  int tri_idx = 1;
  for (int i=vert_start_idx; i<mesh_out.n_vertices_host(); ++i) {
    const auto xyz = Kokkos::subview(h_vert_crds, i, Kokkos::ALL);
    const Real lon = SphereGeometry::longitude(xyz);
    const Real lat = SphereGeometry::latitude(xyz);

    h_vert_lag_crds(i, 0) =
      c_intrc1(input.n(), lat, lon, input.h_x.data(), input.h_y.data(),
        input.h_z.data(), input.h_lag_x.data(), del_tri.list.data(),
        del_tri.lptr.data(), del_tri.lend.data(), sigma_flag,
        sigma1.data(), grad_flag, grad1.data(), tri_idx);

    h_vert_lag_crds(i, 1) =
      c_intrc1(input.n(), lat, lon, input.h_x.data(), input.h_y.data(),
        input.h_z.data(), input.h_lag_y.data(), del_tri.list.data(),
        del_tri.lptr.data(), del_tri.lend.data(), sigma_flag,
        sigma2.data(), grad_flag, grad2.data(), tri_idx);

    h_vert_lag_crds(i, 2) =
      c_intrc1(input.n(), lat, lon, input.h_x.data(), input.h_y.data(),
        input.h_z.data(), input.h_lag_z.data(), del_tri.list.data(),
        del_tri.lptr.data(), del_tri.lend.data(), sigma_flag,
        sigma3.data(), grad_flag, grad3.data(), tri_idx);
  }
  for (int i=face_start_idx; i<mesh_out.n_faces_host(); ++i) {
    const auto xyz = Kokkos::subview(h_face_crds, i, Kokkos::ALL);
    const Real lon = SphereGeometry::longitude(xyz);
    const Real lat = SphereGeometry::latitude(xyz);

    h_face_lag_crds(i,0) = c_intrc1(input.n(), lat, lon, input.h_x.data(), input.h_y.data(),
        input.h_z.data(), input.h_lag_x.data(), del_tri.list.data(),
        del_tri.lptr.data(), del_tri.lend.data(), sigma_flag,
        sigma1.data(), grad_flag, grad1.data(), tri_idx);
    h_face_lag_crds(i,1) = c_intrc1(input.n(), lat, lon, input.h_x.data(), input.h_y.data(),
        input.h_z.data(), input.h_lag_y.data(), del_tri.list.data(),
        del_tri.lptr.data(), del_tri.lend.data(), sigma_flag,
        sigma2.data(), grad_flag, grad2.data(), tri_idx);
    h_face_lag_crds(i,2) = c_intrc1(input.n(), lat, lon, input.h_x.data(), input.h_y.data(),
        input.h_z.data(), input.h_lag_z.data(), del_tri.list.data(),
        del_tri.lptr.data(), del_tri.lend.data(), sigma_flag,
        sigma3.data(), grad_flag, grad3.data(), tri_idx);
  }
}

template <typename SeedType>
void SSRFPackInterface<SeedType>::interpolate(
    const typename Kokkos::View<Real* [3]>::HostMirror output_pts,
    const sfield_map& scalar_fields,
    const vfield_map& vector_fields) {
  for (const auto& sio : scalar_in_out_map) {
    LPM_REQUIRE_MSG(
        input.h_scalar_fields.count(sio.first) == 1,
        "SSRFPACK interpolate : input scalar not found.");
    LPM_REQUIRE_MSG(scalar_fields.count(sio.second) == 1,
                    "SSRFPACK interpolate : output scalar not found.");

    set_scalar_source_data(sio.first);
    const auto out_view = scalar_fields.at(sio.second);

    int tri_idx = 1;
    for (int i = 0; i < output_pts.extent(0); ++i) {
      const auto xyz = Kokkos::subview(output_pts, i, Kokkos::ALL);
      LPM_ASSERT(xyz.extent(0) == 3);
      const Real lon = SphereGeometry::longitude(xyz);
      const Real lat = SphereGeometry::latitude(xyz);

      out_view(i) = c_intrc1(
          input.n(), lat, lon, input.h_x.data(), input.h_y.data(),
          input.h_z.data(), input.h_scalar_fields.at(sio.first).data(),
          del_tri.list.data(), del_tri.lptr.data(), del_tri.lend.data(),
          sigma_flag, sigma1.data(), grad_flag, grad1.data(), tri_idx);
    }
  }
  for (const auto& vio : vector_in_out_map) {
    LPM_REQUIRE_MSG(
        input.h_vector_fields.count(vio.first) == 1,
        "SSRFPACK interpolate: input vector not found.");
    LPM_REQUIRE_MSG(vector_fields.count(vio.second) == 1,
                    "SSRFPACK interpolate: output vector not found.");

    set_vector_source_data(vio.first);
    const auto out_view = vector_fields.at(vio.second);

    int tri_idx = 1;
    for (int i = 0; i < output_pts.extent(0); ++i) {
      const auto xyz = Kokkos::subview(output_pts, i, Kokkos::ALL);
      const Real lon = SphereGeometry::longitude(xyz);
      const Real lat = SphereGeometry::latitude(xyz);

      out_view(i, 0) =
          c_intrc1(input.n(), lat, lon, input.h_x.data(), input.h_y.data(),
                   input.h_z.data(), comp1.data(), del_tri.list.data(),
                   del_tri.lptr.data(), del_tri.lend.data(), sigma_flag,
                   sigma1.data(), grad_flag, grad1.data(), tri_idx);

      out_view(i, 1) =
          c_intrc1(input.n(), lat, lon, input.h_x.data(), input.h_y.data(),
                   input.h_z.data(), comp2.data(), del_tri.list.data(),
                   del_tri.lptr.data(), del_tri.lend.data(), sigma_flag,
                   sigma2.data(), grad_flag, grad2.data(), tri_idx);

      out_view(i, 2) =
          c_intrc1(input.n(), lat, lon, input.h_x.data(), input.h_y.data(),
                   input.h_z.data(), comp3.data(), del_tri.list.data(),
                   del_tri.lptr.data(), del_tri.lend.data(), sigma_flag,
                   sigma3.data(), grad_flag, grad3.data(), tri_idx);
    }
  }
}

template <typename SeedType>
void SSRFPackInterface<SeedType>::set_scalar_source_data(
    const std::string& field_name) {
  LPM_ASSERT_MSG(
      input.h_scalar_fields.find(field_name) != input.h_scalar_fields.end(),
      "SSRFPackInterface::set_scalar_source_data input not found.");

  const auto field_vals = input.h_scalar_fields.at(field_name);

  for (int i = 0; i < input.n(); ++i) {
    const auto grad_vals = Kokkos::subview(grad1, Kokkos::ALL, i);
    LPM_ASSERT(grad_vals.extent(0) == 3);
    Real grad[3];
    int ier =
        c_gradl(input.n(), i, input.h_x.data(), input.h_y.data(),
                input.h_z.data(), field_vals.data(), del_tri.list.data(),
                del_tri.lptr.data(), del_tri.lend.data(), grad);
    for (int j=0; j<3; ++j) {
      grad_vals[j] = grad[j];
    }
#ifndef NDEBUG
    if (ier < 6) {
      std::ostringstream ss;
      ss << "ssrfpack error: gradl ier = " << ier << "\n";
      std::cout << ss.str();
    }
#endif
    LPM_ASSERT(ier >= 6);
  }

  if (sigma_flag > 0) {
    double dsig;
    int ier = c_getsig(input.n(), input.h_x.data(), input.h_y.data(),
                       input.h_z.data(), field_vals.data(), del_tri.list.data(),
                       del_tri.lptr.data(), del_tri.lend.data(), grad1.data(),
                       sigma_tol, sigma1.data(), dsig);
    LPM_ASSERT(ier > 0);
  }
}

template <typename SeedType>
void SSRFPackInterface<SeedType>::set_vector_source_data(
    const std::string& field_name) {
  LPM_ASSERT_MSG(
      input.h_vector_fields.find(field_name) != input.h_vector_fields.end(),
      "SSRFPackInterface::set_vector_source_data input not found.");

  const auto src_vals = input.h_vector_fields.at(field_name);
  for (int i = 0; i < input.n(); ++i) {
    comp1(i) = src_vals(i, 0);
    comp2(i) = src_vals(i, 1);
    comp3(i) = src_vals(i, 2);
  }

  for (int i = 0; i < input.n(); ++i) {
    const auto comp1_grad_vals = Kokkos::subview(grad1, i, Kokkos::ALL);
    const auto comp2_grad_vals = Kokkos::subview(grad2, i, Kokkos::ALL);
    const auto comp3_grad_vals = Kokkos::subview(grad3, i, Kokkos::ALL);
    LPM_ASSERT(comp1_grad_vals.extent(0) == 3);
    int ier = c_gradl(input.n(), i, input.h_x.data(), input.h_y.data(),
                      input.h_z.data(), comp1.data(), del_tri.list.data(),
                      del_tri.lptr.data(), del_tri.lend.data(),
                      comp1_grad_vals.data());
    LPM_ASSERT(ier >= 6);
    ier = c_gradl(input.n(), i, input.h_x.data(), input.h_y.data(),
                  input.h_z.data(), comp2.data(), del_tri.list.data(),
                  del_tri.lptr.data(), del_tri.lend.data(),
                  comp2_grad_vals.data());
    LPM_ASSERT(ier >= 6);
    ier = c_gradl(input.n(), i, input.h_x.data(), input.h_y.data(),
                  input.h_z.data(), comp3.data(), del_tri.list.data(),
                  del_tri.lptr.data(), del_tri.lend.data(),
                  comp3_grad_vals.data());
    LPM_ASSERT(ier >= 6);
  }

  if (sigma_flag > 0) {
    double dsig;
    int ier = c_getsig(input.n(), input.h_x.data(), input.h_y.data(),
                       input.h_z.data(), comp1.data(), del_tri.list.data(),
                       del_tri.lptr.data(), del_tri.lend.data(), grad1.data(),
                       sigma_tol, sigma1.data(), dsig);
    LPM_ASSERT(ier > 0);
    ier = c_getsig(input.n(), input.h_x.data(), input.h_y.data(),
                   input.h_z.data(), comp2.data(), del_tri.list.data(),
                   del_tri.lptr.data(), del_tri.lend.data(), grad2.data(),
                   sigma_tol, sigma2.data(), dsig);
    LPM_ASSERT(ier > 0);
    ier = c_getsig(input.n(), input.h_x.data(), input.h_y.data(),
                   input.h_z.data(), comp3.data(), del_tri.list.data(),
                   del_tri.lptr.data(), del_tri.lend.data(), grad3.data(),
                   sigma_tol, sigma3.data(), dsig);
    LPM_ASSERT(ier > 0);
  }
}

template <typename SeedType>
void SSRFPackInterface<SeedType>::set_lag_crd_source_data() {

  for (int i=0; i<input.n(); ++i) {

      Real lag_x_grad[3];
      Real lag_y_grad[3];
      Real lag_z_grad[3];

      int ier;

      ier = c_gradl(input.n(), i, input.h_x.data(), input.h_y.data(), input.h_z.data(),
        input.h_lag_x.data(), del_tri.list.data(), del_tri.lptr.data(), del_tri.lend.data(),
        lag_x_grad);
      LPM_ASSERT(ier >= 6);
      ier = c_gradl(input.n(), i, input.h_x.data(), input.h_y.data(), input.h_z.data(),
        input.h_lag_y.data(), del_tri.list.data(), del_tri.lptr.data(), del_tri.lend.data(),
        lag_y_grad);
      LPM_ASSERT(ier >= 6);
      ier = c_gradl(input.n(), i, input.h_x.data(), input.h_y.data(), input.h_z.data(),
        input.h_lag_z.data(), del_tri.list.data(), del_tri.lptr.data(), del_tri.lend.data(),
        lag_z_grad);
      LPM_ASSERT(ier >= 6);

      for (int j=0; j<3; ++j) {
        grad1(j,i) = lag_x_grad[j];
        grad2(j,i) = lag_y_grad[j];
        grad3(j,i) = lag_z_grad[j];
      }
  }

  if (sigma_flag > 0) {
    double dsig;
    int ier;

    ier = c_getsig(input.n(), input.h_x.data(), input.h_y.data(), input.h_z.data(),
      input.h_lag_x.data(), del_tri.list.data(), del_tri.lptr.data(), del_tri.lend.data(),
      grad1.data(), sigma_tol, sigma1.data(), dsig);
    LPM_ASSERT(ier >= 6);

    ier = c_getsig(input.n(), input.h_x.data(), input.h_y.data(), input.h_z.data(),
      input.h_lag_y.data(), del_tri.list.data(), del_tri.lptr.data(), del_tri.lend.data(),
      grad2.data(), sigma_tol, sigma2.data(), dsig);
    LPM_ASSERT(ier >= 6);

    ier = c_getsig(input.n(), input.h_x.data(), input.h_y.data(), input.h_z.data(),
      input.h_lag_z.data(), del_tri.list.data(), del_tri.lptr.data(), del_tri.lend.data(),
      grad3.data(), sigma_tol, sigma3.data(), dsig);
    LPM_ASSERT(ier >= 6);
  }
}

template <typename SeedType>
std::string SSRFPackInterface<SeedType>::info_string(const int tab_lev) const {
  auto tabstr = indent_string(tab_lev);
  std::ostringstream ss;
  ss << tabstr << "SSRFPackInterface<" << SeedType::id_string() << "> info:\n";
  tabstr += "\t";
  ss << tabstr << "n input pts = " << input.n() << "\n";
  ss << tabstr << "scalar_in_out_map.size() = " << scalar_in_out_map.size() << "\n";
  ss << tabstr << "vector_in_out_map.size() = " << vector_in_out_map.size() << "\n";
  ss << tabstr << sigma_str() << "\n";
  ss << tabstr << grad_str() << "\n";
  return ss.str();
}

template <typename SeedType>
std::string SSRFPackInterface<SeedType>::sigma_str() const {
  std::ostringstream ss;
  ss << "sigma (tension factor) flag = " << sigma_flag << " ";
  if (sigma_flag <= 0) {
    ss << "(uniform (constant) tension factor for all triangles)";
  }
  else {
    ss << "(variable tension factors)";
  }
  return ss.str();
}


template <typename SeedType>
std::string SSRFPackInterface<SeedType>::grad_str() const {
  std::ostringstream ss;
  ss << "gradient flag = " << grad_flag << " ";
  if (grad_flag <= 0) {
    ss << "(interpolator will estimate gradients)";
  }
  else {
    ss << "(gradient estimates are pre-computed, relative to interpolation)";
  }
  return ss.str();
}

}  // namespace Lpm

#endif
