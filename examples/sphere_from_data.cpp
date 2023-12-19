#include <iostream>
#include <sstream>
#include <iomanip>
#include "LpmConfig.h"
#ifdef LPM_USE_NETCDF

#include "lpm_geometry.hpp"
#include "lpm_comm.hpp"
#include "lpm_logger.hpp"
#include "lpm_compadre.hpp"
#include "lpm_kokkos_defs.hpp"
#include "lpm_field.hpp"
#include "lpm_field_impl.hpp"
#include "mesh/lpm_mesh_seed.hpp"
#include "mesh/lpm_polymesh2d.hpp"
#include "mesh/lpm_polymesh2d_impl.hpp"
#include "mesh/lpm_refinement_flags.hpp"
#include "mesh/lpm_gather_mesh_data.hpp"
#include "mesh/lpm_gather_mesh_data_impl.hpp"
#include "mesh/lpm_scatter_mesh_data.hpp"
#include "mesh/lpm_scatter_mesh_data_impl.hpp"
#ifdef LPM_USE_VTK
#include "vtk/lpm_vtk_io.hpp"
#include "vtk/lpm_vtk_io_impl.hpp"
#endif

#include "netcdf/lpm_netcdf.hpp"
#include "netcdf/lpm_netcdf_impl.hpp"
#include "netcdf/lpm_netcdf_reader.hpp"
#include "netcdf/lpm_netcdf_reader_impl.hpp"

#include <Compadre_Evaluator.hpp>
#include <sstream>
#include <string>
#include <memory>
#include <mpi.h>

using namespace Lpm;

struct Input {
  static constexpr int init_depth_default = 2;
  int init_depth;
  static constexpr int amr_refinement_limit_default = 0;
  int amr_refinement_limit;
  static constexpr int amr_refinement_buffer_default = 0;
  int amr_refinement_buffer;
  int source_timestep;
  int gmls_order;
  Real scalar_max_tol;
  Real scalar_integral_tol;
  Real scalar_var_tol;
  std::string src_filename;
  std::string ofroot;
  std::string vtk_fname;
  std::string nc_fname;
  std::string field_name;

  Input(int argc, char* argv[]);

  std::string info_string() const;

  std::string usage() const;

  bool do_amr() const;

  bool help_and_exit;
};

/**
  Choose a MeshSeed
*/
typedef CubedSphereSeed seed_type;
// typedef IcosTriSphereSeed seed_type;

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  Comm comm(MPI_COMM_WORLD);
  LPM_REQUIRE_MSG(comm.size() == 1, "Distributed data not implemented.");
  ko::initialize(argc, argv);
  {
    Logger<> logger("sphere_mesh_from_data", Log::level::info, comm);

    Input input(argc, argv);
    if (input.help_and_exit or input.src_filename.empty()) {
      return 1;
    }
    logger.info(input.info_string());

    /**
        Load data from source file
    */
    UnstructuredNcReader<SphereGeometry> reader(input.src_filename);
    logger.info(reader.info_string());

    Coords<SphereGeometry> src_coords = reader.create_coords();
    logger.info(src_coords.info_string());
    const auto src_data = reader.create_scalar_field(input.field_name, input.source_timestep);

    /**
        Initialize target mesh
    */
    MeshSeed<seed_type> seed;

    /**
      Build the particle/panel mesh
    */
    constexpr Real sphere_radius = 1;
    auto mesh_params = PolyMeshParameters<seed_type>(input.init_depth,
      sphere_radius,
      input.amr_refinement_buffer);


    auto sphere = std::make_unique<PolyMesh2d<seed_type>>(mesh_params);
    sphere->update_device();
    ScalarField<VertexField> gmls_verts(input.field_name, sphere->n_vertices_host());
    ScalarField<FaceField> gmls_faces(input.field_name, sphere->n_faces_host());
    logger.info(sphere->info_string());

    /**
        Setup uniform interpolation from source data
    */
    std::map<std::string, ScalarField<VertexField>> vert_field_map;
    std::map<std::string, ScalarField<FaceField>> face_field_map;
    vert_field_map.emplace(input.field_name, gmls_verts);
    face_field_map.emplace(input.field_name, gmls_faces);

    gmls::Params gmls_params(input.gmls_order);
    const std::vector<Compadre::TargetOperation> gmls_ops({Compadre::ScalarPointEvaluation});
    {
      GatherMeshData<seed_type> gather(*sphere);
      gather.update_host();
      gather.init_scalar_fields(vert_field_map, face_field_map);

      gmls::Neighborhoods neighbors(src_coords.get_host_crd_view(), gather.h_phys_crds, gmls_params);
      auto scalar_gmls = gmls::sphere_scalar_gmls(src_coords.get_host_crd_view(),
        gather.phys_crds, neighbors, gmls_params, gmls_ops);

      /**
        Interpolate to leaf particles
      */
      Compadre::Evaluator gmls_eval(&scalar_gmls);
      gather.scalar_fields.at(input.field_name) =
        gmls_eval.applyAlphasToDataAllComponentsAllTargetSites<Real*, DevMemory>(
          src_data.view,
          Compadre::ScalarPointEvaluation,
          Compadre::PointSample);
      /**
        Scatter data to uniform mesh
      */
      ScatterMeshData<seed_type> scatter(gather, *sphere);
      scatter.scatter_fields(vert_field_map, face_field_map);
      logger.info(gmls_verts.info_string());
      logger.info(gmls_faces.info_string());
    }
    if (input.do_amr() ) {
      /**
        To start adaptive refinement, we convert relative tolerances from the
        input to absolute tolerances based on the initialized uniform mesh and
        functions defined on it.
      */
      typename Kokkos::MinMax<Real>::value_type minmax;
      Kokkos::parallel_reduce(reader.n_points(),
        KOKKOS_LAMBDA (const Index i, typename Kokkos::MinMax<Real>::value_type& mm) {
          if (src_data.view(i) > mm.max_val) mm.max_val = src_data.view(i);
          if (src_data.view(i) < mm.min_val) mm.min_val = src_data.view(i);
        }
      , Kokkos::MinMax<Real>(minmax));

      const Real max_tol = input.scalar_max_tol * minmax.max_val;
      const Real mass_tol = input.scalar_integral_tol * minmax.max_val * square(sphere->appx_mesh_size());
      const Real var_tol = input.scalar_var_tol * (minmax.max_val - minmax.min_val);

      Kokkos::View<bool*> flags("refinement_flags", mesh_params.nmaxfaces);
      auto h_flags = Kokkos::create_mirror_view(flags);
      Index verts_start_idx = 0;
      Index faces_start_idx = 0;
      for (int i=0; i<input.amr_refinement_limit; ++i) {
        Index verts_end_idx = sphere->n_vertices_host();
        Index faces_end_idx = sphere->n_faces_host();

        auto range = Kokkos::RangePolicy<>(faces_start_idx, faces_end_idx);

        if (input.scalar_max_tol > 0) {
          Kokkos::parallel_for(range,
            ScalarMaxFlag(flags, gmls_faces.view, sphere->faces.mask, sphere->n_faces_host(), max_tol));
        }
        if (input.scalar_integral_tol > 0) {
          Kokkos::parallel_for(range,
            ScalarIntegralFlag(flags, gmls_faces.view, sphere->faces.area, sphere->faces.mask, sphere->n_faces_host(), mass_tol));
        }
        if (input.scalar_var_tol > 0) {
          Kokkos::parallel_for(range,
            ScalarVariationFlag(flags, gmls_faces.view, gmls_verts.view, sphere->faces.verts, sphere->faces.mask, sphere->n_faces_host(), var_tol));
        }

        Kokkos::deep_copy(h_flags, flags);

        sphere->divide_flagged_faces(flags, logger);

        GatherMeshData<seed_type> gather(*sphere);
        gather.update_host();
        gather.init_scalar_fields(vert_field_map, face_field_map);

        gmls::Neighborhoods neighbors(src_coords.get_host_crd_view(), gather.h_phys_crds, gmls_params);
        auto scalar_gmls = gmls::sphere_scalar_gmls(src_coords.get_host_crd_view(),
          gather.phys_crds, neighbors, gmls_params, gmls_ops);

        /**
          Interpolate to leaf particles
        */
        Compadre::Evaluator gmls_eval(&scalar_gmls);
        gather.scalar_fields.at(input.field_name) =
          gmls_eval.applyAlphasToDataAllComponentsAllTargetSites<Real*, DevMemory>(
            src_data.view,
            Compadre::ScalarPointEvaluation,
            Compadre::PointSample);
        /**
          Scatter data to full mesh
        */
        ScatterMeshData<seed_type> scatter(gather, *sphere);
        scatter.scatter_fields(vert_field_map, face_field_map);
        logger.info(gmls_verts.info_string());
        logger.info(gmls_faces.info_string());

        Kokkos::deep_copy(flags, false);

        verts_start_idx = verts_end_idx;
        faces_start_idx = faces_end_idx;
      }
    }

#ifdef LPM_USE_VTK
    /** Output mesh to a vtk file */
    VtkPolymeshInterface<seed_type> vtk(*sphere);
    vtk.add_scalar_point_data(gmls_verts.view);
    vtk.add_scalar_cell_data(gmls_faces.view);
    vtk.write(input.vtk_fname);
#endif
#ifdef LPM_USE_NETCDF
// TODO
//       * Output mesh to a netCDF file */
//       NcWriter<SphereGeometry> nc(input.nc_fname);
//       nc.define_polymesh(*sphere);
//       logger.debug(nc.info_string());
#endif
  }

  ko::finalize();
  MPI_Finalize();
return 0;
}

Input::Input(int argc, char* argv[]) {
  init_depth = init_depth_default;
  amr_refinement_buffer = amr_refinement_buffer_default;
  amr_refinement_limit = amr_refinement_limit_default;
  scalar_max_tol = 0;
  scalar_integral_tol = 0;
  scalar_var_tol = 0;
  ofroot = "sph_from_data_";
  help_and_exit = false;
  source_timestep = 0;
  gmls_order = 3;
  if (argc < 2) {
    src_filename = "";
  }
  else {
    src_filename = argv[1];
  }
  for (int i=2; i<argc; ++i) {
    const std::string& token = argv[i];
    if (token == "-o") {
      vtk_fname = argv[++i];
    }
    else if (token == "-d" or token == "--init-depth") {
      init_depth = std::stoi(argv[++i]);
      LPM_REQUIRE(init_depth >= 0);
    }
    else if (token == "-h") {
      std::cout << usage() << "\n";
      help_and_exit = true;
    }
    else if (token == "-i") {
      src_filename = argv[++i];
    }
    else if (token == "-amr") {
      amr_refinement_buffer = std::stoi(argv[++i]);
      amr_refinement_limit = amr_refinement_buffer;
      LPM_REQUIRE(amr_refinement_buffer >= 0);
    }
    else if (token == "--amr-limit") {
      amr_refinement_limit = std::stoi(argv[++i]);
      LPM_REQUIRE(amr_refinement_limit >= 0);
    }
    else if (token == "--amr-buffer") {
      amr_refinement_buffer = std::stoi(argv[++i]);
      LPM_REQUIRE(amr_refinement_buffer >= 0);
    }
    else if (token == "--max-tol") {
      scalar_max_tol = std::stod(argv[++i]);
      LPM_REQUIRE(scalar_max_tol > 0);
    }
    else if (token == "--integral-tol") {
      scalar_integral_tol = std::stod(argv[++i]);
      LPM_REQUIRE(scalar_integral_tol > 0);
    }
    else if (token == "--var-tol") {
      scalar_var_tol = std::stod(argv[++i]);
      LPM_REQUIRE(scalar_var_tol > 0);
    }
    else if (token == "-f" or token == "--field-name") {
      field_name = argv[++i];
    }
    else if (token == "-t") {
      source_timestep = std::stoi(argv[++i]);
    }
    else if (token == "-g" or token == "--gmls-order") {
      gmls_order = std::stoi(argv[++i]);
      LPM_REQUIRE(gmls_order > 0);
    }
  }
  bool amr_valid = ( (amr_refinement_buffer == 0 and amr_refinement_limit == 0)
                  or (amr_refinement_buffer > 0 and amr_refinement_limit > 0) );
  LPM_REQUIRE_MSG(amr_valid, "valid input for amr requires both refinement buffer and refinement limit to be zero (no amr) or to be positive");
  ofroot += (std::is_same<CubedSphereSeed, seed_type>::value ?
    "cubed_sphere" : "icos_tri_sphere") + std::to_string(init_depth)
      + (amr_refinement_buffer > 0 ? "_amr" + std::to_string(amr_refinement_buffer) : "_unif")
      + "_t" + std::to_string(source_timestep);
  vtk_fname = ofroot + ".vtp";
  nc_fname = ofroot + ".nc";
}

std::string Input::usage() const {
  std::ostringstream ss;
  ss << "Sphere Mesh from data: This program initializes a uniform spherical mesh \n" <<
    "and writes the mesh to data files in 2 formats: \n\tVTK's .vtp format and the NetCDF4 .nc format.\n";
  ss << "\t" << "first argument, or '-i' option: source data file.\n";
  ss << "\t" << "optional arguments:\n";
  ss << "\t   " << "-o [output_filename_root] \n";
  ss << "\t   " << "-i [filename] source data file path\n";
  ss << "\t   " << "-d [nonnegative integer] ; defines the initial depth of the uniform mesh's face quadtree.\n";
  ss << "\t   " << "-f --field-name ; name of the variable to interpolate\n";
  ss << "\t   " << "-g --gmls-order ; order of the GMLS basis polynomials\n";
  ss << "\t   " << "-amr [nonnegative integer]; sets both amr buffer and amr limit\n";
  ss << "\t   " << "--amr-limit [nonnegative integer]; sets the maximum number of times a panel may be refined relative to its immediate neighbors; --amr-buffer must be positive whenever this option is positive";
  ss << "\t   " << "--amr-buffer [nonnegative integer]; sets maximum memory limit for the particle/panel mesh to be a uniform mesh of depth init-depth + buffer.  --amr-limit must be positive whenever --amr-buffer is positive\n";
  ss << "\t   " << "--max-tol sets maximum value threshold for scalar data amr\n";
  ss << "\t   " << "--integral-tol sets maximum integral threshold for scalar data amr\n";
  ss << "\t   " << "--var-tol sets maximum variation threshold for scalar data amr\n";
  ss << "\t   " << "-h print help message and exit\n";
  return ss.str();
}

bool Input::do_amr() const {
  bool result = (amr_refinement_buffer > 0 and amr_refinement_limit > 0);
  return result;
}

std::string Input::info_string() const {
  std::ostringstream ss;
  ss << "Sphere mesh from data:\n";
  ss << "\tsource data file: " << src_filename << "\n";
  ss << "\toutput vtk file: " << vtk_fname << "\n";
  ss << "\toutput netcdf file: " << nc_fname << "\n";
  ss << "\tInitializing from seed: " << seed_type::id_string() << "\n";
  ss << "\tTo depth: " << init_depth << "\n";
  ss << "\tvariable name: " << field_name << "\n";
  ss << "\tgmls_oder: " << gmls_order << "\n";
  if (amr_refinement_buffer > 0) {
    ss << "\tAMR:\n";
    ss << "\t\t(buffer, limit) = (" << amr_refinement_buffer << ", " << amr_refinement_limit << ")\n";
    ss << "\t\tmax_tol = " << scalar_max_tol << "\n";
    ss << "\t\tintegral_tol = " << scalar_integral_tol << "\n";
    ss << "\t\tvar_tol = " << scalar_var_tol << "\n";
  }
  return ss.str();
}

#endif // LPM_USE_NETCDF
