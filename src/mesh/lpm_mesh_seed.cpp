#include "lpm_mesh_seed.hpp"
#include <sstream>
#include <iomanip>
#include <fstream>
#include <cmath>

namespace Lpm {

template <typename SeedType>
MeshSeed<SeedType>::MeshSeed(const Real& maxr) : MeshSeed() {
  for (Int i=0; i<seed_crds.extent(0); ++i) {
    const auto mcrd = ko::subview(seed_crds, i, ko::ALL);
    for (Int j=0; j<SeedType::geo::ndim; ++j) {
      mcrd(j) *= maxr;
    }
  }
}

template <typename SeedType>
void MeshSeed<SeedType>::read_file() {
  std::ifstream file(full_filename());
  std::ostringstream oss;
  if (!file.is_open()) {
    oss << "MeshSeed::read_file error: cannot open file " << full_filename();
    LPM_REQUIRE_MSG(false, oss.str());
  }

  /// parse file
  std::string line;
  Int lineNumber = 0;
  Int edgeHeaderLine = constants::NULL_IND;
  Int faceVertHeaderLine = constants::NULL_IND;
  Int faceEdgeHeaderLine = constants::NULL_IND;
  Int vertEdgeHeaderLine = constants::NULL_IND;
  Int crdCounter = 0;
  Int edgeCounter = 0;
  Int faceVertCounter = 0;
  Int faceEdgeCounter = 0;
  Int vertCounter = 0;


  while (std::getline(file, line)) {
    ++lineNumber;
    if (line.find("edgeO") != std::string::npos) {
      edgeHeaderLine = lineNumber;
    }
    if (line.find("faceverts") != std::string::npos) {
      faceVertHeaderLine = lineNumber;
    }
    if (line.find("faceedges") != std::string::npos) {
      faceEdgeHeaderLine = lineNumber;
    }
    if (line.find("vertEdges") != std::string::npos) {
      vertEdgeHeaderLine = lineNumber;
    }
    std::istringstream iss(line);
    if (lineNumber > 1 && lineNumber < ncrds+2) {
      Real x,y,z;
      bool crdErr = false;
      switch (SeedType::geo::ndim) {
        case (2) : {
          if (!(iss >> x >> y))
            crdErr = true;
          seed_crds(crdCounter, 0) = x;
          seed_crds(crdCounter++,1) = y;
          break;
        }
        case (3) : {
          if (!(iss >> x >> y >> z))
            crdErr = true;
          seed_crds(crdCounter, 0) = x;
          seed_crds(crdCounter, 1) = y;
          seed_crds(crdCounter++,2) = z;
          break;
        }
      }
      if (crdErr) {
        oss << "MeshSeed::read_file error: cannot read coordinate from line " << lineNumber
          << " of file " << full_filename();
        LPM_REQUIRE_MSG(false, oss.str());
      }
    }
    else if (edgeHeaderLine > 0 && lineNumber > edgeHeaderLine
      && lineNumber < edgeHeaderLine+SeedType::nedges+1) {
      Index orig;
      Index dest;
      Index left;
      Index right;
      Index cwo;
      Index ccwo;
      Index cwd;
      Index ccwd;
      bool edgeErr = false;
      if (!(iss >> orig >> dest >> left >> right)) {
        edgeErr = true;
      }
      seed_edges(edgeCounter, 0) = orig;
      seed_edges(edgeCounter, 1) = dest;
      seed_edges(edgeCounter, 2) = left;
      seed_edges(edgeCounter, 3) = right;
      ++edgeCounter;
      if (edgeErr) {
        oss << "MeshSeed::read_file error: cannot read edge from line " << lineNumber
          << " of file " << full_filename();
        LPM_REQUIRE_MSG(false, oss.str());
      }
    }
    else if (faceVertHeaderLine > 0 && lineNumber > faceVertHeaderLine
      && lineNumber < faceVertHeaderLine + SeedType::nfaces + 1) {
      Index v0, v1, v2, v3, v4;
      bool faceErr = false;
      switch (SeedType::nfaceverts) {
        case (3) : {
          if (!(iss >> v0 >> v1 >> v2))
            faceErr = true;
          seed_face_verts(faceVertCounter, 0) = v0;
          seed_face_verts(faceVertCounter, 1) = v1;
          seed_face_verts(faceVertCounter++, 2) = v2;
        	break;
        }
        case (4) : {
          if (!(iss >> v0 >> v1 >> v2 >> v3))
            faceErr = true;
          seed_face_verts(faceVertCounter, 0) = v0;
          seed_face_verts(faceVertCounter, 1) = v1;
          seed_face_verts(faceVertCounter, 2) = v2;
          seed_face_verts(faceVertCounter++, 3) = v3;
          break;
        }
        case (5) : {
          if (!(iss >> v0 >> v1 >> v2 >> v3 >> v4))
            faceErr = true;
          seed_face_verts(faceVertCounter, 0) = v0;
          seed_face_verts(faceVertCounter, 1) = v1;
          seed_face_verts(faceVertCounter, 2) = v2;
          seed_face_verts(faceVertCounter, 3) = v3;
          seed_face_verts(faceVertCounter++,4) = v4;
          break;
        }
      }
      if (faceErr) {
        oss << "MeshSeed::read_file error: cannot read face vertices from line " << lineNumber
          << " of file " << full_filename();
        LPM_REQUIRE_MSG(false, oss.str());
      }
    }
    else if (faceEdgeHeaderLine > 0 && lineNumber > faceEdgeHeaderLine
      && lineNumber < faceEdgeHeaderLine + SeedType::nfaces + 1) {
      Index e0, e1, e2, e3, e4;
      bool faceErr = false;
      switch (SeedType::nfaceverts) {
        case (3) : {
          if (!(iss >> e0 >> e1 >> e2))
            faceErr = true;
          seed_face_edges(faceEdgeCounter, 0) = e0;
          seed_face_edges(faceEdgeCounter, 1) = e1;
          seed_face_edges(faceEdgeCounter++, 2) = e2;
          break;
        }
        case (4) : {
          if (!(iss >> e0 >> e1 >> e2 >> e3))
            faceErr = true;
          seed_face_edges(faceEdgeCounter, 0) = e0;
          seed_face_edges(faceEdgeCounter, 1) = e1;
          seed_face_edges(faceEdgeCounter, 2) = e2;
          seed_face_edges(faceEdgeCounter++, 3) = e3;
          break;
        }
        case (5) : {
          if (!(iss >> e0 >> e1 >> e2 >> e3 >> e4))
            faceErr = true;
          seed_face_edges(faceEdgeCounter, 0) = e0;
          seed_face_edges(faceEdgeCounter, 1) = e1;
          seed_face_edges(faceEdgeCounter, 2) = e2;
          seed_face_edges(faceEdgeCounter, 3) = e3;
          seed_face_edges(faceEdgeCounter++, 4) = e4;
          break;
        }
      }
      if (faceErr) {
        oss << "MeshSeed::read_file error: cannot read face edges from line " << lineNumber
          << " of file " << full_filename();
        LPM_REQUIRE_MSG(false, oss.str());
      }
    }
    else if (vertEdgeHeaderLine > 0 && lineNumber > vertEdgeHeaderLine &&
      lineNumber < vertEdgeHeaderLine + SeedType::nverts +1) {
      Index e0, e1, e2, e3, e4, e5;
      bool dualError = false;
      switch (SeedType::vertex_degree) {
        case (4) : {
          if (!(iss >> e0 >> e1 >> e2 >> e3))
            dualError = true;
          seed_vert_edges(vertCounter, 0) = e0;
          seed_vert_edges(vertCounter, 1) = e1;
          seed_vert_edges(vertCounter, 2) = e2;
          seed_vert_edges(vertCounter++, 3) = e3;
          break;
        }
        case (6) : {
          if (!(iss >> e0 >> e1 >> e2 >> e3 >> e4 >> e5))
            dualError = true;
          seed_vert_edges(vertCounter, 0) = e0;
          seed_vert_edges(vertCounter, 1) = e1;
          seed_vert_edges(vertCounter, 2) = e2;
          seed_vert_edges(vertCounter, 3) = e3;
          seed_vert_edges(vertCounter, 4) = e4;
          seed_vert_edges(vertCounter++, 5) = e5;
          break;
        }
      }
      if (dualError) {
        oss << "MeshSeed::read_file error: cannot read vertex edges from line " << lineNumber
          << " of file " << full_filename();
        LPM_REQUIRE_MSG(false, oss.str());
      }
    }
  }
  file.close();
}

template <typename SeedType>
std::string MeshSeed<SeedType>::full_filename() const {
  std::string result(LPM_MESH_SEED_DIR);
  result += "/" + SeedType::filename();
  return result;
}

template <typename SeedType>
std::string MeshSeed<SeedType>::info_string() const {
  std::ostringstream ss;
  ss << "Mesh seed info: id = " << SeedType::id_string() << std::endl;
  ss << "\tseed file = " << full_filename() << std::endl;
  ss << "\tseed coordinates:" <<std::endl;
  for (int i=0; i<ncrds; ++i) {
    ss << "\t(";
    for (int j=0; j<SeedType::geo::ndim; ++j) {
      ss << seed_crds(i,j) << (j == SeedType::geo::ndim-1 ? ")" :",");
    }
    ss << std::endl;
  }
  ss << "\tseed edges:" << std::endl;
  for (int i=0; i<SeedType::nedges; ++i) {
    ss << "\t";
    for (int j=0; j<4; ++j) {
      ss << seed_edges(i,j) << " ";
    }
    ss << std::endl;
  }
  ss << "\tseed face vertices:" << std::endl;
  for (int i=0; i<SeedType::nfaces; ++i) {
    ss << "\t";
    for (int j=0; j<SeedType::nfaceverts; ++j) {
      ss << seed_face_verts(i,j) << " ";
    }
    ss << std::endl;
  }
  ss << "\tseed face edges:" << std::endl;
  for (int i=0; i<SeedType::nfaces; ++i) {
    ss << "\t";
    for (int j=0; j<SeedType::nfaceverts; ++j) {
      ss << seed_face_edges(i,j) << " ";
    }
    ss << std::endl;
  }
  ss << "\tseed vertex edges:" << "\n";
  for (int i=0; i<SeedType::nverts; ++i) {
    ss << "\t";
    for (int j=0; j<SeedType::vertex_degree; ++j) {
      ss << seed_vert_edges(i,j) << " ";
    }
    ss << "\n";
  }
  ss << "\t" << "total_area() = " << total_area() << "\n";
  return ss.str();
}

template <typename SeedType>
void MeshSeed<SeedType>::set_max_allocations(Index& nboundary, Index& nedges, Index& nfaces, const Int lev) const {
  nboundary = SeedType::n_vertices_at_tree_level(lev);
  nedges = 0;
  nfaces = 0;
  for (int i=0; i<=lev; ++i) {
    nfaces += SeedType::n_faces_at_tree_level(i);
    nedges += SeedType::n_edges_at_tree_level(SeedType::n_vertices_at_tree_level(i), SeedType::n_faces_at_tree_level(i));
  }
}

template <typename SeedType>
Real MeshSeed<SeedType>::face_area(const Int ind) const {
  ko::View<Real[SeedType::geo::ndim], Host> ctrcrds("ctrcrds");
  ko::View<Real[SeedType::nfaceverts][SeedType::geo::ndim], Host> vertcrds("vertcrds");
  for (int i=0; i<SeedType::geo::ndim; ++i) {
    ctrcrds(i) = seed_crds(SeedType::nverts+ind, i);
  }
  for (int i=0; i<SeedType::nfaceverts; ++i) {
    for (int j=0; j<SeedType::geo::ndim; ++j) {
      vertcrds(i,j) = seed_crds(seed_face_verts(ind,i), j);
    }
  }
  return SeedType::geo::polygon_area(ctrcrds, vertcrds, SeedType::nfaceverts);
}

template <typename SeedType>
Real MeshSeed<SeedType>::total_area() const {
  Real result = 0;
  for (int i=0; i<SeedType::nfaces; ++i) {
    result += face_area(i);
  }
  return result;
}


Index QuadRectSeed::n_vertices_at_tree_level(const Int lev) {
  Index result = 3;
  for (int i=1; i<=lev; ++i) {
    result += std::pow(2,i);
  }
  result *= result;
  return result;
}

Index UnitDiskSeed::n_faces_at_tree_level(const Int lev) {
  return 5*std::pow(4,lev);
}

Index UnitDiskSeed::n_vertices_at_tree_level(const Int lev) {
  return 2 + 6*std::pow(4,lev);
}

Index UnitDiskSeed::n_edges_at_tree_level(const Index nv, const Index nf) {
  return nv + nf - 1;
}

Index QuadRectSeed::n_faces_at_tree_level(const Int lev) {
  return 4*std::pow(4, lev);
}

Index QuadRectSeed::n_edges_at_tree_level(const Index nv, const Index nf) {
  return nv + nf - 1;
}

Index TriHexSeed::n_vertices_at_tree_level(const Int lev) {
  Index result = 0;
  for (Index i=std::pow(2,lev)+1; i<=std::pow(2,lev+1); ++i) {
    result += i;
  }
  result *= 2;
  result += std::pow(2,lev+1)+1;
  return result;
}

Index TriHexSeed::n_faces_at_tree_level(const Int lev) {
  return 6*std::pow(4,lev);
}

Index TriHexSeed::n_edges_at_tree_level(const Index nv, const Index nf) {
  return nv + nf - 1;
}

Index CubedSphereSeed::n_vertices_at_tree_level(const Int lev) {
  return 2+6*std::pow(4,lev);
}

Index CubedSphereSeed::n_faces_at_tree_level(const Int lev) {
  return 6*std::pow(4,lev);
}

Index CubedSphereSeed::n_edges_at_tree_level(const Index nv, const Index nf) {
  return nf + nv - 2;
}

Index IcosTriSphereSeed::n_vertices_at_tree_level(const Int lev) {
  return 2 + 10*std::pow(4,lev);
}

Index IcosTriSphereSeed::n_faces_at_tree_level(const Int lev) {
  return 20*std::pow(4,lev);
}

Index IcosTriSphereSeed::n_edges_at_tree_level(const Index nv, const Index nf) {
  return nv + nf - 2;
}

/// ETI
template struct MeshSeed<QuadRectSeed>;
template struct MeshSeed<TriHexSeed>;
template struct MeshSeed<IcosTriSphereSeed>;
template struct MeshSeed<CubedSphereSeed>;
}
