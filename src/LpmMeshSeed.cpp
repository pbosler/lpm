#include "LpmMeshSeed.hpp"
#include "LpmSphereVoronoiPrimitives.hpp"
#include <sstream>
#include <iomanip>
#include <fstream>
#include <cmath>

namespace Lpm {

template <typename SeedType>
MeshSeed<SeedType>::MeshSeed(const Real& maxr) : MeshSeed() {
  for (Int i=0; i<scrds.extent(0); ++i) {
    const auto mcrd = ko::subview(scrds, i, ko::ALL);
    for (Int j=0; j<SeedType::geo::ndim; ++j) {
      mcrd(j) *= maxr;
    }
  }
}

template <typename SeedType>
void MeshSeed<SeedType>::readfile() {
  std::ifstream file(fullFilename());
  std::ostringstream oss;
  if (!file.is_open()) {
    oss << "MeshSeed::readfile error: cannot open file " << fullFilename();
    LPM_THROW_IF(true, oss.str());
  }

  /// parse file
  std::string line;
  Int lineNumber = 0;
  Int edgeHeaderLine = NULL_IND;
  Int faceVertHeaderLine = NULL_IND;
  Int faceEdgeHeaderLine = NULL_IND;
  Int vertEdgeHeaderLine = NULL_IND;
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
          scrds(crdCounter, 0) = x;
          scrds(crdCounter++,1) = y;
          break;
        }
        case (3) : {
          if (!(iss >> x >> y >> z))
            crdErr = true;
          scrds(crdCounter, 0) = x;
          scrds(crdCounter, 1) = y;
          scrds(crdCounter++,2) = z;
          break;
        }
      }
      if (crdErr) {
        oss << "MeshSeed::readfile error: cannot read coordinate from line " << lineNumber
          << " of file " << fullFilename();
        LPM_THROW_IF(true, oss.str());
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
      if (SeedType::isDual) {
        if (!(iss >> orig >> dest >> left >> right >> cwo >> ccwo >> cwd >> ccwd))
          edgeErr = true;
      }
      else {
        if (!(iss >> orig >> dest >> left >> right))
          edgeErr = true;
      }
      sedges(edgeCounter, 0) = orig;
      sedges(edgeCounter, 1) = dest;
      sedges(edgeCounter, 2) = left;
      sedges(edgeCounter, 3) = right;
      if (SeedType::isDual) {
        sedges(edgeCounter,4) = cwo;
        sedges(edgeCounter,5) = ccwo;
        sedges(edgeCounter,6) = cwd;
        sedges(edgeCounter,7) = ccwd;
      }
      ++edgeCounter;
      if (edgeErr) {
        oss << "MeshSeed::readfile error: cannot read edge from line " << lineNumber
          << " of file " << fullFilename();
        LPM_THROW_IF(true, oss.str());
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
          sfaceverts(faceVertCounter, 0) = v0;
          sfaceverts(faceVertCounter, 1) = v1;
          sfaceverts(faceVertCounter++, 2) = v2;
        	break;
        }
        case (4) : {
          if (!(iss >> v0 >> v1 >> v2 >> v3))
            faceErr = true;
          sfaceverts(faceVertCounter, 0) = v0;
          sfaceverts(faceVertCounter, 1) = v1;
          sfaceverts(faceVertCounter, 2) = v2;
          sfaceverts(faceVertCounter++, 3) = v3;
          break;
        }
        case (5) : {
          if (!(iss >> v0 >> v1 >> v2 >> v3 >> v4))
            faceErr = true;
          sfaceverts(faceVertCounter, 0) = v0;
          sfaceverts(faceVertCounter, 1) = v1;
          sfaceverts(faceVertCounter, 2) = v2;
          sfaceverts(faceVertCounter, 3) = v3;
          sfaceverts(faceVertCounter++,4) = v4;
          break;
        }
      }
      if (faceErr) {
        oss << "MeshSeed::readfile error: cannot read face vertices from line " << lineNumber
          << " of file " << fullFilename();
        LPM_THROW_IF(true, oss.str());
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
          sfaceedges(faceEdgeCounter, 0) = e0;
          sfaceedges(faceEdgeCounter, 1) = e1;
          sfaceedges(faceEdgeCounter++, 2) = e2;
          break;
        }
        case (4) : {
          if (!(iss >> e0 >> e1 >> e2 >> e3))
            faceErr = true;
          sfaceedges(faceEdgeCounter, 0) = e0;
          sfaceedges(faceEdgeCounter, 1) = e1;
          sfaceedges(faceEdgeCounter, 2) = e2;
          sfaceedges(faceEdgeCounter++, 3) = e3;
          break;
        }
        case (5) : {
          if (!(iss >> e0 >> e1 >> e2 >> e3 >> e4))
            faceErr = true;
          sfaceedges(faceEdgeCounter, 0) = e0;
          sfaceedges(faceEdgeCounter, 1) = e1;
          sfaceedges(faceEdgeCounter, 2) = e2;
          sfaceedges(faceEdgeCounter, 3) = e3;
          sfaceedges(faceEdgeCounter++, 4) = e4;
          break;
        }
      }
      if (faceErr) {
        oss << "MeshSeed::readfile error: cannot read face edges from line " << lineNumber
          << " of file " << fullFilename();
        LPM_THROW_IF(true, oss.str());
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
          svertedges(vertCounter, 0) = e0;
          svertedges(vertCounter, 1) = e1;
          svertedges(vertCounter, 2) = e2;
          svertedges(vertCounter++, 3) = e3;
          break;
        }
        case (6) : {
          if (!(iss >> e0 >> e1 >> e2 >> e3 >> e4 >> e5))
            dualError = true;
          svertedges(vertCounter, 0) = e0;
          svertedges(vertCounter, 1) = e1;
          svertedges(vertCounter, 2) = e2;
          svertedges(vertCounter, 3) = e3;
          svertedges(vertCounter, 4) = e4;
          svertedges(vertCounter++, 5) = e5;
          break;
        }
      }
      if (dualError) {
        oss << "MeshSeed::readfile error: cannot read vertex edges from line " << lineNumber
          << " of file " << fullFilename();
        LPM_THROW_IF(true, oss.str());
      }
    }
  }
  file.close();
}

template <typename SeedType>
std::string MeshSeed<SeedType>::fullFilename() const {
  std::string result(LPM_MESH_SEED_DIR);
  result += "/" + SeedType::filename();
  return result;
}

template <typename SeedType>
std::string MeshSeed<SeedType>::infoString() const {
  std::ostringstream ss;
  ss << "Mesh seed info: id = " << SeedType::idString() << std::endl;
  ss << "\tseed file = " << fullFilename() << std::endl;
  ss << "\tseed coordinates:" <<std::endl;
  for (int i=0; i<ncrds; ++i) {
    ss << "\t(";
    for (int j=0; j<SeedType::geo::ndim; ++j) {
      ss << scrds(i,j) << (j == SeedType::geo::ndim-1 ? ")" :",");
    }
    ss << std::endl;
  }
  ss << "\tseed edges:" << std::endl;
  for (int i=0; i<SeedType::nedges; ++i) {
    ss << "\t";
    if (SeedType::isDual) {
      for (int j=0; j<8; ++j) {
        ss << sedges(i,j) << " ";
      }
    }
    else {
      for (int j=0; j<4; ++j) {
        ss << sedges(i,j) << " ";
      }
    }
    ss << std::endl;
  }
  ss << "\tseed face vertices:" << std::endl;
  for (int i=0; i<SeedType::nfaces; ++i) {
    ss << "\t";
    for (int j=0; j<SeedType::nfaceverts; ++j) {
      ss << sfaceverts(i,j) << " ";
    }
    ss << std::endl;
  }
  ss << "\tseed face edges:" << std::endl;
  for (int i=0; i<SeedType::nfaces; ++i) {
    ss << "\t";
    for (int j=0; j<SeedType::nfaceverts; ++j) {
      ss << sfaceedges(i,j) << " ";
    }
    ss << std::endl;
  }
  ss << "\tseed vertex edges:" << "\n";
  for (int i=0; i<SeedType::nverts; ++i) {
    ss << "\t";
    for (int j=0; j<SeedType::vertex_degree; ++j) {
      ss << svertedges(i,j) << " ";
    }
    ss << "\n";
  }
  return ss.str();
}

template <typename SeedType>
void MeshSeed<SeedType>::setMaxAllocations(Index& nboundary, Index& nedges, Index& nfaces, const Int lev) const {
  nboundary = SeedType::nVerticesAtTreeLevel(lev);
  nedges = 0;
  nfaces = 0;
  for (int i=0; i<=lev; ++i) {
    nfaces += SeedType::nFacesAtTreeLevel(i);
    nedges += SeedType::nEdgesAtTreeLevel(SeedType::nVerticesAtTreeLevel(i), SeedType::nFacesAtTreeLevel(i));
  }
}

template <typename SeedType>
Real MeshSeed<SeedType>::faceArea(const Int ind) const {
  ko::View<Real[SeedType::geo::ndim], Host> ctrcrds("ctrcrds");
  ko::View<Real[SeedType::nfaceverts][SeedType::geo::ndim], Host> vertcrds("vertcrds");
  for (int i=0; i<SeedType::geo::ndim; ++i) {
    ctrcrds(i) = scrds(SeedType::nverts+ind, i);
  }
  for (int i=0; i<SeedType::nfaceverts; ++i) {
    for (int j=0; j<SeedType::geo::ndim; ++j) {
      vertcrds(i,j) = scrds(sfaceverts(ind,i), j);
    }
  }
  return SeedType::geo::polygonArea(ctrcrds, vertcrds, SeedType::nfaceverts);
}

Index QuadRectSeed::nVerticesAtTreeLevel(const Int lev) {
  Index result = 3;
  for (int i=1; i<=lev; ++i) {
    result += std::pow(2,i);
  }
  result *= result;
  return result;
}

Index UnitDiskSeed::nFacesAtTreeLevel(const Int lev) {
  return 5*std::pow(4,lev);
}

Index UnitDiskSeed::nVerticesAtTreeLevel(const Int lev) {
  return 2 + 6*std::pow(4,lev);
}

Index UnitDiskSeed::nEdgesAtTreeLevel(const Index nv, const Index nf) {
  return nv + nf - 1;
}

Index QuadRectSeed::nFacesAtTreeLevel(const Int lev) {
  return 4*std::pow(4, lev);
}

Index QuadRectSeed::nEdgesAtTreeLevel(const Index nv, const Index nf) {
  return nv + nf - 1;
}

Index TriHexSeed::nVerticesAtTreeLevel(const Int lev) {
  Index result = 0;
  for (Index i=std::pow(2,lev)+1; i<=std::pow(2,lev+1); ++i) {
    result += i;
  }
  result *= 2;
  result += std::pow(2,lev+1)+1;
  return result;
}

Index TriHexSeed::nFacesAtTreeLevel(const Int lev) {
  return 6*std::pow(4,lev);
}

Index TriHexSeed::nEdgesAtTreeLevel(const Index nv, const Index nf) {
  return nv + nf - 1;
}

Index CubedSphereSeed::nVerticesAtTreeLevel(const Int lev) {
  return 2+6*std::pow(4,lev);
}

Index CubedSphereSeed::nFacesAtTreeLevel(const Int lev) {
  return 6*std::pow(4,lev);
}

Index CubedSphereSeed::nEdgesAtTreeLevel(const Index nv, const Index nf) {
  return nf + nv - 2;
}

Index IcosTriSphereSeed::nVerticesAtTreeLevel(const Int lev) {
  return 2 + 10*std::pow(4,lev);
}

Index IcosTriSphereSeed::nFacesAtTreeLevel(const Int lev) {
  return 20*std::pow(4,lev);
}

Index IcosTriSphereSeed::nEdgesAtTreeLevel(const Index nv, const Index nf) {
  return nv + nf - 2;
}

Index IcosTriDualSeed::nFacesAtTreeLevel(const Int lev) {
  return Voronoi::nfacesAtUniformRefinementLevel(12, lev);
}

Index IcosTriDualSeed::nVerticesAtTreeLevel(const Int lev) {
  return Voronoi::nvertsAtUniformRefinementLevel(12, lev);
}

Index IcosTriDualSeed::nEdgesAtTreeLevel(const Index nv, const Index nf) {
  return nf + nv - 2;
}




/// ETI
template struct MeshSeed<QuadRectSeed>;
template struct MeshSeed<TriHexSeed>;
template struct MeshSeed<IcosTriSphereSeed>;
template struct MeshSeed<CubedSphereSeed>;
template struct MeshSeed<IcosTriDualSeed>;
template struct MeshSeed<UnitDiskSeed>;

}
