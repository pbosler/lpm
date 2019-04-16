#include "LpmMeshSeed.hpp"
#include <sstream>
#include <iomanip>
#include <fstream>
#include <cmath>

namespace Lpm {

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
    Int crdCounter = 0;
    Int edgeCounter = 0;
    Int faceVertCounter = 0;
    Int faceEdgeCounter = 0;
    
    
    
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
            bool edgeErr = false;
            if (!(iss >> orig >> dest >> left >> right))
                edgeErr = true;
            sedges(edgeCounter, 0) = orig;
            sedges(edgeCounter, 1) = dest;
            sedges(edgeCounter, 2) = left;
            sedges(edgeCounter++, 3) = right;
            if (edgeErr) {
                oss << "MeshSeed::readfile error: cannot read edge from line " << lineNumber
                    << " of file " << fullFilename();
                LPM_THROW_IF(true, oss.str());
            }
        }
        else if (faceVertHeaderLine > 0 && lineNumber > faceVertHeaderLine
            && lineNumber < faceVertHeaderLine + SeedType::nfaces + 1) {
            Index v0, v1, v2, v3;
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
            }
            if (faceErr) {
                oss << "MeshSeed::readfile error: cannot read face vertices from line " << lineNumber
                    << " of file " << fullFilename();
                LPM_THROW_IF(true, oss.str());
            }
        }
        else if (faceEdgeHeaderLine > 0 && lineNumber > faceEdgeHeaderLine 
            && lineNumber < faceEdgeHeaderLine + SeedType::nfaces + 1) {
            Index e0, e1, e2, e3;
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
            }
            if (faceErr) {
                oss << "MeshSeed::readfile error: cannot read face edges from line " << lineNumber
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
        for (int j=0; j<4; ++j) {
            ss << sedges(i,j) << " ";
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

/// ETI
template struct MeshSeed<QuadRectSeed>;
template struct MeshSeed<TriHexSeed>;
template struct MeshSeed<IcosTriSphereSeed>;
template struct MeshSeed<CubedSphereSeed>;

}
