#include <iostream>
#include <iomanip>
#include <sstream>

#include "LpmSphereVoronoiPrimitives.hpp"

namespace Lpm {
namespace Voronoi {

std::string Cell::infoString() const {
    std::stringstream ss;
    ss << "Cell info: xyz = (" << std::setw(4) << xyz[0] << ", " << std::setw(4) << xyz[1] << ", "
       << std::setw(4) << xyz[2] << ") latlon = (" << std::setw(4) << latlon[0] << ", " << std::setw(4)
       << latlon[1] << ") edgeAroundCell = " << edgeAroundCell << "\n";
    return ss.str();
}

std::string Vertex::infoString() const {
    std::ostringstream ss;
    ss << "Vertex info: xyz = (" << std::setw(4) << xyz[0] << ", " << std::setw(4) << xyz[1] << ", "
       << std::setw(4) << xyz[2] << ") latlon = (" << std::setw(4) << latlon[0] << ", " << std::setw(4)
       << latlon[1] << ") edgeAtVertex = " << edgeAtVertex << "\n";
    return ss.str();
}

std::string WingedEdge::infoString() const {
    std::ostringstream ss;
    ss << "WingedEdge info: xyz = (" << std::setw(4) << xyz[0] << ", " << std::setw(4) << xyz[1] << ", "
       << std::setw(4) << xyz[2] << ") latlon = (" << std::setw(4) << latlon[0] << ", " << std::setw(4)
       << latlon[1] << ") orig_vertex = " << orig_vertex << " dest_vertex = " << dest_vertex
       << " cw_orig = " << cw_orig << " ccw_orig = " << ccw_orig << " cw_dest = " << cw_dest
       << " ccw_dest = " << ccw_dest << "\n";
    return ss.str();
}

Index nfacesAtUniformRefinementLevel(const Short& nrootfaces, const Short& lev) {
    Index nf = nrootfaces;
    Index nv = 2*nf-4;
    for (Short i=1; i<=lev; ++i) {
        nf = nf + nv;
        nv = 2*nf - 4;
    }
    return nf;
}

Index nvertsAtUniformRefinementLevel(const Short& nrootfaces, const Short& lev) {
    Index nf = nrootfaces;
    Index nv = 2*nf-4;
    for (Short i=1; i<=lev; ++i) {
        nf = nf + nv;
        nv = 2*nf-4;
    }
    return nv;
}


}}
