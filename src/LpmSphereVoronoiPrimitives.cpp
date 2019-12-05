#include <iostream>
#include <iomanip>
#include <sstream>

#include "LpmSphereVoronoiPrimitives.hpp"

namespace Lpm {
namespace Voronoi {

std::string Cell::infoString() const {
    std::stringstream ss;
    ss << "Cell info: xyz = (" << std::setw(6) << xyz[0] << ", " << std::setw(6) << xyz[1] << ", "
       << std::setw(6) << xyz[2] << ") latlon = (" << std::setw(6) << latlon[0] << ", " << std::setw(6)
       << latlon[1] << ") edgeAroundCell = " << edgeAroundCell << "\n";
    return ss.str();
}

std::string Vertex::infoString() const {
    std::ostringstream ss;
    ss << "Vertex info: xyz = (" << std::setw(6) << xyz[0] << ", " << std::setw(6) << xyz[1] << ", "
       << std::setw(6) << xyz[2] << ") latlon = (" << std::setw(6) << latlon[0] << ", " << std::setw(6)
       << latlon[1] << ") edgeAtVertex = " << edgeAtVertex << "\n";
    return ss.str();
}

std::string WingedEdge::infoString() const {
    std::ostringstream ss;
    ss << "WingedEdge info: xyz = (" << std::setw(6) << xyz[0] << ", " << std::setw(6) << xyz[1] << ", "
       << std::setw(6) << xyz[2] << ") latlon = (" << std::setw(6) << latlon[0] << ", " << std::setw(6)
       << latlon[1] << ") orig_vertex = " << orig_vertex << " dest_vertex = " << dest_vertex << "\n";
    return ss.str();
}

}}
