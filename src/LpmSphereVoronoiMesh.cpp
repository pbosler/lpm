#include "LpmSphereVoronoiMesh.hpp"
#include <cassert>

namespace Lpm {
namespace Voronoi {

void VoronoiMesh::getEdgesAndCellsAtVertex(std::vector<Index>& edge_inds, std::vector<Index>& cell_inds,
	const Index& vert_ind) const {

	edge_inds.clear();
	cell_inds.clear();

	Index k = vertices[vert_ind].edgeAtVertex;
	Index kstart = k;
	bool keepGoing = true;
	Short ctr = 0;
	while (keepGoing) {
		edge_inds.push_back(k);

		assert(vert_ind == edges[k].orig_vertex || vert_ind == edges[k].dest_vertex);

		if (vert_ind == edges[k].orig_vertex) {
			cell_inds.push_back(edges[k].left_cell);
			k = edges[k].ccw_orig;
		}
		else {
			cell_inds.push_back(edges[k].right_cell);
			k = edges[k].ccw_dest;
		}
		ctr++;
		assert( ctr <= MAX_VERTEX_DEGREE );
		keepGoing = !(k == kstart);
	}
}

void VoronoiMesh::getEdgesAndVerticesInCell(std::vector<Index>& edge_inds, std::vector<Index>& vert_inds,
	const Index& cell_ind) const {

	edge_inds.clear();
	vert_inds.clear();

	Index k = cells[cell_ind].edgeAroundCell;
	Index kstart = k;
	bool keepGoing = true;
	Short ctr = 0;
	while (keepGoing) {
		edge_inds.push_back(k);

		assert(cell_ind == edges[k].left_cell || cell_ind == edges[k].right_cell);

		if (cell_ind == edges[k].left_cell) {
			vert_inds.push_back(edges[k].dest_vertex);
			k = edges[k].cw_dest;
		}
		else {
			vert_inds.push_back(edges[k].orig_vertex);
			k = edges[k].cw_orig;
		}
		ctr++;
		assert( ctr <= MAX_POLYGON_SIDES );
		keepGoing = !(k == kstart);
	}
}

std::string VoronoiMesh::infoString(const bool& verbose) const {
    std::ostringstream ss;
    return ss.str();
}

}}
