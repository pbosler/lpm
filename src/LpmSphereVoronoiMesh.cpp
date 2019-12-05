#include "LpmSphereVoronoiMesh.hpp"
#include <cassert>

namespace Lpm {
namespace Voronoi {

template <typename SeedType>
void VoronoiMesh<SeedType>::getEdgesAndCellsAtVertex(std::vector<Index>& edge_inds, std::vector<Index>& cell_inds,
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

template <typename SeedType>
void VoronoiMesh<SeedType>::getEdgesAndVerticesInCell(std::vector<Index>& edge_inds, std::vector<Index>& vert_inds,
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

template <typename SeedType>
std::string VoronoiMesh<SeedType>::infoString(const bool& verbose) const {
    std::ostringstream ss;
    ss << "Voronoi Mesh info:\n";
    ss << "\tnverts = " << vertices.size() << "\n";
    ss << "\tnedges = " << edges.size() << "\n";
    ss << "\tnfaces = " << cells.size() << "\n";
    if (verbose) {
        for (Index i=0; i<vertices.size(); ++i) {
            ss << vertices[i].infoString(1, i);
        }
        for (Index i=0; i<edges.size(); ++i) {
            ss << edges[i].infoString(1, i);
        }
        for (Index i=0; i<cells.size(); ++i) {
            ss << cells[i].infoString(1, i);
        }
    }
    return ss.str();
}

template <typename SeedType>
VoronoiMesh<SeedType>::VoronoiMesh(const MeshSeed<SeedType>& seed, const Short& init_refinement_level) {
    Index nmax_verts, nmax_edges, nmax_faces;
    seed.setMaxAllocations(nmax_verts, nmax_edges, nmax_faces, init_refinement_level);

    vertices.reserve(nmax_verts);
    edges.reserve(nmax_edges);
    cells.reserve(nmax_faces);

    seedInit(seed);
}

template <typename SeedType>
void VoronoiMesh<SeedType>::seedInit(const MeshSeed<SeedType>& seed) {
    // initialize cells
    const Short face_offset = SeedType::nfaces;
    for (Short i=0; i<SeedType::nfaces; ++i) {
        Real xyz[3] = {seed.scrds(i,0), seed.scrds(i,1), seed.scrds(i,2)};
        cells.push_back(Cell(xyz, seed.sfaceedges(i,0)));
    }
    // initialize vertices
    std::vector<Short> edges_at_vertex(SeedType::nverts, -1);
    for (Short i=0; i<SeedType::nedges; ++i) {
        if (edges_at_vertex[seed.sedges(i,0)-face_offset] == -1) {
            edges_at_vertex[seed.sedges(i,0)-face_offset] = i;
        }
        if (edges_at_vertex[seed.sedges(i,1)-face_offset] == -1) {
            edges_at_vertex[seed.sedges(i,1)-face_offset] = i;
        }
    }

    for (Short i=0; i<SeedType::nverts; ++i) {
        Real xyz[3] = {seed.scrds(i+face_offset, 0), seed.scrds(i+face_offset, 1), seed.scrds(i+face_offset, 2)};
        vertices.push_back(Vertex(xyz, edges_at_vertex[i]));
    }
    for (Short i=0; i<SeedType::nedges; ++i) {
        edges.push_back(WingedEdge(seed.sedges(i,0)-face_offset, seed.sedges(i,1)-face_offset, seed.sedges(i,2), seed.sedges(i,3),
            seed.sedges(i,4), seed.sedges(i,5), seed.sedges(i,6), seed.sedges(i,7), vertices));
    }
}

template struct VoronoiMesh<IcosTriDualSeed>;

}}
