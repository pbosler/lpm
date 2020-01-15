#include "LpmSphereVoronoiMesh.hpp"
#include "LpmGeometry.hpp"
#include "LpmVtkIO.hpp"
#include <cassert>
#include <algorithm>
#include <deque>
#include <map>
#include <sstream>
#include <exception>
#include <cmath>
namespace Lpm {
namespace Voronoi {

#define TEST_FILE_ROOT "voronoi_mesh"

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

#ifdef LPM_ENABLE_DEBUG
		if (!(vert_ind == edges[k].orig_vertex || vert_ind == edges[k].dest_vertex)) {
		    std::ostringstream ss;
		    ss << __FILE__ << ": " << __LINE__ << "\n";
		    ss << "VoronoiMesh::getEdgesAndCellsAtVertex error: edge " << k << " does not connect to vertex " << vert_ind << "\n";
		    ss << "\tedge[" << k << "]: " << edges[k].infoString();
		    ss << "\tvertex[" << vert_ind << "]: " << vertices[vert_ind].infoString();
		    throw std::logic_error(ss.str());
		}
#endif
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

	assert(edge_inds.size()==3);
}

template <typename SeedType>
std::vector<Index> VoronoiMesh<SeedType>::getAdjacentCells(const Index& cell_ind) const {
    std::vector<Index> vinds;
    std::vector<Index> einds;
    std::vector<Index> result;
    getEdgesAndVerticesInCell(einds, vinds, cell_ind);
    for (Short i=0; i<einds.size(); ++i) {
        if (cell_ind == edges[einds[i]].left_cell) {
            result.push_back(edges[einds[i]].right_cell);
        }
        else {
            result.push_back(edges[einds[i]].left_cell);
        }
    }
    return result;
}

template <typename SeedType>
Index VoronoiMesh<SeedType>::cellContainingPoint(const Real* xyz, const Index start) const {
    const std::vector<Index> adj_cells = getAdjacentCells(start);
//     Real dist = SphereGeometry::sqEuclideanDistance(xyz, cells[start].xyz);
    Real dist = SphereGeometry::distance(xyz, cells[start].xyz);
    Index next_index = start;
    for (Short i=0; i<adj_cells.size(); ++i) {
//         const Real testdist = SphereGeometry::sqEuclideanDistance(xyz, cells[adj_cells[i]].xyz);
        const Real testdist = SphereGeometry::distance(xyz, cells[adj_cells[i]].xyz);
        if (testdist < dist) {
            dist = testdist;
            next_index = adj_cells[i];
        }
    }
    if (next_index == start) {
        return next_index;
    }
    else {
        return cellContainingPoint(xyz, next_index);
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

#ifdef LPM_ENABLE_DEBUG
		if (!(cell_ind == edges[k].left_cell || cell_ind == edges[k].right_cell)) {
		    std::ostringstream ss;
		    ss << __FILE__ << ": " << __LINE__ << "\n";
		    ss << "VoronoiMesh::getEdgesAndVerticesInCell error: edge " << k << " does not connect to cell " << cell_ind << "\n";
		    ss << "\tedge[" << k << "]: " << edges[k].infoString();
		    ss << "\tcell[" << cell_ind << "]: " << cells[cell_ind].infoString();
		    throw std::logic_error(ss.str());
		}
#endif
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
    ss << "\tsurfarea = " << surfarea << "\n";
    ss << "\tavg mesh size (degrees) " << std::sqrt(4*PI/cells.size())*RAD2DEG << "\n";
    if (verbose) {
        for (Index i=0; i<vertices.size(); ++i) {
            ss << vertices[i].infoString(1);
        }
        for (Index i=0; i<edges.size(); ++i) {
            ss << edges[i].infoString(1);
        }
        for (Index i=0; i<cells.size(); ++i) {
            ss << cells[i].infoString(1);
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
    for (Short i=0; i<init_refinement_level; ++i) {
        const Index nverts = vertices.size();
        std::vector<Real[3]> newx(nverts);
        for (Index j=0; j<nverts; ++j) {
            newx[j][0] = vertices[j].xyz[0];
            newx[j][1] = vertices[j].xyz[1];
            newx[j][2] = vertices[j].xyz[2];
        }
        for (Index j=0; j<nverts; ++j) {
//             std::cout << "adding vertex " << j << " to cells.\n";
            insertCellAtPoint(newx[j], j);
        }
        std::cout << "uniform refinement " << i+1 << " done.\n";
    }
    std::cout << infoString(false);
}

template <typename SeedType>
Index VoronoiMesh<SeedType>::nearestCornerToPoint(const Real* xyz, const Index cell_ind) const {
    std::vector<Index> celledges;
    std::vector<Index> cellverts;
    getEdgesAndVerticesInCell(celledges, cellverts, cell_ind);
    Index vert_ind = cellverts[0];
//     Real dist = SphereGeometry::sqEuclideanDistance(xyz, vertices[vert_ind].xyz);
    Real dist = SphereGeometry::distance(xyz, vertices[vert_ind].xyz);
    for (Short i=1; i<cellverts.size(); ++i) {
//         const Real test_dist = SphereGeometry::sqEuclideanDistance(xyz, vertices[cellverts[i]].xyz);
        const Real test_dist = SphereGeometry::distance(xyz, vertices[cellverts[i]].xyz);

        if (test_dist < dist) {
            dist = test_dist;
            vert_ind = cellverts[i];
        }
    }
    return vert_ind;
}

template <typename SeedType>
Real VoronoiMesh<SeedType>::cellArea(const Index& cell_ind) {
    std::vector<Index> cedges;
    std::vector<Index> cverts;
    getEdgesAndVerticesInCell(cedges, cverts, cell_ind);

    const Int nverts = cverts.size();
    ko::View<Real*[3],Host> vertxyz("vertex_xyz", nverts);
    for (Int i=0; i<nverts; ++i) {
        const Index v = cverts[i];
        for (Int j=0; j<3; ++j) {
            vertxyz(i,j) = vertices[v].xyz[j];
        }
    }
    Real bc[3];
    SphereGeometry::barycenter(bc, vertxyz, nverts);
    Real result = 0;
//     const Real result = SphereGeometry::polygonArea(bc, vertxyz, nverts);
    for (Int i=0; i<nverts; ++i) {
        const auto bview = ko::subview(vertxyz, i, ko::ALL());
        const auto cview = ko::subview(vertxyz, (i+1)%nverts, ko::ALL());
        if (SphereGeometry::distance(bview, cview) > ZERO_TOL) {
            result += SphereGeometry::triArea(bc, bview, cview);
        }
    }

#ifdef LPM_ENABLE_DEBUG
    if (std::isnan(result)) {
        std::ostringstream ss;
        ss << __FILE__ << ": " << __LINE__ << "\n";
        ss << "VoronoiMesh::cellArea error: nan at cell " << cells[cell_ind].infoString();
        ss << "cell_edges = ";
        for (const auto& ce : cedges) {
            ss << "\t" << edges[ce].infoString();
        }
        ss << "cell_verts = ";
        for (const auto& cv : cverts) {
            ss << "\t" << vertices[cv].infoString();
        }
        ss << "vertxyz = ";
        for (Int i=0; i<nverts; ++i) {
            for (Short j=0; j<3; ++j) {
                ss << ( (j==0 && i>0) ? "          ": "") << vertxyz(i,j) << (j<2 ? " " : "\n");
            }
        }
        ss << "barycenter = (";
        for (Short i=0; i<3; ++i) {
            ss << bc[i] << (i<2 ? " " : ")\n");
        }
        std::cout << ss.str();
//         throw std::logic_error(ss.str());
    }
#endif

    cells[cell_ind].area = result;

    return result;
}

template <typename SeedType>
void VoronoiMesh<SeedType>::updateSurfArea() {
    Real sa = 0;
    for (Index i=0; i<cells.size(); ++i) {
        sa += cellArea(i);
    }
    surfarea = sa;
}

template <typename SeedType>
std::vector<Index> VoronoiMesh<SeedType>::buildBrokenCornerList(const Real* xyz, const Index& first_bc, const bool& verbose) const {
    std::vector<Index> result;
    result.push_back(first_bc);
    std::deque<Index> vertex_que;
    std::vector<Index> vertedges;
    std::vector<Index> vertcells;
    getEdgesAndCellsAtVertex(vertedges, vertcells, first_bc);
    for (const auto& e : vertedges) {
        const Index opposite_vert = (edges[e].orig_vertex == first_bc ? edges[e].dest_vertex : edges[e].orig_vertex);
        vertex_que.push_back(opposite_vert);
    }
    Index pass_ctr = 0;
    std::set<Index> tested_verts;
    tested_verts.insert(first_bc);

#ifdef LPM_ENABLE_DEBUG
    std::ostringstream ss;
    ss << __FILE__ << " " << __LINE__ << "\n";
    ss << "VoronoiMesh::buildBrokenCornerList \n";
    ss << "\txyz = (" << xyz[0] << " " << xyz[1] << " " << xyz[2] << ")\n";
    ss << "\tfirst_bc = " << first_bc << "\n";
    ss << "\tbc test " << SphereGeometry::sqEuclideanDistance(xyz, vertices[first_bc].xyz) - vertices[first_bc].circumradius << "\n";
    ss << "\tbc gc test " << SphereGeometry::distance(xyz, vertices[first_bc].xyz) - vertices[first_bc].circumradius << "\n";
#endif


    while (!vertex_que.empty()) {
        const Index vind = vertex_que.front();
        vertex_que.pop_front();
        tested_verts.insert(vind);
        if (brokenCorner(xyz, vind)) {
            result.push_back(vind);

            getEdgesAndCellsAtVertex(vertedges, vertcells, vind);
            for (const auto& e : vertedges) {

                assert(edges[e].orig_vertex == vind || edges[e].dest_vertex == vind);

                const Index opposite_vert = (edges[e].orig_vertex == vind ? edges[e].dest_vertex : edges[e].orig_vertex);
                const bool is_new = tested_verts.find(opposite_vert) == tested_verts.end();
                if (is_new) {
                    vertex_que.push_back(opposite_vert);
                }
            }
        }
        ++pass_ctr;

#ifdef LPM_ENABLE_DEBUG
        ss << "\tend of pass " << pass_ctr << "\n";

        ss << "\t\tvertex_que = ";
        for (const auto& v : vertex_que) {
            ss << v << " ";
        }
        ss << "\n";

        ss << "\t\tbroken_corners (result) = ";
        for (const auto& bc : result) {
            ss << bc << " ";
        }
        ss << "\n";
#endif
        assert(pass_ctr <= 100);
    }
#ifdef LPM_ENABLE_DEBUG
    if (result.empty()) {
        std::cout << ss.str();
    }
#endif

    return result;
}

template <typename SeedType>
void VoronoiMesh<SeedType>::pointInsertionBookkeeping(std::vector<Index>& edges_to_delete, std::vector<Index>& edges_to_update_ccw,
    std::vector<Index>& new_vert_inds, std::vector<CCWVertexGenerators>& gens, std::vector<Index>& new_edge_inds, const std::vector<Index>& broken_corners) {

    // collect all edges connected to a broken corner
    // separate edges that have both vertices broken (these will be deleted)
    // from those that only have one vertex broken (these will create new corners)
#ifdef LPM_ENABLE_DEBUG
    Int nerr = 0;
    std::ostringstream ss;
#endif
    const Index bc0 = *(broken_corners.begin());
    const Index e0 = vertices[bc0].edgeAtVertex;
    bool keepGoing = true;
    Index bc = bc0;
    Index e = e0;
    Index next_bc = -1;
    Index next_e = -1;

    std::set<EdgeCellIndexPair> tested_pairs;
    std::set<Index> tested_edges;
    edges_to_delete.clear();
    edges_to_update_ccw.clear();
    new_vert_inds.clear();
    gens.clear();
    new_edge_inds.clear();

//     std::cout << "DEBUG: VoronoiMesh::pointInsertionBookkeeping\n";
//     std::cout << "\tbroken_corners = ";
//     for (const auto& bc : broken_corners) {
//         std::cout << bc << " ";
//     }
//     std::cout << "\n";

    while (keepGoing) {
        const bool inbound = (bc == edges[e].dest_vertex);
        const auto working_pair = EdgeCellIndexPair(e, (inbound ? edges[e].left_cell : edges[e].right_cell));
        if (tested_pairs.find(working_pair) == tested_pairs.end()) {
            const Index opposite_vert = (inbound ? edges[e].orig_vertex : edges[e].dest_vertex);
            const bool delete_e = (std::find(broken_corners.begin(), broken_corners.end(), opposite_vert) != broken_corners.end());
            if (delete_e) {
                if (tested_edges.find(e) == tested_edges.end()) {
                    edges_to_delete.push_back(e);
//                     std::cout << "\tadding edge " << e << " to delete list.\n";
                }
                else {
//                     std::cout << "\tedge " << e << " is already marked for deletion.\n";
                }
                next_e = (inbound ? edges[e].ccw_orig : edges[e].ccw_dest);
                next_bc = opposite_vert;
            }
            else {
                if (tested_edges.find(e) == tested_edges.end()) {
                    edges_to_update_ccw.push_back(e);
                    const Index cb = (inbound ? edges[e].left_cell : edges[e].right_cell);
                    const Index cc = (inbound ? edges[e].right_cell : edges[e].left_cell);
                    gens.push_back(CCWVertexGenerators(cells.size(), cb, cc));
//                     std::cout << "\tadding edge " << e << " to update list.\n";
                }
                else {
//                     std::cout << "\tedge " << e << " has already been tested.\n";
                }
                next_e = (inbound ? edges[e].ccw_dest : edges[e].ccw_orig);
                next_bc = bc;
            }
            tested_edges.insert(e);
            tested_pairs.insert(working_pair);
//             std::cout << "\t\tnext_edge = " << next_e << " next bc = " << next_bc << "\n";
            bc = next_bc;
            e = next_e;
        }
        else {
            keepGoing = false;
        }
    }

//     std::cout << "\tdone with while loop.\n";

    const Int nnewedges = edges_to_update_ccw.size();
    const Int nnewverts = nnewedges;
    new_vert_inds = std::vector<Index>(nnewverts, -1);
    new_edge_inds = std::vector<Index>(nnewedges, -1);
    Int j=0;
    for (const auto& e : edges_to_delete) {
        Index replacement_edge = -1;
        std::vector<Index> cedges;
        std::vector<Index> cverts;
        const Index lc = edges[e].left_cell;
        const Index rc = edges[e].right_cell;
        if (cells[lc].edgeAroundCell == e) {
            getEdgesAndVerticesInCell(cedges, cverts, lc);
            for (Int i=0; i<cedges.size(); ++i) {
                const Index ee = cedges[i];
                const bool ee_in_delete_list = (std::find(edges_to_delete.begin(), edges_to_delete.end(), ee) !=
                    edges_to_delete.end());
                const bool ee_in_update_list = (std::find(edges_to_update_ccw.begin(), edges_to_update_ccw.end(), ee) !=
                    edges_to_update_ccw.end());
//                 const bool use_ee = ((!ee_in_delete_list) && (!ee_in_update_list));
                const bool use_ee = !ee_in_delete_list;
                if (use_ee) {
                    replacement_edge = ee;
                    break;
                }
            }
#ifdef LPM_ENABLE_DEBUG
            if (replacement_edge < 0) {
                ++nerr;
                ss << __FILE__ << ": " << __LINE__ << '\n';
                ss << "VoronoiMesh::pointInsertionBookkeeping error: replacement edge not found (left)\n";
                ss << "\t" << cells[lc].infoString();
                ss << "\t" << edges[e].infoString();
                ss << "\t cedges around cell " << lc << ":\n";
                for (const auto& c : cedges) {
                    ss << "\t\t" << edges[c].infoString();
                }
            }
#endif
//             assert(replacement_edge >= 0);
            cells[lc].edgeAroundCell = replacement_edge;
        }
        replacement_edge = -1;
        if (cells[rc].edgeAroundCell == e) {
            getEdgesAndVerticesInCell(cedges, cverts, rc);
            for (Int i=0; i<cedges.size(); ++i) {
                const Index ee = cedges[i];
                const bool ee_in_delete_list = (std::find(edges_to_delete.begin(), edges_to_delete.end(), ee) !=
                    edges_to_delete.end());
                const bool ee_in_update_list = (std::find(edges_to_update_ccw.begin(), edges_to_update_ccw.end(), ee) !=
                    edges_to_update_ccw.end());
//                 const bool use_ee ((!ee_in_delete_list) && (!ee_in_update_list));
                const bool use_ee = !ee_in_delete_list;
                if (use_ee) {
                    replacement_edge = ee;
                    break;
                }
            }
#ifdef LPM_ENABLE_DEBUG
            if (replacement_edge < 0) {
                ++nerr;
                ss << __FILE__ << ": " << __LINE__ << '\n';
                ss << "VoronoiMesh::pointInsertionBookkeeping error: replacement edge not found (right)\n";
                ss << "\t" << cells[rc].infoString();
                ss << "\t" << edges[e].infoString();
                ss << "\t cedges around cell " << rc << ":\n";
                for (const auto& c : cedges) {
                    ss << "\t\t" << edges[c].infoString();
                }
            }
#endif
//             assert(replacement_edge >= 0);
            cells[rc].edgeAroundCell = replacement_edge;
        }
        new_edge_inds[j++] = e;
    }
    for (Int i=j; i<nnewedges; ++i) {
        edges.push_back(WingedEdge());
        new_edge_inds[i] = edges.size()-1;
    }
    j=0;
    for (const auto& bc : broken_corners) {
        new_vert_inds[j++] = bc;
    }
    for (Int i=j; i<nnewverts; ++i) {
        vertices.push_back(Vertex());
        new_vert_inds[i] = vertices.size()-1;
    }

#ifdef LPM_ENABLE_DEBUG
    for (Int i=0; i<new_edge_inds.size(); ++i) {
        if (new_edge_inds[i] < 0) {
            ++nerr;
            ss << __FILE__ << ": " << __LINE__ << "\n";
            ss << "VoronoiMesh::pointInsertionBookkeeping error: invalid edge id\n";
            break;
        }
    }
    for (Int i=0; i<new_vert_inds.size(); ++i) {
        if (new_vert_inds[i] < 0) {
            ++nerr;
            ss << __FILE__ << ": " << __LINE__ << "\n";
            ss << "VoronoiMesh::pointInsertionBookkeeping error: invalid vertex id\n";
        }
    }


    if (nerr > 0) {
        ss << "\tbroken_corners:\n";
        for (const auto& bc : broken_corners) {
            ss << "\t\t" << bc << " " << vertices[bc].infoString();
        }
        ss << "\tedges_to_delete:\n";
        for (const auto& e: edges_to_delete) {
            ss << "\t\t" << edges[e].infoString();
        }
        ss << "\tedges_to_update_ccw:\n";
        for (const auto& e : edges_to_update_ccw) {
            ss << "\t\t" << edges[e].infoString();
        }
        ss << "\tnew_vert_inds:";
        for (const auto& v : new_vert_inds) {
            ss << v << " ";
        }
        ss << "\n";
        ss << "\tnew_edge_inds: ";
        for (const auto& e : new_edge_inds) {
            ss << e << " ";
        }
        ss << "\n";

        throw std::logic_error(ss.str());
    }
#endif

//
//     std::cout << "\tedges_to_delete: \n";
//     for (const auto& e :edges_to_delete) {
//         std::cout << "\t\t" << edges[e].infoString();
//     }
//     std::cout << "\tedges_to_update_ccw:\n";
//     for (const auto& e : edges_to_update_ccw) {
//         std::cout << "\t\t"  << edges[e].infoString();
//     }
//     std::cout << "\tnew_vert_inds: ";
//     for (const auto& v : new_vert_inds) {
//         std::cout << v << " ";
//     }
//     std::cout << "\n";
//     std::cout << "\tnew_edge_inds: ";
//     for (const auto& e : new_edge_inds) {
//         std::cout << e << " ";
//     }
//     std::cout << "\n";

}

template <typename SeedType>
void VoronoiMesh<SeedType>::makeNewVertices(std::vector<Vertex>& newverts, const std::vector<CCWVertexGenerators>& gens,
    const Real* xyz, const std::vector<Index>& edges_to_update_ccw) const {
    newverts.clear();
    const Index newcellind = cells.size();
    for (Int i=0; i<gens.size(); ++i) {
        Real newxyz[3];
        SphereGeometry::circumcenter(newxyz, xyz, cells[gens[i].cell_b].xyz, cells[gens[i].cell_c].xyz);
        Vertex newvert(newxyz, edges_to_update_ccw[i]);
        newvert.circumradius = SphereGeometry::sqEuclideanDistance(xyz, newxyz);
//         newvert.circumradius = SphereGeometry::distance(xyz, newxyz);
        newverts.push_back(newvert);
    }
}



/// Algorithm 4.6.1 from Okabe et. al.
template <typename SeedType>
void VoronoiMesh<SeedType>::insertCellAtPoint(const Real* xyz, const Index& first_guess_cell_ind) {

    const Index newcellind = cells.size();

    // Step 1
    const Index ptloc = cellContainingPoint(xyz, first_guess_cell_ind);

    // Step 2
    const Index vert_ind = nearestCornerToPoint(xyz, ptloc);

//     Step 3
    const std::vector<Index> broken_corners = buildBrokenCornerList(xyz, vert_ind); // Okabe's "set T"

#ifdef LPM_ENABLE_DEBUG
    if (broken_corners.empty()) {
        std::ostringstream ss;
        ss << __FILE__ << ": " << __LINE__ << "\n";
        ss << "VoronoiMesh::insertCellAtPoint error: broken corners list is empty.\n";
        ss << "\tadding point: (" << xyz[0] << " " << xyz[1] << " " << xyz[2] << ")\n";
        ss << "\tcell containg point (ptloc) = " << ptloc << " first guess ind = " << first_guess_cell_ind << "\n";
        ss << cells[ptloc].infoString();
        ss << "\tnearest corner (vert_ind): " << vert_ind << "\n";
        ss << vertices[vert_ind].infoString();

        std::cout << ss.str();

        throw std::logic_error("Empty broken corners list.");
    }
#endif

//     bookkeeping
    std::vector<Index> edges_to_delete;
    std::vector<Index> edges_to_update_ccw;
    std::vector<Index> new_vert_inds;
    std::vector<Index> new_edge_inds;
    std::vector<CCWVertexGenerators> gens;
    pointInsertionBookkeeping(edges_to_delete, edges_to_update_ccw, new_vert_inds, gens, new_edge_inds, broken_corners);

//     Step 4
    std::vector<Vertex> newverts;
    makeNewVertices(newverts, gens, xyz, edges_to_update_ccw);
    for (Int i=0; i<newverts.size(); ++i) {
        newverts[i].id = new_vert_inds[i];
        vertices[new_vert_inds[i]] = newverts[i];
    }

    const Int nnewverts = edges_to_update_ccw.size();

//     Step 5
    for (Int i=0; i<edges_to_update_ccw.size(); ++i) {
        const Index e = edges_to_update_ccw[i];
        const bool inbound = gens[i].cell_b == edges[e].left_cell;
        const Index orig = new_vert_inds[i];
        const Index dest = new_vert_inds[(i+1)%nnewverts];
        const Index left = newcellind;
        const Index right = gens[i].cell_c;
        const Index cwo = e;
        const Index ccwo = new_edge_inds[(i+nnewverts-1)%nnewverts];
        const Index cwd = new_edge_inds[(i+1)%nnewverts];
        const Index ccwd = edges_to_update_ccw[(i+1)%nnewverts];

        if (inbound) {
            edges[e].cw_dest = new_edge_inds[(i+nnewverts-1)%nnewverts];
            edges[e].ccw_dest = new_edge_inds[i];
            edges[e].dest_vertex = new_vert_inds[i];
        }
        else {
            edges[e].cw_orig = new_edge_inds[(i+nnewverts-1)%nnewverts];
            edges[e].ccw_orig = new_edge_inds[i];
            edges[e].orig_vertex = new_vert_inds[i];
        }

        WingedEdge newedge(orig, dest, left, right, cwo, ccwo, cwd, ccwd, vertices);
        newedge.id = new_edge_inds[i];
        edges[new_edge_inds[i]] = newedge;
    }

    assert(new_edge_inds[0] >= 0);

    auto newcell = Cell(xyz, new_edge_inds[0]);
    newcell.id = newcellind;
    cells.push_back(newcell);
    newcell.area = cellArea(newcell.id);
    updateSurfArea();


#ifdef LPM_ENABLE_DEBUG
    VtkInterface<SeedType> vtk;
    auto pd = vtk.toVtkPolyData(*this);
    std::ostringstream ss;
    ss << TEST_FILE_ROOT << cells.size() << ".vtk";
    vtk.writePolyData(ss.str(), pd);

    if (std::isnan(surfarea) || std::abs(surfarea - 4*PI) > 5*ZERO_TOL) {
        std::ostringstream ss;
        ss << __FILE__ << ": " << __LINE__ << "\n";
        ss << "VoronoiMesh::insertCellAtPoint error: Surface area test failed: surfarea = " << surfarea << " abs(err) = " << std::abs(surfarea-4*PI) << "\n";
        ss << "\tnewcellind = " << newcell.id << "\n";
        ss << "\tpt = (" << xyz[0] << " " << xyz[1] << " " << xyz[2] << ") found in cell " << ptloc << " closest corner " << vert_ind << "\n";
        ss << "\tbroken_corners = ";
        for (auto& bc : broken_corners) {
            ss << bc << " ";
        }
        ss << "\n";
        ss << "\tedges_to_delete = ";
        for (auto& de : edges_to_delete){
            ss << de << " ";
        }
        ss << "\n";
        ss << "\tedges_to_update_ccw = ";
        for (auto& eu : edges_to_update_ccw) {
            ss << eu << " ";
        }
        ss << "\n";
        ss << "\tnew_vert_inds = ";
        for (auto& vi : new_vert_inds) {
            ss << vi << " ";
        }
        ss << "\n";
        ss << "\tnew_edge_inds = ";
        for (auto ei : new_edge_inds) {
            ss << ei << " ";
        }
        ss << "\n";
        ss << "\tnewverts.size() = " << newverts.size() << " nverts = " << vertices.size() << "\n";
        for (Int i=0; i<newverts.size(); ++i) {
            ss << "\t\tnewverts[" << i << "]: gens " << gens[i] << newverts[i].infoString();
        }
        throw std::logic_error(ss.str());
    }


//     bool print_verbose = false;
//     if (vertices.size() > 72) {
//         print_verbose = true;
//     }
//     std::cout << infoString(print_verbose);
#endif
}

template <typename SeedType>
void VoronoiMesh<SeedType>::seedInit(const MeshSeed<SeedType>& seed) {
    // initialize cells
    const Short face_offset = SeedType::nfaces;
    for (Short i=0; i<SeedType::nfaces; ++i) {
        Real xyz[3] = {seed.scrds(i,0), seed.scrds(i,1), seed.scrds(i,2)};
        auto c = Cell(xyz,seed.sfaceedges(i,0));
        c.id = i;
        cells.push_back(c);
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

    // initialize edges
    for (Short i=0; i<SeedType::nedges; ++i) {
        edges.push_back(WingedEdge(seed.sedges(i,0)-face_offset, seed.sedges(i,1)-face_offset, seed.sedges(i,2), seed.sedges(i,3),
            seed.sedges(i,4), seed.sedges(i,5), seed.sedges(i,6), seed.sedges(i,7), vertices));
    }

    // initialize mesh variables
    //  circumradii for vertices
    for (Short i=0; i<SeedType::nverts; ++i ) {
        const Index edge_ind = vertices[i].edgeAtVertex;
        const Index cell_ind = edges[edge_ind].left_cell;
        vertices[i].circumradius = SphereGeometry::sqEuclideanDistance(vertices[i].xyz, cells[cell_ind].xyz);
//         vertices[i].circumradius = SphereGeometry::distance(vertices[i].xyz, cells[cell_ind].xyz);
        vertices[i].id = i;
    }
    surfarea = 0.0;
    for (Short i=0; i<cells.size(); ++i) {
        surfarea += cellArea(i);
    }

    for (Short i=0; i<edges.size(); ++i) {
        edges[i].setLength(vertices);
    }

    cell_flags = std::vector<bool>(cells.size(), false);
    vertex_flags = std::vector<bool>(vertices.size(), false);

#ifdef LPM_ENABLE_DEBUG
//     VtkInterface<SeedType> vtk;
//     auto pd = vtk.toVtkPolyData(*this);
//     std::ostringstream ss;
//     ss << TEST_FILE_ROOT << cells.size() << ".vtk";
//     vtk.writePolyData(ss.str(), pd);
//     std::cout << infoString(true);
#endif
}

template <typename SeedType>
bool VoronoiMesh<SeedType>::brokenCorner(const Real* newxyz, const Index& vert_ind) const {
    return SphereGeometry::sqEuclideanDistance(newxyz, vertices[vert_ind].xyz) - vertices[vert_ind].circumradius < 0; //ZERO_TOL;
//     return SphereGeometry::distance(newxyz, vertices[vert_ind].xyz) - vertices[vert_ind].circumradius < 0; //ZERO_TOL;
}


template struct VoronoiMesh<IcosTriDualSeed>;

}}
