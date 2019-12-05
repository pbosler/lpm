import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import codecs

indexBase = 0

def bilinearPlaneMap( v0, v1, v2, v3, r1, r2 ):
    result = np.zeros(2)
    result[0] = 0.25 * ( (1.0-r1)*(1.0+r2)*v0[0] + (1.0-r1)*(1.0-r2)*v1[0] +
        (1.0+r1)*(1.0-r2)*v2[0] + (1.0+r1)*(1.0+r2)*v3[0])
    result[1] = 0.25 * ( (1.0-r1)*(1.0+r2)*v0[1] + (1.0-r1)*(1.0-r2)*v1[1] +
        (1.0+r1)*(1.0-r2)*v2[1] + (1.0+r1)*(1.0+r2)*v3[1])
    return result

def quadRectSeed():
    xyz = np.zeros([13,2])
    xyz[0] = (-1.0, 1.0 )
    xyz[1] = (-1.0, 0.0 )
    xyz[2] = (-1.0,-1.0 )
    xyz[3] = (0.0,-1.0 )
    xyz[4] = (1.0,-1.0 )
    xyz[5] = (1.0, 0.0 )
    xyz[6] = (1.0, 1.0 )
    xyz[7] = (0.0, 1.0 )
    xyz[8] = (0.0, 0.0 )
    xyz[9] = (-0.5, 0.5 )
    xyz[10] = (-0.5,-0.5 )
    xyz[11] = (0.5,-0.5 )
    xyz[12] = (0.5, 0.5)
    edgeOrigs = np.array([ 0, 1,2,3,4,5,6,7,1,8,3,8],dtype=int)
    edgeDests = np.array([ 1, 2,3,4,5,6,7,0,8,5,8,7],dtype=int)
    edgeLefts = np.array([ 0, 1, 1, 2, 2, 3, 3, 0,0,3,1,0],dtype=int)
    edgeRights= np.array([-1,-1,-1,-1,-1,-1,-1,-1,1,2,2,3], dtype=int)
    faceVerts = np.array([[0,1,8,7],[1,2,3,8],[8,3,4,5],[7,8,5,6]],dtype=int)
    faceEdges = np.array([[0,8,11,7],[1,2,10,8],[10,3,4,9],[11,9,5,6]],dtype=int)
    faceCenters = np.array([9,10,11,12],dtype=int)
    edgeInteriors = None
    return xyz, edgeOrigs, edgeDests, edgeLefts, edgeRights, edgeInteriors, \
        faceVerts, faceCenters, faceEdges


def cubedSphereSeed():
    oor3 = 1.0 / np.sqrt(3.0)
    xyz = np.zeros([14,3])
    xyz[0] = (oor3, -oor3, oor3)
    xyz[1] = (oor3, -oor3, -oor3)
    xyz[2] = (oor3, oor3, -oor3)
    xyz[3] = (oor3, oor3, oor3)
    xyz[4] = (-oor3, oor3, -oor3)
    xyz[5] = (-oor3, oor3, oor3)
    xyz[6] = (-oor3, -oor3, -oor3)
    xyz[7] = (-oor3, -oor3, oor3)
    xyz[8] = (1.0, 0.0, 0.0)
    xyz[9] = (0.0, 1.0, 0.0)
    xyz[10] = (-1.0, 0.0, 0.0)
    xyz[11] = (0.0, -1.0, 0.0)
    xyz[12] = (0.0, 0.0, 1.0)
    xyz[13] = (0.0, 0.0, -1.0)
    edgeOrigs = np.array([0,1,2,3,2,4,5,4,6,7,6,0],dtype=int)
    edgeDests = np.array([1,2,3,0,4,5,3,6,7,5,1,7],dtype=int)
    edgeLefts = np.array([0,0,0,0,1,1,1,2,2,2,3,3],dtype=int)
    edgeRights= np.array([3,5,1,4,5,2,4,5,3,4,5,4],dtype=int)
    edgeInteriors = None
    faceVerts = np.array([[0,1,2,3], [3,2,4,5], [5,4,6,7], [7,6,1,0], [7,0,3,5],[1,6,4,2]],dtype=int)
    faceEdges = np.array([[0,1,2,3], [2,4,5,6], [5,7,8,9], [8,10,0,11], [11,3,6,9],[10,7,4,1]],dtype=int)
    faceCenters = np.array([8,9,10,11,12,13],dtype=int)
    return xyz, edgeOrigs, edgeDests, edgeLefts, edgeRights, edgeInteriors, \
        faceVerts, faceCenters, faceEdges

def sphereTriCenter(v0, v1, v2):
    result = (v0 + v1 + v2)/3.0
    norm = np.sqrt(np.sum(np.square(result)))
    result /= norm
    return result

def icosTriSeed():
    xyz = np.zeros([32,3])
    xyz[0] = (0.0,0.0,1.0)
    xyz[1] = (0.723606797749978969640917366873,0.525731112119133606025669084848,0.447213595499957939281834733746)
    xyz[2] = (-0.276393202250021030359082633126,0.850650808352039932181540497063,0.447213595499957939281834733746)
    xyz[3] = (-0.894427190999915878563669467492,0.0,0.447213595499957939281834733746,)
    xyz[4] = (-0.276393202250021030359082633127,-0.850650808352039932181540497063,0.447213595499957939281834733746,)
    xyz[5] = (0.723606797749978969640917366873,-0.525731112119133606025669084848,0.447213595499957939281834733746,)
    xyz[6] = (0.894427190999915878563669467492,0.0,-0.447213595499957939281834733746,)
    xyz[7] = (0.276393202250021030359082633127,0.850650808352039932181540497063,-0.447213595499957939281834733746,)
    xyz[8] = (-0.723606797749978969640917366873,0.525731112119133606025669084848,-0.447213595499957939281834733746,)
    xyz[9] = (-0.723606797749978969640917366873,-0.525731112119133606025669084848,-0.447213595499957939281834733746,)
    xyz[10] = (0.276393202250021030359082633127,-0.850650808352039932181540497063,-0.447213595499957939281834733746,)
    xyz[11] = (0.0,0.0,-1.0,)
    edgeOrigs = np.array([0,1,2,2,0,3,4,4,5,5,1,6,7,7,7,8,8,8,3,9,9,10,10,10,6,11,11,8,11,10],dtype=int)
    edgeDests = np.array([1,2,0,3,3,4,0,5,0,1,6,7,1,2,8,2,3,9,9,4,10,4,5,6,5,6,7,11,9,11],dtype=int)
    edgeLefts = np.array([0,0,0,1,2,2,2,3,3,4,5,5,5,6,7,7,8,9,10,10,11,11,12,13,13,19,15,17,17,19],dtype=int)
    edgeRights= np.array([4,6,1,8,1,10,3,12,4,14,14,15,6,7,16,8,9,17,9,11,18,12,13,19,14,15,16,16,18,18],dtype=int)
    edgeInteriors = None
    faceVerts = np.zeros([20,3], dtype=int)
    faceVerts[0] = (0,1,2)
    faceVerts[1] = (0,2,3)
    faceVerts[2] = (0,3,4)
    faceVerts[3] = (0,4,5)
    faceVerts[4] = (0,5,1)
    faceVerts[5] = (1,6,7)
    faceVerts[6] = (7,2,1)
    faceVerts[7] = (2,7,8)
    faceVerts[8] = (8,3,2)
    faceVerts[9] = (3,8,9)
    faceVerts[10] = (9,4,3)
    faceVerts[11] = (4,9,10)
    faceVerts[12] = (10,5,4)
    faceVerts[13] = (5,10,6)
    faceVerts[14] = (6,1,5)
    faceVerts[15] = (11,7,6)
    faceVerts[16] = (11,8,7)
    faceVerts[17] = (11,9,8)
    faceVerts[18] = (11,10,9)
    faceVerts[19] = (11,6,10)
    faceEdges = np.zeros([20,3],dtype=int)
    faceEdges[0] = (0,1,2)
    faceEdges[1] = (2,3,4)
    faceEdges[2] = (4,5,6)
    faceEdges[3] = (6,7,8)
    faceEdges[4] = (8,9,0)
    faceEdges[5] = (10,11,12)
    faceEdges[6] = (13,1,12)
    faceEdges[7] = (13,14,15)
    faceEdges[8] = (16,3,15)
    faceEdges[9] = (16,17,18)
    faceEdges[10] = (19,5,18)
    faceEdges[11] = (19,20,21)
    faceEdges[12] = (22,7,21)
    faceEdges[13] = (22,23,24)
    faceEdges[14] = (10,9,24)
    faceEdges[15] = (26,11,25)
    faceEdges[16] = (27,14,26)
    faceEdges[17] = (28,17,27)
    faceEdges[18] = (29,20,28)
    faceEdges[19] = (25,23,29)
    faceCenters = np.array(range(12,32),dtype=int)
    for i in range(20):
        xyz[i+12] = sphereTriCenter(xyz[faceVerts[i][0]], xyz[faceVerts[i][1]], xyz[faceVerts[i][2]])
    return xyz, edgeOrigs, edgeDests, edgeLefts, edgeRights, edgeInteriors, \
        faceVerts, faceCenters, faceEdges

def icosTriDualSeed():
    trixyz, tri_origs, tri_dests, tri_lefts, tri_rights, tri_interiors, tri_face_verts, tri_face_centers, tri_face_edges = icosTriSeed()
    xyz = trixyz;
#                           0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29
    edgeOrigs =   np.array([12, 13, 14, 15, 16, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 18, 20, 22, 24, 26, 17, 19, 21, 23, 25, 27, 28, 29, 30, 31], dtype=int)
    edgeDests =   np.array([13, 14, 15, 16, 12, 18, 20, 22, 24, 26, 17, 19, 21, 23, 25, 19, 21, 23, 25, 17, 27, 28, 29, 30, 31, 28, 29, 30, 31, 27], dtype=int)
    edgeLefts =   np.array([0,  0,  0,  0,  0,  2,  3,  4,  5,  1,  7,  8,  9,  10, 6,  2,  3,  4,  5,  1,  7,  8,  9,  10, 6,  7,  8,  9,  10, 6], dtype=int)
    edgeRights =  np.array([2,  3,  4,  5,  1,  1,  2,  3,  4,  5,  1,  2,  3,  4,  5,  7,  8,  9,  10, 6,  6,  7,  8,  9,  10, 11, 11, 11, 11, 11], dtype=int)
    edgeCwOrig =  np.array([5,  6,  7,  8,  9,  4,  0,  1,  2,  3,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 19, 15, 16, 17, 18, 29, 25, 26, 27, 28], dtype=int)
    edgeCcwOrig = np.array([4,  0,  1,  2,  3,  0,  1,  2,  3,  4,  15, 16, 17, 18, 19,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24], dtype=int)
    edgeCwDest =  np.array([1,  2,  3,  4,  0,  15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 11, 12, 13, 14, 10, 25, 26, 27, 28, 29, 21, 22, 23, 24, 20], dtype=int)
    edgeCcwDest = np.array([6,  7,  8,  9,  5,  10, 11, 12, 13, 14, 19, 15, 16, 17, 18, 21, 22, 23, 24, 20, 29, 25, 26, 27, 28, 26, 27, 28, 29, 25], dtype=int)
    edgeInteriors = None
    faceVerts = np.zeros([12,5],dtype=int)
    faceVerts[0] = (12,13,14,15,16)
    faceVerts[1] = (17,18,12,16,26)
    faceVerts[2] = (12,18,19,20,13)
    faceVerts[3] = (13,20,21,22,14)
    faceVerts[4] = (14,22,23,24,15)
    faceVerts[5] = (15,24,25,26,16)
    faceVerts[6] = (25,31,27,17,26)
    faceVerts[7] = (17,27,28,19,18)
    faceVerts[8] = (19,28,29,21,20)
    faceVerts[9] = (21,29,30,23,22)
    faceVerts[10]= (23,30,31,25,24)
    faceVerts[11]= (27,28,29,30,31)
    faceCenters = np.array(range(12), dtype=int)
    faceEdges = np.zeros([12,5],dtype=int)
    faceEdges[0] = ( 0, 1, 2, 3, 4)
    faceEdges[1] = ( 9,19,10, 5, 4)
    faceEdges[2] = ( 5,15,11, 6, 0)
    faceEdges[3] = ( 6,16,12, 7, 1)
    faceEdges[4] = ( 7,17,13, 8, 2)
    faceEdges[5] = ( 8,18,14, 9, 3)
    faceEdges[6] = (24,29,20,19,14)
    faceEdges[7] = (20,25,21,15,10)
    faceEdges[8] = (21,26,22,16,11)
    faceEdges[9] = (22,27,23,17,12)
    faceEdges[10]= (23,28,24,18,13)
    faceEdges[11]= (25,26,27,28,29)
    return xyz, edgeOrigs, edgeDests, edgeLefts, edgeRights, edgeInteriors, \
        faceVerts, faceCenters, faceEdges, edgeCwOrig, edgeCcwOrig, edgeCwDest, edgeCcwDest

def triHexSeed():
    pio3 = np.pi/3.0
    xyz = np.zeros([13,2])
    xyz[0] = (0.0, 0.0)
    xyz[1] = (np.cos(pio3), np.sin(pio3))
    xyz[2] = (np.cos(2*pio3), np.sin(2*pio3))
    xyz[3] = (-1.0, 0.0)
    xyz[4] = (np.cos(4*pio3), np.sin(4*pio3))
    xyz[5] = (np.cos(5*pio3), np.sin(5*pio3))
    xyz[6] = (1.0, 0.0)
    xyz[7] = (xyz[0] + xyz[1] + xyz[2]) / 3.0
    xyz[8] = (xyz[0] + xyz[2] + xyz[3]) / 3.0
    xyz[9] = (xyz[0] + xyz[3] + xyz[4]) / 3.0
    xyz[10]= (xyz[0] + xyz[4] + xyz[5]) / 3.0
    xyz[11] = (xyz[0] + xyz[5] + xyz[6])/ 3.0
    xyz[12] = (xyz[0] + xyz[6] + xyz[1])/ 3.0
    edgeOrigs = np.array([0,1,2,3,4,5,6,2,3,4,0,0],dtype=int)
    edgeDests = np.array([1,2,3,4,5,6,1,0,0,0,5,6],dtype=int)
    edgeLefts = np.array([0,0,1,2,3,4,5,0,1,2,4,5],dtype=int)
    edgeRights= np.array([5,-1,-1,-1,-1,-1,-1,1,2,3,3,4],dtype=int)
    edgeInteriors = None
    faceVerts = np.array([[0,1,2], [2,3,0],[4,0,3], [0,4,5], [5,6,0], [1,0,6]],dtype=int)
    faceCenters = np.array([7,8,9,10,11,12],dtype=int)
    faceEdges = np.array([[0,1,7],[2,8,7],[9,8,3],[9,4,10],[5,11,10],[0,11,6]],dtype=int)
    return xyz, edgeOrigs, edgeDests, edgeLefts, edgeRights, edgeInteriors, \
        faceVerts, faceCenters, faceEdges

def refQuadCubic():
    r5 = np.sqrt(5.0)
    pts = np.zeros([16,2])
    qps = np.array([-1.0, -1.0/r5, 1.0/r5, 1.0])
    qws = np.array([1.0/6.0, 5.0/6.0, 5.0/6.0, 1.0/6.0])
    for i, qp in enumerate(reversed(qps)):
        pts[i] = (-1.0, qp)
    for i in range(3):
        pts[4+i] = (qps[i+1], -1.0)
        pts[7+i] = (1.0, qps[i+1])
    pts[10] = (qps[2],1.)
    pts[11] = (qps[1],1.)
    pts[12] = (qps[1], qps[2])
    pts[13] = (qps[1], qps[1])
    pts[14] = (qps[2], qps[1])
    pts[15] = (qps[2], qps[2])
    return pts, qps, qws

def quadCubicSeed():
    sqrt5 = np.sqrt(5.0)

    f0verts = np.array(range(12),dtype=int)
    f1verts = np.array([3,16,17,18,19,20,21,22,23,6,5,4],dtype=int)
    f2verts = np.array([6,23,22,21,28,29,30,31,32,33,34,35],dtype=int)
    f3verts = np.array([9,8,7,6,35,34,33,40,41,42,43,44],dtype=int)

    faceVerts = np.array([f0verts, f1verts, f2verts, f3verts], dtype=int)

    f0centers = np.array([12,13,14,15],dtype=int)
    f1centers = np.array([24,25,26,27],dtype=int)
    f2centers = np.array([36,37,38,39],dtype=int)
    f3centers = np.array([45,46,47,48],dtype=int)

    faceCenters = np.array([f0centers, f1centers, f2centers, f3centers], dtype=int)

    faceEdges = np.array([[0,8,11,7], [1,2,10,8], [10,3,4,9], [11,9,5,6]], dtype=int)

    edgeOrigs = np.array([0,3,18,21,30,33,42,9,3,6,21,6],dtype=int)
    edgeDests = np.array([3,18,21,30,33,42,9,0,6,33,6,9],dtype=int)
    edgeInteriors = np.array([[1,2],[16,17], [19,20], [28,29], [31,32], [40,41], [43,44], [10,11],
        [4,5],[35,34],[22,23],[7,8]])
    edgeLefts = np.array([0,1,1,2,2,3,3,0,0,3,1,0],dtype=int)
    edgeRights = np.array([-1,-1,-1,-1,-1,-1,-1,-1,1,2,2,3],dtype=int)

    #rpts, qp, qw = refQuadCubic()

    xyz = np.zeros([49,2])
    xyz[0] = (-1.0, 1.0)
    xyz[3] = (-1.0, 0.0)
    xyz[18] = (-1.0,-1.0)
    xyz[21] = (0.0, -1.0)
    xyz[30] = (1.0, -1.0)
    xyz[33] = (1.0, 0.0)
    xyz[42] = (1.0, 1.0)
    xyz[9] = (0.0, 1.0)
    xyz[6] = (0.0,0.0)

    xyz[1] = bilinearPlaneMap(xyz[0], xyz[3], xyz[6], xyz[9], -1.0, 1.0/sqrt5)
    xyz[2] = bilinearPlaneMap(xyz[0], xyz[3], xyz[6], xyz[9], -1.0, -1.0/sqrt5)
    xyz[4] = bilinearPlaneMap(xyz[0], xyz[3], xyz[6], xyz[9], -1.0/sqrt5, -1.0)
    xyz[5] = bilinearPlaneMap(xyz[0], xyz[3], xyz[6], xyz[9], 1.0/sqrt5, -1.0)
    xyz[7] = bilinearPlaneMap(xyz[0], xyz[3], xyz[6], xyz[9], 1.0, -1.0/sqrt5)
    xyz[8] = bilinearPlaneMap(xyz[0], xyz[3], xyz[6], xyz[9], 1.0, 1.0/sqrt5)
    xyz[10] = bilinearPlaneMap(xyz[0], xyz[3], xyz[6], xyz[9], 1.0/sqrt5, 1.0)
    xyz[11] = bilinearPlaneMap(xyz[0], xyz[3], xyz[6], xyz[9], -1.0/sqrt5, 1.0)
    xyz[12] = bilinearPlaneMap(xyz[0], xyz[3], xyz[6], xyz[9], -1.0/sqrt5, 1.0/sqrt5)
    xyz[13] = bilinearPlaneMap(xyz[0], xyz[3], xyz[6], xyz[9], -1.0/sqrt5, -1.0/sqrt5)
    xyz[14] = bilinearPlaneMap(xyz[0], xyz[3], xyz[6], xyz[9], 1.0/sqrt5, -1.0/sqrt5)
    xyz[15] = bilinearPlaneMap(xyz[0], xyz[3], xyz[6], xyz[9], 1.0/sqrt5, 1.0/sqrt5)

    xyz[16] = bilinearPlaneMap(xyz[3], xyz[18], xyz[21], xyz[6], -1.0, 1.0/sqrt5)
    xyz[17] = bilinearPlaneMap(xyz[3], xyz[18], xyz[21], xyz[6], -1.0, -1.0/sqrt5)
    xyz[19] = bilinearPlaneMap(xyz[3], xyz[18], xyz[21], xyz[6], -1.0/sqrt5, -1.0)
    xyz[20] = bilinearPlaneMap(xyz[3], xyz[18], xyz[21], xyz[6], 1.0/sqrt5, -1.0)
    xyz[22] = bilinearPlaneMap(xyz[3], xyz[18], xyz[21], xyz[6], 1.0, -1.0/sqrt5)
    xyz[23] = bilinearPlaneMap(xyz[3], xyz[18], xyz[21], xyz[6], 1.0, 1.0/sqrt5)
    xyz[24] = bilinearPlaneMap(xyz[3], xyz[18], xyz[21], xyz[6], -1.0/sqrt5, 1.0/sqrt5)
    xyz[25] = bilinearPlaneMap(xyz[3], xyz[18], xyz[21], xyz[6], -1.0/sqrt5, -1.0/sqrt5)
    xyz[26] = bilinearPlaneMap(xyz[3], xyz[18], xyz[21], xyz[6], 1.0/sqrt5, -1.0/sqrt5)
    xyz[27] = bilinearPlaneMap(xyz[3], xyz[18], xyz[21], xyz[6], 1.0/sqrt5, 1.0/sqrt5)

    xyz[28] = bilinearPlaneMap(xyz[6], xyz[21], xyz[30], xyz[33], -1.0/sqrt5, -1.0)
    xyz[29] = bilinearPlaneMap(xyz[6], xyz[21], xyz[30], xyz[33], 1.0/sqrt5, -1.0)
    xyz[31] = bilinearPlaneMap(xyz[6], xyz[21], xyz[30], xyz[33], 1.0, -1.0/sqrt5)
    xyz[32] = bilinearPlaneMap(xyz[6], xyz[21], xyz[30], xyz[33], 1.0, 1.0/sqrt5)
    xyz[34] = bilinearPlaneMap(xyz[6], xyz[21], xyz[30], xyz[33], 1.0/sqrt5, 1.0)
    xyz[35] = bilinearPlaneMap(xyz[6], xyz[21], xyz[30], xyz[33], -1.0/sqrt5, 1.0)
    xyz[36] = bilinearPlaneMap(xyz[6], xyz[21], xyz[30], xyz[33], -1.0/sqrt5, 1.0/sqrt5)
    xyz[37] = bilinearPlaneMap(xyz[6], xyz[21], xyz[30], xyz[33], -1.0/sqrt5, -1.0/sqrt5)
    xyz[38] = bilinearPlaneMap(xyz[6], xyz[21], xyz[30], xyz[33], 1.0/sqrt5, -1.0/sqrt5)
    xyz[39] = bilinearPlaneMap(xyz[6], xyz[21], xyz[30], xyz[33], 1.0/sqrt5, 1.0/sqrt5)

    xyz[40] = bilinearPlaneMap(xyz[9], xyz[6], xyz[33], xyz[42], 1.0, -1.0/sqrt5)
    xyz[41] = bilinearPlaneMap(xyz[9], xyz[6], xyz[33], xyz[42], 1.0, 1.0/sqrt5)
    xyz[43] = bilinearPlaneMap(xyz[9], xyz[6], xyz[33], xyz[42], 1.0/sqrt5, 1.0)
    xyz[44] = bilinearPlaneMap(xyz[9], xyz[6], xyz[33], xyz[42], -1.0/sqrt5, 1.0)
    xyz[45] = bilinearPlaneMap(xyz[9], xyz[6], xyz[33], xyz[42], -1.0/sqrt5, 1.0/sqrt5)
    xyz[46] = bilinearPlaneMap(xyz[9], xyz[6], xyz[33], xyz[42], -1.0/sqrt5, -1.0/sqrt5)
    xyz[47] = bilinearPlaneMap(xyz[9], xyz[6], xyz[33], xyz[42], 1.0/sqrt5, -1.0/sqrt5)
    xyz[48] = bilinearPlaneMap(xyz[9], xyz[6], xyz[33], xyz[42], 1.0/sqrt5, 1.0/sqrt5)

    return xyz, edgeOrigs, edgeDests, edgeLefts, edgeRights, edgeInteriors, \
        faceVerts, faceCenters, faceEdges

def edgeXyz(xyz, orig, dest, ints):
    ptsPerEdge = 2
    if ints is not None:
        ptsPerEdge += len(ints)
    result = np.zeros([ptsPerEdge,2])
    result[0] = xyz[orig]
    if ints is not None:
        for i in range(len(ints)):
            result[i+1] = xyz[ints[i]]
    result[-1] = xyz[dest]
    return result

def writeNamelistFile(fname, xyz, origs, dests, lefts, rights, ints, faceVerts, faceCenters, faceEdges):
    nparticles = len(xyz)
    nedges = len(origs)
    nfaces = len(faceVerts)
    ncenters = len(faceCenters)
    nverts = len(faceVerts[0])
    with open(fname, 'w') as f:
        f.write('&seed\n')
        f.write('xyz = ')
        if np.shape(xyz)[1]==2:
            for i,[x,y] in enumerate(xyz):
                if i<nparticles-1:
                    ws = '%.17f,%.17f,%.17f,'%(x,y,0.0)
                else:
                    ws = '%.17f,%.17f,%.17f\n'%(x,y,0.0)
                f.write(ws)
        else:
            for i,[x,y,z] in enumerate(xyz):
                if i<nparticles-1:
                    ws = '%.17f,%.17f,%.17f,'%(x,y,z)
                else:
                    ws = '%.17f,%.17f,%.17f\n'%(x,y,z)
                f.write(ws)
        f.write('origs = ')
        ws = ','.join(str(o) for o in origs)
        f.write(ws+'\n')

        ws = 'dests = ' + ','.join(str(d) for d in dests) + '\n'
        f.write(ws)
        ws = 'lefts = ' + ','.join(str(l) for l in lefts) + '\n'
        f.write(ws)
        ws = 'rights = '+ ','.join(str(r) for r in rights) + '\n'
        f.write(ws)
        if ints is not None:
            f.write('ints = ')
            nints = len(ints)
            for i, inds in enumerate(ints):
                if i<nints-1:
                    ws = ','.join(str(j) for j in inds) + ','
                else:
                    ws = ','.join(str(j) for j in inds) + '\n'
                f.write(ws)
        f.write('faceverts = ')
        for i, face in enumerate(faceVerts):
            if i<nfaces-1:
                ws = ','.join(str(v) for v in face) + ','
            else:
                ws = ','.join(str(v) for v in face) + '\n'
            f.write(ws)
        f.write('faceedges = ')
        for i, face in enumerate(faceEdges):
            if i < nfaces-1:
                ws = ','.join(str(e) for e in face) + ','
            else:
                ws = ','.join(str(e) for e in face) + '\n'
            f.write(ws)
        f.write('facecenters = ')
        if len(faceCenters.shape) > 1:
            for i, face in enumerate(faceCenters):
                if i < nfaces-1:
                    ws = ','.join(str(c) for c in face) + ','
                else:
                    ws = ','.join(str(c) for c in face) + '\n'
                f.write(ws)
        else:
            ws = ','.join(str(c) for c in faceCenters) + '\n'
            f.write(ws)
        # last line
        f.write('/\n')

def writeSeedFile(fname, xyz, origs, dests, lefts, rights, ints, faceVerts, faceCenters, faceEdges, cwo=None, ccwo=None, cwd=None, ccwd=None):
    nparticles = len(xyz)
    nedges = len(origs)
    nfaces = len(faceVerts)
    ncenters = len(faceCenters)
    nverts = len(faceVerts[0])
    nints = 0
    eformat = '%i   %i  %i  %i  '
    high_order_edges = ints is not None
    winged_edges = cwo is not None and ccwo is not None and cwd is not None and ccwd is not None
    edge_header = "edgeO      edgeD       edgeLeft        edgeRight"
    if high_order_edges:
        nints = len(ints[0])
        edge_header += '    edgeInts'
        for i in range(nints):
            eformat += '%i  '
    if winged_edges:
        edge_header += 'origCw    origCcw    destCw    destCcw'
        for i in range(4):
            eformat += '%i  '
    eformat += '\n'
    edge_header += '\n'
    with open(fname, 'w') as f:
        if np.shape(xyz)[1]==2:
            f.write(("x     y \n"))
            for x, y in xyz:
                f.write(('%.17f  %.17f\n'%(x,y)))
        else:
            f.write(("x   y   z\n"))
            for x, y, z in xyz:
                f.write(('%.17f  %.17f  %.17f\n'%(x,y,z)))
        f.write(edge_header)

        if high_order_edges:
            for i in range(nedges):
                f.write((eformat%(origs[i], dests[i], lefts[i], rights[i], ints[i,0], ints[i,1])))
        elif winged_edges:
            for i in range(nedges):
                f.write((eformat%(origs[i], dests[i], lefts[i], rights[i], cwo[i], ccwo[i], cwd[i], ccwd[i])))
        else:
            for i in range(nedges):
                f.write((eformat%(origs[i], dests[i], lefts[i], rights[i])))

        f.write(("faceverts\n"))
        for v in faceVerts:
            f.write(((str(v)[1:-1]+'\n').lstrip()))

        f.write(('faceedges\n'))
        for e in faceEdges:
            f.write(((str(e)[1:-1]+"\n").lstrip()))
        f.write(('facecenters\n'))
        if ints is None:
            for c in faceCenters:
                f.write((str(c) + '\n'))
        else:
            for c in faceCenters:
                f.write((str(c)[1:-1] + "\n"))

def make_ticklabels_invisible(fig):
    for i, ax in enumerate(fig.axes):
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_visible(False)

def plotSphereSeed(oname, xyz, origs, dests, lefts, rights, ints, faceVerts, faceCenters, faceEdges):
    nparticles = len(xyz)
    nedges = len(origs)
    nfaces = len(faceVerts)
    ncenters = len(faceCenters)
    nverts = len(faceVerts[0])
    nints = 0
    if ints is not None:
        nints = len(ints[0])
    m_size = 4.0
    m_width = m_size / 4.0
    l_width= 2.0

    fig0 = plt.figure()

    gs = GridSpec(2,1)
    ax0 = plt.subplot(gs[0,0])
    ax0.axis('off')
    ax1 = plt.subplot(gs[1,0])
    ax1.axis('off')

    if 'cubed' in oname.lower():
        if 'cubic' in oname.lower():
            pass
        else:
            pxy = np.zeros([20,2])
            pxy[0] = (-0.5, 0.5)
            pxy[1] = (-0.5, -0.5)
            pxy[2] = (0.5, -0.5)
            pxy[3] = (0.5, 0.5)
            pxy[4] = (1.5, -0.5)
            pxy[5] = (1.5, 0.5)
            pxy[6] = (2.5, -0.5)
            pxy[7] = (2.5, 0.5)
            pxy[8] = (-1.5, -0.5) # 6
            pxy[9] = (-1.5, 0.5) # 7
            pxy[10] = (-0.5, 1.5) #7
            pxy[11] = (-0.5, -1.5) #6
            pxy[12] = (0.5, -1.5) #4
            pxy[13] = (0.5, 1.5) #5
            pxy[14] = (0.,0.)
            pxy[15] = (1.0, 0.)
            pxy[16] = (2.0, 0.)
            pxy[17] = (-1.0, 0.)
            pxy[18] = (0.,1.)
            pxy[19] = (0.,-1)
            pinds = np.array([0,1,2,3,4,5,6,7,6,7,7,6,4,5,8,9,10,11,12,13],dtype=int)
            einds =  np.array([0,1,2,3,4,5,6,7,8,9,10,11,8,11, 9,6,10, 7,4],dtype=int)
            pdests = np.array([1,2,3,0,4,5,3,6,7,5, 1, 9,9,10,13,3, 1,11,12],dtype=int)
            porigs = np.array([0,1,2,3,2,4,5,4,6,7, 8, 0,8, 0,10,13,11,12,2],dtype=int)
            for i in range(ncenters):
                ax1.text(pxy[14+i,0], pxy[14+i,1], str(i+indexBase), color='b', bbox=dict(facecolor='b', alpha=0.25))
            ax0.plot(pxy[:,0], pxy[:,1], 'ko', markersize=m_size)
            for i in range(len(pinds)):
                ax0.text(pxy[i,0], pxy[i,1], str(pinds[i]+indexBase), color='k')
            ax0.set(title='edges & particles')
            ax1.set(title='edges & faces')
            for i in range(len(einds)):
                if ints is not None:
                    exy = edgeXyz(pxy, porigs[i], pdests[i], ints[i])
                else:
                    exy = edgeXyz(pxy, porigs[i], pdests[i], None)
                dx = exy[1:,0] - exy[0:-1,0]
                dy = exy[1:,1] - exy[0:-1,1]
                midpt = 0.5 * (exy[0] + exy[-1])
                ax0.arrow(exy[0,0], exy[0,1], midpt[0]-exy[0,0], midpt[1]-exy[0,1], head_width=0.1,
                    head_length=0.05, fc='r', ec='r', length_includes_head=False)
                ax0.plot([midpt[0],exy[-1,0]],[midpt[1],exy[-1,1]],'r-')
                ax0.text(midpt[0], midpt[1]+0.05, str(einds[i] + indexBase), color='r')
                ax1.arrow(exy[0,0], exy[0,1], midpt[0]-exy[0,0], midpt[1]-exy[0,1], head_width=0.1,
                    head_length=0.05, fc='r', ec='r', length_includes_head=False)
                ax1.plot([midpt[0],exy[-1,0]],[midpt[1],exy[-1,1]],'r-')
                ax1.text(midpt[0], midpt[1]+0.05, str(einds[i]+indexBase), color='r')
    elif 'icos' in oname.lower():
        pio3 = np.pi/3.0
        pxy = np.zeros([42,2])
        pinds = np.zeros(42,dtype=int)
        #1st row
        y = np.sin(pio3)
        pxy[0] = (np.cos(2*pio3), y)
        pxy[1] = (np.cos(pio3), y)
        pxy[2] = (1+np.cos(pio3), y)
        pxy[3] = (2+np.cos(pio3), y)
        pxy[4] = (3+np.cos(pio3), y)
        pinds[0:5] = 0
        #2nd row)
        pxy[5] = (-1.,0.) #1)
        pxy[6] = (0.,0.) #2)
        pxy[7] = (1.,0.) #3)
        pxy[8] = (2.,0.)#4)
        pxy[9] = (3.,0.)#5)
        pxy[10]= (4.,0)#1)
        pinds[5:11] = (1,2,3,4,5,1)
        #3rd row)
        y = np.sin(4*pio3)
        pxy[11] = (-1.0 + np.cos(4*pio3),y) #6)
        pxy[12] = (np.cos(4*pio3), y) #7)
        pxy[13] = (np.cos(5*pio3), y) #8)
        pxy[14] = (1 + np.cos(5*pio3), y) #9)
        pxy[15] = (2 + np.cos(5*pio3), y) #10)
        pxy[16] = (3 + np.cos(5*pio3), y) #6)
        pinds[11:17] = (6,7,8,9,10,6)
        #4th row)
        y *= 2
        pxy[17] = (-1.,y)
        pxy[18] = (0.,y)
        pxy[19] = (1.,y)
        pxy[20] = (2.,y)
        pxy[21] = (3.,y)
        pinds[17:22] = 11
        # centers
        pxy[22] = (pxy[0] + pxy[5] + pxy[6]) / 3.0 #0
        pxy[23] = (pxy[1] + pxy[6] + pxy[7]) / 3.0 #1
        pxy[24] = (pxy[2] + pxy[7] + pxy[8]) / 3.0 #2
        pxy[25] = (pxy[3] + pxy[8] + pxy[9]) / 3.0 #3
        pxy[26] = (pxy[4] + pxy[9] + pxy[10]) / 3.0 #4
        #
        pxy[27] = (pxy[5] + pxy[11] + pxy[12])/3.0 #5
        pxy[28] = (pxy[5] + pxy[6] + pxy[12])/3.0 #6
        pxy[29] = (pxy[6] + pxy[12] + pxy[13])/3.0 #7
        pxy[30] = (pxy[6] + pxy[7] + pxy[13])/3.0 #8
        pxy[31] = (pxy[7] + pxy[13] + pxy[14])/3.0 #9
        pxy[32] = (pxy[7] + pxy[8] + pxy[14])/3.0 #10
        pxy[33] = (pxy[8] + pxy[14] + pxy[15])/3.0 #11
        pxy[34] = (pxy[8] + pxy[9] + pxy[15])/3.0 #12
        pxy[35] = (pxy[9] + pxy[15] + pxy[16])/3.0 #13
        pxy[36] = (pxy[9] + pxy[10] + pxy[16])/3.0 #14
        #
        pxy[37] = (pxy[11] + pxy[12] + pxy[17]) / 3.0 #15
        pxy[38] = (pxy[12] + pxy[13] + pxy[18]) / 3.0 #16
        pxy[39] = (pxy[13] + pxy[14] + pxy[19]) / 3.0 #17
        pxy[40] = (pxy[14] + pxy[15] + pxy[20]) / 3.0 #18
        pxy[41] = (pxy[15] + pxy[16] + pxy[21]) / 3.0 #19
        pinds[22:42] = range(12,32)
        if "dual" in oname.lower():
            dinds = np.zeros(40,dtype=int)
            ncenters = 12
            pio6 = 0.5*pio3
            dxy = np.zeros([48,2])
            #vertices
            dxy[0] = (pxy[0,0]+np.cos(5*pio6), pxy[0,1]+0.5*np.sin(5*pio6))
            dxy[1:6] = pxy[22:27]
            dxy[6] = (pxy[4,0] + np.cos(pio6), pxy[4,1]+0.5*np.sin(pio6))
            dinds[0] = 16
            dinds[1:6] = range(12,17)
            dinds[6] = 12
            dxy[7] = (pxy[5,0] + np.cos(-5*pio6), pxy[5,1] + np.sin(-5*pio6))
            dxy[8:18] = pxy[27:37]
            dxy[18] = (pxy[36,0] + np.cos(-pio6), pxy[36,1]+0.5*np.sin(-pio6))
            dxy[19] = (pxy[37,0] + np.cos(-5*pio6), pxy[37,1]+np.sin(-5*pio6))
            dinds[7] = 26
            dinds[8:18] = range(17,27)
            dinds[18] = 17
            dinds[19] = 29
            dxy[20:25] = pxy[37:42]
            dinds[20:25] = range(27,32)
            dxy[25] = (dxy[24,0]+np.cos(-pio6), dxy[24,1]+np.sin(-pio6))
            dinds[25] = 27
            #centers
            dxy[26] = pxy[2]
            dinds[26] = 0
            dxy[27:33] = pxy[5:11]
            dinds[27:33] = (1,2,3,4,5,1)
            dxy[33:39] = pxy[11:17]
#             dxy[37:43] = pxy[11:17]
            dinds[33:39] = (6,7,8,9,10,6)
#             dinds[37:43] = (6,7,8,9,10,6)
            dxy[39] = pxy[19]
#             dxy[43:48] = pxy[17:22]
            dinds[39] = 11
#             dinds[43:48] = 11
#                                0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32
            edge_inds =np.array([4,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 29])
            edge_origs=np.array([0,  1,  2,  3,  4,  5,  1,  2,  3,  4,  5,  9,  11, 13, 15, 17, 9,  11, 13, 15, 17, 7,  8,  10, 12, 14, 16, 20, 21, 22, 23, 24, 19])
            edge_dests=np.array([1,  2,  3,  4,  5,  6,  9, 11, 13, 15, 17,  8,  10, 12, 14, 16, 10, 12, 14, 16, 18, 8,  20, 21, 22, 23, 24, 21, 22, 23, 24, 25, 20])

#                 ax1.text(dxy[26+i,0]-0.05, dxy[26+i,1], str(i+indexBase), color='b', bbox=dict(facecolor='b', alpha=0.25))
            make_ticklabels_invisible(plt.gcf())

            points_list = list(range(40))
            points_list.remove(0)
            points_list.remove(6)
            points_list.remove(7)
            points_list.remove(18)
            points_list.remove(19)
            points_list.remove(25)
            for i in points_list:
                ax0.plot(dxy[i,0],dxy[i,1], 'ko', markersize=m_size)

            for i in points_list:
                ax0.text(dxy[i,0], dxy[i,1], '{}'.format(dinds[i]), color='k')
            ax0.set(title='edges & particles')
            ax1.set(title='edges & faces')
            for i in range(len(edge_inds)):
                if ints is not None:
                    exy = edgeXyz(dxy, edge_origs[i], edge_dests[i], ints[i])
                else:
                    exy = edgeXyz(dxy, edge_origs[i], edge_dests[i], None)
                dx = exy[1:,0] - exy[0:-1,0]
                dy = exy[1:,1] - exy[0:-1,1]
                midpt = 0.5 * (exy[0] + exy[-1])
                ax0.arrow(exy[0,0], exy[0,1], midpt[0]-exy[0,0], midpt[1]-exy[0,1], head_width=0.1,
                    head_length=0.05, fc='r', ec='r', length_includes_head=False)
                ax0.plot([midpt[0],exy[-1,0]],[midpt[1],exy[-1,1]],'r-')
                ax0.text(midpt[0], midpt[1]+0.05, str(edge_inds[i] + indexBase), color='r')
                ax1.arrow(exy[0,0], exy[0,1], midpt[0]-exy[0,0], midpt[1]-exy[0,1], head_width=0.1,
                    head_length=0.05, fc='r', ec='r', length_includes_head=False)
                ax1.plot([midpt[0],exy[-1,0]],[midpt[1],exy[-1,1]],'r-')
                ax1.text(midpt[0], midpt[1]+0.05, str(edge_inds[i]+indexBase), color='r')
            cell_inds = np.array([0,1,2,3,4,5,1,6,7,8,9,10,6,11])
            for i in range(14):
                ax1.text(dxy[26+i,0]-0.05, dxy[26+i,1]-0.05, str(cell_inds[i]+indexBase), color='b', bbox=dict(facecolor='r',alpha=0.25))
        else:
            einds  = np.array([0,1,2,2,3,4,4,5,6,6,7,8,8, 9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,10,25,26,26,27,27,28,28,29,29,25], dtype=int)
            porigs = np.array([0,5,6,6,6,1,2,7,8,8,8,9,9, 9,4, 5,11,12,12,12,13,13,13, 7,14,14,15,15,15,16,10,17,17,18,13,13,19,20,15,15,21], dtype=int)
            pdests = np.array([5,6,0,1,7,7,7,8,2,3,9,3,4,10,10,11,12, 5, 6,13, 6, 7,14,14, 8,15, 8, 9,16, 9,16,11,12,12,18,19,14,14,20,21,16], dtype=int)
            for i in range(ncenters):
                ax1.text(pxy[22+i,0]-0.05, pxy[22+i,1], str(i+indexBase), color='b', bbox=dict(facecolor='r', alpha=0.25))
            make_ticklabels_invisible(plt.gcf())

            ax0.plot(pxy[:,0], pxy[:,1], 'ko', markersize=m_size)
            for i in range(len(pinds)):
                ax0.text(pxy[i,0], pxy[i,1], str(pinds[i]+indexBase), color='k')
            ax0.set(title='edges & particles')
            ax1.set(title='edges & faces')
            for i in range(len(einds)):
                if ints is not None:
                    exy = edgeXyz(pxy, porigs[i], pdests[i], ints[i])
                else:
                    exy = edgeXyz(pxy, porigs[i], pdests[i], None)
                dx = exy[1:,0] - exy[0:-1,0]
                dy = exy[1:,1] - exy[0:-1,1]
                midpt = 0.5 * (exy[0] + exy[-1])
                ax0.arrow(exy[0,0], exy[0,1], midpt[0]-exy[0,0], midpt[1]-exy[0,1], head_width=0.1,
                    head_length=0.05, fc='r', ec='r', length_includes_head=False)
                ax0.plot([midpt[0],exy[-1,0]],[midpt[1],exy[-1,1]],'r-')
                ax0.text(midpt[0], midpt[1]+0.05, str(einds[i] + indexBase), color='r')
                ax1.arrow(exy[0,0], exy[0,1], midpt[0]-exy[0,0], midpt[1]-exy[0,1], head_width=0.1,
                    head_length=0.05, fc='r', ec='r', length_includes_head=False)
                ax1.plot([midpt[0],exy[-1,0]],[midpt[1],exy[-1,1]],'r-')
                ax1.text(midpt[0], midpt[1]+0.05, str(einds[i]+indexBase), color='r')



    fig0.savefig(oname, bbox_inches='tight')
    plt.close(fig0)

def sphereEdgeXyz(xyz, orig, dest, crossPi=False):
    ptsPerEdge = 5
    result = np.zeros([ptsPerEdge,3])
    tparam = np.linspace(0.0, 1.0, ptsPerEdge)
    evec = xyz[dest] - xyz[orig]
    for i,t in enumerate(tparam):
        result[i,0] = xyz[orig][0] + t * evec[0]
        result[i,1] = xyz[orig][1] + t * evec[1]
        result[i,2] = xyz[orig][2] + t * evec[2]
        norm = np.sqrt(np.sum(np.square(result[i])))
        result[i] /= norm
    return result

def plotPlaneSeed(oname, xyz, origs, dests, lefts, rights, ints, faceVerts, faceCenters, faceEdges):
    m_size = 4.0
    m_width = m_size / 4.0
    l_width= 2.0

    fig0 = plt.figure()
    #plt.tight_layout(pad=0.1, w_pad=6, h_pad=6)
    gs = GridSpec(2,2)

    ax0=plt.subplot(gs[0,0])
    ax1=plt.subplot(gs[0,1])
    ax2=plt.subplot(gs[1,:])
    ax2.axis('off')

    # panel-relative numbering
    if 'tri' in oname:
        v1 = np.array([[0,np.sin(np.pi/3.0)],[-0.5,0],[0.5,0],[0.,np.sin(np.pi/3.0)/3.0]])
        v2 = np.array([[0,0],[np.cos(np.pi/3.0),np.sin(np.pi/3.0)],[np.cos(2*np.pi/3.0),np.sin(2*np.pi/3.0)],[0,2*np.sin(np.pi/3.0)/3.0]])
        leftshift = np.zeros([4,2])
        rightshift = np.zeros([4,2])
        leftshift[:,0] = -0.5
        rightshift[:,0] = 0.5
        v1 += leftshift
        v2 += rightshift
        ax2.set_aspect('equal')
        ax2.plot(v1[:,0], v1[:,1], 'k.',markersize=8)
        ax2.plot(v2[:,0], v2[:,1], 'k.',markersize=8)
        uptri = plt.Polygon(v1[:3,:], facecolor='white', edgecolor='r')
        downtri=plt.Polygon(v2[:3,:], facecolor='white',edgecolor='r')
        ax2.add_patch(uptri)
        ax2.add_patch(downtri)
        ax2.text(v1[1,0],v1[1,1]+0.15,'panel-relative indexing',color='k',rotation=60,rotation_mode='anchor')
        for i in range(3):
            ax2.text(v1[i,0]+0.01,v1[i,1]+0.01,'v'+str(i+indexBase),color='k')
            ax2.text(v2[i,0]+0.01,v2[i,1]+0.01,'v'+str(i+indexBase),color='k')
            ax2.text(0.5*(v1[i,0]+v1[(i+1)%3,0])+0.02, 0.5*(v1[i,1]+v1[(i+1)%3,1]), 'e'+str(i+indexBase),color='r')
            ax2.text(0.5*(v2[i,0]+v2[(i+1)%3,0])+0.02, 0.5*(v2[i,1]+v2[(i+1)%3,1]), 'e'+str(i+indexBase),color='r')
        ax2.text(v1[3,0]+0.01,v1[3,1]+0.01,'c'+str(indexBase),color='k')
        ax2.text(v2[3,0]+0.01,v2[3,1]+0.01,'c'+str(indexBase),color='k')
    elif 'quad' in oname:
        corners = np.array([[-1.,1.],[-1.,-1.],[1.,-1.],[1.,1.]])
        ax2.set_aspect('equal')
        rangle=ax2.transData.transform_angles(np.array((90,)),corners[0].reshape((1,2)))[0]
        ax2.text(corners[1,0]-0.25,corners[1,1]+0.15,'panel-relative indexing',color='k',rotation=90,rotation_mode='anchor')
        sq = plt.Polygon(corners, facecolor='white', edgecolor='r')
        ax2.add_patch(sq)
        for i in range(4):
            ax2.text(0.5*(corners[i,0]+corners[(i+1)%4,0]), 0.5*(corners[i,1]+corners[(i+1)%4,1]), 'e'+str(i+indexBase),color='r')

        if 'cubic' in oname or 'Cubic' in oname:
            pts, qp, qw = refQuadCubic()
            ax2.plot(pts[:,0],pts[:,1],'k.',markersize=8)
            for i in range(12):
                ax2.text(pts[i,0]+0.01,pts[i,1]+0.01,'v'+str(i+indexBase),color='k')
            for i,pt in enumerate(pts[12:]):
                ax2.text(pt[0]+0.01,pt[1]+0.01,'c'+str(i+indexBase), color='k')
        else:
            ax2.plot(corners[:,0],corners[:,1],'k.',markersize=8)
            ax2.plot(0,0,'k.',markersize=8)
            for i in range(4):
                ax2.text(corners[i,0]+0.01,corners[i,1]+0.01,'v'+str(i+indexBase),color='k')
            ax2.text(0.01,0.01,'c'+str(indexBase), color='k')

    ax0.plot(xyz[:,0], xyz[:,1], 'ko', markersize=m_size)
    ax0.set(title='edges & particles') #, xlabel='x', ylabel='y')
    ax0.set_aspect('equal','box')

    nparticles = np.shape(xyz)[0]
    nedges = np.shape(origs)[0]
    nfaces = np.shape(faceVerts)[0]
    ncenters = np.shape(faceCenters)[0]
    nverts = np.shape(faceVerts)[1]
    for i in range(nparticles):
        ax0.text(xyz[i,0], xyz[i,1], str(i+indexBase), color='k')

    for i in range(nedges):
        if ints is not None:
            exy = edgeXyz(xyz, origs[i], dests[i], ints[i])
        else:
            exy = edgeXyz(xyz, origs[i], dests[i], None)
        dx = exy[1:,0] - exy[0:-1,0]
        dy = exy[1:,1] - exy[0:-1,1]
        midpt = 0.5 * (exy[0] + exy[-1])
        ax0.arrow(exy[0,0], exy[0,1], midpt[0]-exy[0,0], midpt[1]-exy[0,1], head_width=0.1,
            head_length=0.05, fc='r', ec='r', length_includes_head=False)
        ax0.plot([midpt[0],exy[-1,0]],[midpt[1],exy[-1,1]],'r-')
        ax0.text(midpt[0], midpt[1]+0.05, str(i+indexBase), color='r')

    ax1.set_aspect('equal','box')
    ax1.set(title='faces & edges' )#,xlabel='x',ylabel='y')
    for i in range(nedges):
        if ints is not None:
            exy = edgeXyz(xyz, origs[i], dests[i], ints[i])
        else:
            exy = edgeXyz(xyz, origs[i], dests[i], None)
        dx = exy[1:,0] - exy[0:-1,0]
        dy = exy[1:,1] - exy[0:-1,1]
        midpt = 0.5 * (exy[0] + exy[-1])
        ax1.arrow(exy[0,0], exy[0,1], midpt[0]-exy[0,0], midpt[1]-exy[0,1], head_width=0.1,
            head_length=0.05, fc='r', ec='r', length_includes_head=False)
        ax1.plot([midpt[0],exy[-1,0]],[midpt[1],exy[-1,1]],'r-')
        ax1.text(midpt[0], midpt[1]+0.1, str(i+indexBase), color='r')

    for i in range(nfaces):
        cntd = np.zeros(2)
        for j in range(nverts):
            cntd += xyz[faceVerts[i,j]]
        for j in range(ncenters):
            if len(np.shape(faceCenters)) == 2:
                cntd += xyz[faceCenters[i,j]]
            else:
                cntd += xyz[faceCenters[i]]
        cntd /= (nverts+ncenters)
        ax1.text(cntd[0], cntd[1], str(i+indexBase), color='b', bbox=dict(facecolor='b', alpha=0.25))




    fig0.savefig(oname, bbox_inches='tight')
    plt.close(fig0)

if (__name__ == "__main__"):

    print("tri hex seed")
    xyz, origs, dests, lefts, rights, ints, faceVerts, faceCenters, faceEdges = triHexSeed()
    writeSeedFile("triHexSeed.dat", xyz, origs, dests, lefts, rights, ints, faceVerts, faceCenters, faceEdges)
    plotPlaneSeed("triHexSeed.pdf", xyz, origs, dests, lefts, rights, ints, faceVerts, faceCenters, faceEdges)
    writeNamelistFile("triHex.namelist", xyz, origs, dests, lefts, rights, ints, faceVerts, faceCenters, faceEdges)

    print("quad rect seed")
    xyz, origs, dests, lefts, rights, ints, faceVerts, faceCenters, faceEdges = quadRectSeed()
    writeSeedFile("quadRectSeed.dat", xyz, origs, dests, lefts, rights, ints, faceVerts, faceCenters, faceEdges)
    plotPlaneSeed("quadRectSeed.pdf", xyz, origs, dests, lefts, rights, ints, faceVerts, faceCenters, faceEdges)
    writeNamelistFile("quadRect.namelist", xyz, origs, dests, lefts, rights, ints, faceVerts, faceCenters, faceEdges)

    print("quad cubic seed")
    xyz, origs, dests, lefts, rights, ints, faceVerts, faceCenters, faceEdges = quadCubicSeed()
    writeSeedFile("quadCubicSeed.dat", xyz, origs, dests, lefts, rights, ints, faceVerts, faceCenters, faceEdges)
    plotPlaneSeed("quadCubicSeed.pdf", xyz, origs, dests, lefts, rights, ints, faceVerts, faceCenters, faceEdges)
    writeNamelistFile('quadCubic.namelist', xyz, origs, dests, lefts, rights, ints, faceVerts, faceCenters, faceEdges)

    print("cubed sphere seed")
    xyz, origs, dests, lefts, rights, ints, faceVerts, faceCenters, faceEdges = cubedSphereSeed()
    writeSeedFile('cubedSphereSeed.dat', xyz, origs, dests, lefts, rights, ints, faceVerts, faceCenters, faceEdges)
    plotSphereSeed('cubedSphereSeed.pdf',xyz, origs, dests, lefts, rights, ints, faceVerts, faceCenters, faceEdges)
    writeNamelistFile("cubedSphere.namelist", xyz, origs, dests, lefts, rights, ints, faceVerts, faceCenters, faceEdges)

    print("icosahedral triangle sphere seed")
    xyz, origs, dests, lefts, rights, ints, faceVerts, faceCenters, faceEdges = icosTriSeed()
    writeSeedFile('icosTriSphereSeed.dat', xyz, origs, dests, lefts, rights, ints, faceVerts, faceCenters, faceEdges)
    plotSphereSeed('icosTriSphereSeed.pdf', xyz, origs, dests, lefts, rights, ints, faceVerts, faceCenters, faceEdges)
    writeNamelistFile("icosTri.namelist", xyz, origs, dests, lefts, rights, ints, faceVerts, faceCenters, faceEdges)

    print("icosahedral dual sphere seed")
    xyz, origs, dests, lefts, rights, ints, faceVerts, faceCenters, faceEdges, origCw, origCcw, destCw, destCcw = icosTriDualSeed()
    writeSeedFile('icosTriDualSeed.dat', xyz, origs, dests, lefts, rights, ints, faceVerts, faceCenters, faceEdges, origCw, origCcw, destCw, destCcw)
    plotSphereSeed('icosTriDualSeed.pdf', xyz, origs, dests, lefts, rights, ints, faceVerts, faceCenters, faceEdges)

