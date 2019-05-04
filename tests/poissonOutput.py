import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    serial_output = """icostri: 45.4164: linf_verts = 2.42561, linf_faces = 1.49743
    cubedsphere: 82.9186: linf_verts = 0.2566, linf_faces = 0
    icostri: 22.7082: linf_verts = 0.175552, linf_faces = 0.446258
    cubedsphere: 41.4593: linf_verts = 0.898719, linf_faces = 0.466355
    icostri: 11.3541: linf_verts = 0.0327947, linf_faces = 0.18008
    cubedsphere: 20.7296: linf_verts = 0.105744, linf_faces = 0.330876
    icostri: 5.67705: linf_verts = 0.0078577, linf_faces = 0.057519
    cubedsphere: 10.3648: linf_verts = 0.0242884, linf_faces = 0.153158
    icostri: 2.83852: linf_verts = 0.00199034, linf_faces = 0.0170644
    cubedsphere: 5.18241: linf_verts = 0.00611779, linf_faces = 0.0500934
    icostri: 1.41926: linf_verts = 0.000499979, linf_faces = 0.00492556
    cubedsphere: 2.59121: linf_verts = 0.00152096, linf_faces = 0.0148438"""
    
    lines = serial_output.split('\n')
    ic = []
    cs = []
    for l in lines:
        if 'icostri' in l:
            ic.append(l)
        elif: 'cubedsphere' in l:
            cs.append(l)
    
    numtrials = len(ic)
    icmeshsize = np.zeros(numtrials)
    iclinfverts = np.zeros(numtrials)
    iclinffaces = np.zeros(numtrials)
    
    csmeshsize = np.zeros(numtrials)
    cslinfverts = np.zeros(numtrials)
    cslinffaces = np.zeros(numtrials)
    
    for i, l in enumerate(ic):
        ls = l.split()
        icmeshsize[i] = float(ls[1])
        iclinfverts[i] = float(ls[4])
        iclinffaces[i] = float(ls[7])
    
    for i, l in enumerate(cs):
        ls = l.split()
        csmeshsize[i] = float(ls[1])
        cslinfverts[i] = float(ls[4])
        cslinffaces[i] = float(ls[7])
    
    
    
    

