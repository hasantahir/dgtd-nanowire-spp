"""
dg_connect.py — Stage 2: face-node connectivity (vmapM, vmapP).
For each face node of each element, find the coincident node in the neighboring
element by coordinate matching. Boundary nodes map to themselves (vmapP=vmapM).
Validated by coordinate-gap check in test_connect.py.
"""
import numpy as np

def build_face_connectivity(EToV, x, y, z, fmask, Nfp, tol=1e-6):
    """
    EToV: (K,4) vertex indices per element (mesh connectivity, 0-indexed).
    x,y,z: (K,Np) physical node coords.
    Returns vmapM, vmapP (flattened global indices into the (K*Np) node list),
    and EToE/EToF element/face adjacency.
    """
    K = EToV.shape[0]
    Np = x.shape[1]

    # --- element-to-element via shared faces (vertex triples) ---
    # tet faces by local vertex (matching reference face ordering used elsewhere)
    # f0 t=-1: verts (0,1,2); f1 s=-1: (0,1,3); f2: (1,2,3); f3 r=-1: (0,2,3)
    face_vtx = np.array([[0,1,2],[0,1,3],[1,2,3],[0,2,3]])
    # build (4K, 3) sorted vertex triples
    triples = np.sort(EToV[:, face_vtx], axis=2).reshape(K*4, 3)
    elem_of = np.repeat(np.arange(K), 4)
    face_of = np.tile(np.arange(4), K)
    order = np.lexsort((triples[:,2],triples[:,1],triples[:,0]))
    ts = triples[order]
    EToE = np.repeat(np.arange(K),4).reshape(K,4).astype(int)
    EToF = np.tile(np.arange(4),K).reshape(K,4).astype(int)
    same = np.all(ts[1:]==ts[:-1], axis=1)
    pr = np.where(same)[0]
    a0,a1 = order[pr], order[pr+1]
    e0,f0 = elem_of[a0],face_of[a0]; e1,f1 = elem_of[a1],face_of[a1]
    EToE[e0,f0]=e1; EToF[e0,f0]=f1
    EToE[e1,f1]=e0; EToF[e1,f1]=f0

    # --- node-level matching ---
    Nfaces=4
    vmapM = np.zeros((K, Nfaces, Nfp), dtype=int)
    vmapP = np.zeros((K, Nfaces, Nfp), dtype=int)
    # global index of node n in elem k is k*Np + n
    for k in range(K):
        for f in range(Nfaces):
            idM = fmask[f]
            vmapM[k,f,:] = k*Np + idM
    # fill vmapP
    for k in range(K):
        for f in range(Nfaces):
            kn = EToE[k,f]; fn = EToF[k,f]
            idM = fmask[f]
            gM = k*Np + idM
            if kn==k and fn==f:
                vmapP[k,f,:] = gM      # boundary: self
                continue
            idP = fmask[fn]
            # coordinates on this face (minus side) and neighbor face (plus side)
            xM=x[k,idM]; yM=y[k,idM]; zM=z[k,idM]
            xP=x[kn,idP]; yP=y[kn,idP]; zP=z[kn,idP]
            # match each minus node to nearest plus node
            for i in range(Nfp):
                d2=(xP-xM[i])**2+(yP-yM[i])**2+(zP-zM[i])**2
                j=np.argmin(d2)
                vmapP[k,f,i] = kn*Np + idP[j]
    return vmapM.reshape(K,Nfaces*Nfp), vmapP.reshape(K,Nfaces*Nfp), EToE, EToF
