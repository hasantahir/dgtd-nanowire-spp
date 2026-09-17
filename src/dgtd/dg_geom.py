"""
dg_geom.py — Stage 1: geometric factors for straight-sided tetrahedra.
Maps reference operators (Dr,Ds,Dt) to physical space and computes face
normals + surface Jacobians. Validated in test_geom.py.
"""
import numpy as np

def geometric_factors(x, y, z, Dr, Ds, Dt):
    """
    x,y,z: (K, Np) physical node coords per element.
    Returns rx,ry,rz, sx,sy,sz, tx,ty,tz, J  each (K, Np).
    For straight-sided tets these are constant per element, but we keep the
    nodal shape for generality.
    """
    xr = x @ Dr.T; xs = x @ Ds.T; xt = x @ Dt.T
    yr = y @ Dr.T; ys = y @ Ds.T; yt = y @ Dt.T
    zr = z @ Dr.T; zs = z @ Ds.T; zt = z @ Dt.T

    J = xr*(ys*zt - zs*yt) - yr*(xs*zt - zs*xt) + zr*(xs*yt - ys*xt)

    rx =  (ys*zt - zs*yt)/J
    ry = -(xs*zt - zs*xt)/J
    rz =  (xs*yt - ys*xt)/J
    sx = -(yr*zt - zr*yt)/J
    sy =  (xr*zt - zr*xt)/J
    sz = -(xr*yt - yr*xt)/J
    tx =  (yr*zs - zr*ys)/J
    ty = -(xr*zs - zr*xs)/J
    tz =  (xr*ys - yr*xs)/J
    return rx,ry,rz, sx,sy,sz, tx,ty,tz, J

def normals_surface(x, y, z, Dr, Ds, Dt, fmask, Nfp):
    """
    Outward unit normals (nx,ny,nz) and surface scaling Fscale = sJ/J(face)
    for each of the 4 faces. Returns arrays shaped (K, 4*Nfp).
    """
    rx,ry,rz, sx,sy,sz, tx,ty,tz, J = geometric_factors(x,y,z,Dr,Ds,Dt)
    K = x.shape[0]
    # reference face normals in (r,s,t) space
    # f0 t=-1: -t ; f1 s=-1: -s ; f2 r+s+t=-1: +(r+s+t) ; f3 r=-1: -r
    nx = np.zeros((K, 4*Nfp)); ny = np.zeros((K, 4*Nfp)); nz = np.zeros((K, 4*Nfp))
    sJ = np.zeros((K, 4*Nfp))
    # gather face-node geometric factors
    def face_slice(f): return slice(f*Nfp,(f+1)*Nfp)
    for f, (cr,cs,ct) in enumerate([(0,0,-1),(0,-1,0),(1,1,1),(-1,0,0)]):
        idx = fmask[f]
        nxf = cr*rx[:,idx] + cs*sx[:,idx] + ct*tx[:,idx]
        nyf = cr*ry[:,idx] + cs*sy[:,idx] + ct*ty[:,idx]
        nzf = cr*rz[:,idx] + cs*sz[:,idx] + ct*tz[:,idx]
        mag = np.sqrt(nxf**2+nyf**2+nzf**2)
        nx[:,face_slice(f)] = nxf/mag
        ny[:,face_slice(f)] = nyf/mag
        nz[:,face_slice(f)] = nzf/mag
        sJ[:,face_slice(f)] = mag*J[:,idx]
    # Fscale = sJ / J at the face nodes
    Jf = np.concatenate([J[:,fmask[f]] for f in range(4)], axis=1)
    Fscale = sJ/Jf
    return nx,ny,nz, sJ, Fscale, J
