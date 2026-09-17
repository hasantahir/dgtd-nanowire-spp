"""
dg_nodal3d.py
=============
Reference-tetrahedron operators for nodal DGTD (Hesthaven & Warburton, 2008).
Pure NumPy/SciPy. Builds:
  - equispaced->warp&blend interpolation nodes on the reference tet
  - orthonormal modal basis (Jacobi polynomials on the collapsed coords)
  - Vandermonde V, mass M=(V V^T)^-1, differentiation Dr,Ds,Dt
  - face mass / LIFT operator for surface flux integrals

Reference tet vertices: (-1,-1,-1),(1,-1,-1),(-1,1,-1),(-1,-1,1).
This file is unit-tested in test_dg_nodal3d.py.
"""
import numpy as np
from scipy.special import gamma

# ---------------------------------------------------------------------------
# Jacobi polynomials & their derivatives (orthonormal on [-1,1] w.r.t (1-x)^a (1+x)^b)
# ---------------------------------------------------------------------------
def jacobi_p(x, alpha, beta, N):
    """Evaluate orthonormal Jacobi polynomial P_N^{(a,b)}(x), vector x."""
    x = np.asarray(x, dtype=float)
    PL = np.zeros((N+1, x.size))
    gamma0 = (2**(alpha+beta+1)/(alpha+beta+1) *
              gamma(alpha+1)*gamma(beta+1)/gamma(alpha+beta+1))
    PL[0] = 1.0/np.sqrt(gamma0)
    if N == 0:
        return PL[0]
    gamma1 = (alpha+1)*(beta+1)/(alpha+beta+3)*gamma0
    PL[1] = ((alpha+beta+2)*x/2 + (alpha-beta)/2)/np.sqrt(gamma1)
    aold = 2.0/(2+alpha+beta)*np.sqrt((alpha+1)*(beta+1)/(alpha+beta+3))
    for i in range(1, N):
        h1 = 2*i+alpha+beta
        anew = 2.0/(h1+2)*np.sqrt((i+1)*(i+1+alpha+beta)*(i+1+alpha)*
                                  (i+1+beta)/(h1+1)/(h1+3))
        bnew = -(alpha**2-beta**2)/h1/(h1+2)
        PL[i+1] = (1.0/anew)*(-aold*PL[i-1] + (x-bnew)*PL[i])
        aold = anew
    return PL[N]

def grad_jacobi_p(x, alpha, beta, N):
    if N == 0:
        return np.zeros_like(np.asarray(x, dtype=float))
    return np.sqrt(N*(N+alpha+beta+1))*jacobi_p(x, alpha+1, beta+1, N-1)

# ---------------------------------------------------------------------------
# collapsed coordinates  (r,s,t) -> (a,b,c) for the orthonormal tet basis
# ---------------------------------------------------------------------------
def rst_to_abc(r, s, t):
    with np.errstate(divide='ignore', invalid='ignore'):
        a = np.where(np.abs(s+t) > 1e-12, 2*(1+r)/(-(s+t)) - 1, -1.0)
        b = np.where(np.abs(t-1) > 1e-12, 2*(1+s)/(1-t) - 1, -1.0)
    c = t.copy()
    return a, b, c

def simplex3d_p(a, b, c, i, j, k):
    """Orthonormal basis on the reference tet at collapsed coords."""
    h1 = jacobi_p(a, 0, 0, i)
    h2 = jacobi_p(b, 2*i+1, 0, j)
    h3 = jacobi_p(c, 2*(i+j)+2, 0, k)
    return 2*np.sqrt(2.0)*h1*h2*((1-b)**i)*h3*((1-c)**(i+j))

def grad_simplex3d_p(a, b, c, idx, jdx, kdx):
    fa  = jacobi_p(a, 0, 0, idx);        dfa = grad_jacobi_p(a, 0, 0, idx)
    gb  = jacobi_p(b, 2*idx+1, 0, jdx);  dgb = grad_jacobi_p(b, 2*idx+1, 0, jdx)
    hc  = jacobi_p(c, 2*(idx+jdx)+2, 0, kdx); dhc = grad_jacobi_p(c, 2*(idx+jdx)+2, 0, kdx)

    # r-derivative
    dmodedr = dfa*gb*hc
    if idx > 0:    dmodedr *= (0.5*(1-b))**(idx-1)
    if idx+jdx > 0: dmodedr *= (0.5*(1-c))**(idx+jdx-1)
    # s-derivative
    dmodeds = 0.5*(1+a)*dmodedr
    tmp = dgb*((0.5*(1-b))**idx)
    if idx > 0: tmp += (-0.5*idx)*(gb*(0.5*(1-b))**(idx-1))
    if idx+jdx > 0: tmp *= (0.5*(1-c))**(idx+jdx-1)
    tmp = fa*tmp*hc
    dmodeds += tmp
    # t-derivative
    dmodedt = 0.5*(1+a)*dmodedr + 0.5*(1+b)*tmp
    tmp = dhc*((0.5*(1-c))**(idx+jdx))
    if idx+jdx > 0: tmp += (-0.5*(idx+jdx))*(hc*(0.5*(1-c))**(idx+jdx-1))
    tmp = fa*(gb*((0.5*(1-b))**idx))*tmp
    dmodedt += tmp
    # normalization
    f = 2**(2*idx+jdx+1.5)
    return f*dmodedr, f*dmodeds, f*dmodedt

# ---------------------------------------------------------------------------
# node sets
# ---------------------------------------------------------------------------
def equinodes3d(N):
    """Equispaced nodes on reference tet, returns r,s,t and the (i,j,k) order."""
    Np = (N+1)*(N+2)*(N+3)//6
    r = np.zeros(Np); s = np.zeros(Np); t = np.zeros(Np)
    idx = 0
    for n in range(N+1):
        for m in range(N+1-n):
            for q in range(N+1-n-m):
                r[idx] = -1 + 2*q/N if N>0 else -1
                s[idx] = -1 + 2*m/N if N>0 else -1
                t[idx] = -1 + 2*n/N if N>0 else -1
                idx += 1
    return r, s, t

def vandermonde3d(N, r, s, t):
    Np = (N+1)*(N+2)*(N+3)//6
    V = np.zeros((len(r), Np))
    a, b, c = rst_to_abc(r, s, t)
    col = 0
    for i in range(N+1):
        for j in range(N+1-i):
            for k in range(N+1-i-j):
                V[:, col] = simplex3d_p(a, b, c, i, j, k)
                col += 1
    return V

def grad_vandermonde3d(N, r, s, t):
    Np = (N+1)*(N+2)*(N+3)//6
    Vr = np.zeros((len(r), Np)); Vs = np.zeros((len(r), Np)); Vt = np.zeros((len(r), Np))
    a, b, c = rst_to_abc(r, s, t)
    col = 0
    for i in range(N+1):
        for j in range(N+1-i):
            for k in range(N+1-i-j):
                dr, ds, dt = grad_simplex3d_p(a, b, c, i, j, k)
                Vr[:, col], Vs[:, col], Vt[:, col] = dr, ds, dt
                col += 1
    return Vr, Vs, Vt

def build_reference(N):
    """Assemble reference-element operators for order N (equispaced nodes)."""
    r, s, t = equinodes3d(N)
    V = vandermonde3d(N, r, s, t)
    Vinv = np.linalg.inv(V)
    M = Vinv.T @ Vinv                      # mass = (V V^T)^-1
    Vr, Vs, Vt = grad_vandermonde3d(N, r, s, t)
    Dr = Vr @ Vinv; Ds = Vs @ Vinv; Dt = Vt @ Vinv
    return dict(N=N, Np=len(r), r=r, s=s, t=t, V=V, Vinv=Vinv, M=M,
                Dr=Dr, Ds=Ds, Dt=Dt)

# ---------------------------------------------------------------------------
# Face nodes and LIFT operator
# ---------------------------------------------------------------------------
def face_masks(N, r, s, t, tol=1e-8):
    """Boolean index arrays for the 4 faces of the reference tet.
    Faces: f0: t=-1, f1: s=-1, f2: r+s+t=-1, f3: r=-1."""
    f0 = np.abs(t+1) < tol
    f1 = np.abs(s+1) < tol
    f2 = np.abs(1+r+s+t) < tol
    f3 = np.abs(r+1) < tol
    return [np.where(f0)[0], np.where(f1)[0], np.where(f2)[0], np.where(f3)[0]]

def vandermonde2d(N, a, b):
    """2D orthonormal Vandermonde on a reference triangle (collapsed coords)."""
    from .dg_nodal3d import jacobi_p
    Np2 = (N+1)*(N+2)//2
    V = np.zeros((len(a), Np2)); col = 0
    for i in range(N+1):
        for j in range(N+1-i):
            h1 = jacobi_p(a, 0, 0, i); h2 = jacobi_p(b, 2*i+1, 0, j)
            V[:, col] = np.sqrt(2.0)*h1*h2*(1-b)**i
            col += 1
    return V

def build_lift(N, r, s, t, fmask):
    """LIFT = M^{-1} @ (face mass), mapping face-node values to volume residual."""
    Np = len(r)
    Nfp = (N+1)*(N+2)//2
    Emat = np.zeros((Np, 4*Nfp))
    # face-local coordinates for each face -> 2D triangle (a,b)
    face_coords = [
        (r, s),            # f0: t=-1 -> use (r,s)
        (r, t),            # f1: s=-1 -> use (r,t)
        (s, t),            # f2: r+s+t=-1 -> use (s,t)
        (s, t),            # f3: r=-1 -> use (s,t)
    ]
    for f in range(4):
        idx = fmask[f]
        u, v = face_coords[f][0][idx], face_coords[f][1][idx]
        Vf = vandermonde2d(N, u, v)
        massf = np.linalg.inv(Vf @ Vf.T)   # face mass on the triangle
        Emat[idx, f*Nfp:(f+1)*Nfp] = massf
    ref = build_reference(N)
    LIFT = np.linalg.inv(ref['M']) @ Emat
    return LIFT, Nfp

def build_reference_full(N):
    ref = build_reference(N)
    fmask = face_masks(N, ref['r'], ref['s'], ref['t'])
    # ensure each face has exactly Nfp nodes
    Nfp = (N+1)*(N+2)//2
    counts = [len(m) for m in fmask]
    ref['fmask'] = fmask
    ref['Nfp'] = Nfp
    ref['face_counts'] = counts
    return ref

# ---------------------------------------------------------------------------
# CORRECTED face integration with per-face barycentric 2D mapping
# ---------------------------------------------------------------------------
_FACE_VERTS = [
    (np.array([-1,-1,-1.]),np.array([1,-1,-1.]),np.array([-1,1,-1.])),  # f0 t=-1
    (np.array([-1,-1,-1.]),np.array([1,-1,-1.]),np.array([-1,-1,1.])),  # f1 s=-1
    (np.array([1,-1,-1.]),np.array([-1,1,-1.]),np.array([-1,-1,1.])),   # f2 r+s+t=-1
    (np.array([-1,-1,-1.]),np.array([-1,1,-1.]),np.array([-1,-1,1.])),  # f3 r=-1
]
_FACE_NORMALS = [np.array([0,0,-1.]),np.array([0,-1.,0]),
                 np.array([1,1,1.])/np.sqrt(3),np.array([-1.,0,0])]

def face_2d_mass(N, r, s, t, fmask):
    """Return per-face (mass2d, area) using correct barycentric->ref-tri mapping."""
    pts3d = np.stack([r,s,t],axis=1)
    out=[]
    for f in range(4):
        idx=fmask[f]; P0,P1,P2=_FACE_VERTS[f]
        e1=P1-P0; e2=P2-P0
        Emat=np.stack([e1,e2],axis=1)
        uv=np.linalg.lstsq(Emat, (pts3d[idx]-P0).T, rcond=None)[0].T
        a=2*uv[:,0]-1; b=2*uv[:,1]-1
        with np.errstate(divide='ignore', invalid='ignore'):
            aa=np.where(np.abs(b-1)>1e-12, 2*(1+a)/(1-b)-1, -1.0)
        Vf=vandermonde2d(N, aa, b)
        mass2d=np.linalg.inv(Vf@Vf.T)
        area=0.5*np.linalg.norm(np.cross(e1,e2))
        out.append((mass2d, area))
    return out

def build_reference_full(N):   # override the earlier definition
    ref = build_reference(N)
    fmask = face_masks(N, ref['r'], ref['s'], ref['t'])
    Nfp = (N+1)*(N+2)//2
    ref['fmask']=fmask; ref['Nfp']=Nfp
    ref['face_counts']=[len(m) for m in fmask]
    ref['face_normals']=_FACE_NORMALS
    ref['face_2d']=face_2d_mass(N, ref['r'],ref['s'],ref['t'], fmask)
    # LIFT: M^-1 @ Emat, with Emat columns = face mass scaled by (area/2)
    Emat=np.zeros((ref['Np'], 4*Nfp))
    for f in range(4):
        mass2d, area = ref['face_2d'][f]
        Emat[fmask[f], f*Nfp:(f+1)*Nfp] = mass2d*(area/2.0)
    ref['LIFT']=np.linalg.inv(ref['M'])@Emat
    ref['Emat']=Emat
    return ref
