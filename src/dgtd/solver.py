"""
ho_solver_gpu.py — GPU (CuPy) high-order nodal DGTD for aloof-electron SPP
excitation on a silver nanowire.  Zhao et al., PRB 113, 085425 (2026).

Runs on CUDA via CuPy; falls back to NumPy automatically if CuPy is absent
(so you can smoke-test on a login node before submitting).

Formulation (unchanged, already validated):
  - element-wise Drude materials from the mesh "silver" physical group
  - scattered-field: analytic incident field drives the ADE inside metal only
  - upwind flux, vacuum impedances (eps_inf = 1)
  - CFL: dt = C_CFL * min(r_insphere) / (N+1)^2

PHYSICS CORRECTIONS in this version
  1. EELS is now recorded as Ez ALONG THE WHOLE ELECTRON TRAJECTORY LINE
     (paper Eq. 3/5 needs the line integral over z), not at a single node.
  2. Field snapshots are a TRUE y~0 SLICE (previously all y collapsed onto
     the (x,z) plane, which is what produced the speckle).
  3. Raw slice data is dumped to .npz so plots can be remade without re-running.
"""
import numpy as np
import time, os, sys, argparse

# ---------------------------------------------------------------- GPU backend
try:
    import cupy as xp
    GPU = True
except Exception:
    import numpy as xp
    GPU = False

def to_cpu(a):
    return xp.asnumpy(a) if GPU else np.asarray(a)

def _dtype():
    return xp.float32 if FP32 else xp.float64

from .dg_nodal3d import build_reference_full
from .dg_geom import geometric_factors, normals_surface
from .dg_connect import build_face_connectivity

# ---------------------------------------------------------------- parameters
ORDER       = 3
R_WIRE      = 15.0
L_WIRE      = 1200.0
SRC_V       = 0.4
SRC_B       = 5.0
SRC_X0      = 360.0
SRC_Y0      = R_WIRE + SRC_B          # 20 nm
SIGMA_E     = 5.0
Q_AMP       = -1.0
T_CENTER_FS = 2.0
C_CFL       = 0.3
SLICE_YTOL  = 5.0                     # nm half-thickness of the y~0 slice
REC_EVERY   = 20                      # record Ez along the trajectory every Nth
                                      # step (Nyquist at 4 eV needs ~fs/100, so
                                      # this is safe and removes 95% of the
                                      # GPU->CPU syncs that throttle the loop)
FP32        = False                   # True: ~30x faster on consumer/RTX cards
                                      # (fp64 is 1/32 rate). Verify EELS peaks
                                      # against a short fp64 run before trusting.

c_phys = 299.792458
hbar   = 6.582119569e-16
fs     = 1e-15
WP2    = (((9.17  / hbar) * fs) / c_phys)**2
GAMMA  =  ((0.021 / hbar) * fs) / c_phys


# ---------------------------------------------------------------- mesh input
def read_tets(path):
    """Nodes (um->nm), tets, and ELEMENT-WISE metal mask from 'silver' group."""
    import gmsh
    gmsh.initialize()
    gmsh.open(path)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_coords = node_coords.reshape(-1, 3) * 1000.0
    tag2idx = {int(t): i for i, t in enumerate(node_tags)}

    etypes, etags_list, enodes = gmsh.model.mesh.getElements(dim=3)
    ti = list(etypes).index(4)
    EToV = np.vectorize(tag2idx.get)(enodes[ti].reshape(-1, 4)).astype(np.int64)
    tag2row = {int(t): i for i, t in enumerate(etags_list[ti])}

    metal = np.zeros(len(EToV), dtype=bool)
    found = False
    for (dim, ptag) in gmsh.model.getPhysicalGroups(3):
        if gmsh.model.getPhysicalName(dim, ptag) == "silver" or ptag == 1:
            found = True
            for ent in gmsh.model.getEntitiesForPhysicalGroup(dim, ptag):
                et, etg, _ = gmsh.model.mesh.getElements(dim, int(ent))
                for tt, tags in zip(et, etg):
                    if int(tt) == 4:
                        for tg in tags:
                            r = tag2row.get(int(tg))
                            if r is not None:
                                metal[r] = True
    gmsh.finalize()

    if not found:
        print("WARNING: no 'silver' physical group; falling back to centroid test")
        c = node_coords[EToV].mean(axis=1)
        r2 = c[:,1]**2 + c[:,2]**2
        metal = (((np.abs(c[:,0]) <= L_WIRE/2) & (r2 <= R_WIRE**2))
                 | (((c[:,0]+L_WIRE/2)**2 + r2) <= R_WIRE**2)
                 | (((c[:,0]-L_WIRE/2)**2 + r2) <= R_WIRE**2))
    return node_coords, EToV, metal


def build_nodes_highorder(VX, EToV, ref):
    r, s, t = ref['r'], ref['s'], ref['t']
    V0 = VX[EToV[:, 0]]; V1 = VX[EToV[:, 1]]
    V2 = VX[EToV[:, 2]]; V3 = VX[EToV[:, 3]]
    a1 = 0.5*(1+r)[None, :, None]
    a2 = 0.5*(1+s)[None, :, None]
    a3 = 0.5*(1+t)[None, :, None]
    X = (V0[:, None, :] + a1*(V1-V0)[:, None, :]
                        + a2*(V2-V0)[:, None, :]
                        + a3*(V3-V0)[:, None, :])
    return X[:,:,0].copy(), X[:,:,1].copy(), X[:,:,2].copy()


# ---------------------------------------------------------------- GPU kernels
def make_rhs(op, phys):
    """Close over device arrays; returns the RHS function for the RK loop."""
    Dr, Ds, Dt, LIFT = op['Dr'], op['Ds'], op['Dt'], op['LIFT']
    DrT, DsT, DtT, LIFTT = Dr.T.copy(), Ds.T.copy(), Dt.T.copy(), LIFT.T.copy()
    rx, ry, rz = op['rx'], op['ry'], op['rz']
    sx, sy, sz = op['sx'], op['sy'], op['sz']
    tx, ty, tz = op['tx'], op['ty'], op['tz']
    nx, ny, nz, Fscale = op['nx'], op['ny'], op['nz'], op['Fscale']
    vmapM, vmapP, mapB = op['vmapM'], op['vmapP'], op['mapB']
    K, Np, Nfp = op['K'], op['Np'], op['Nfp']
    m       = phys['metal_elems']
    wp2     = phys['wp2']
    gam     = phys['gamma']
    sigma   = phys['sigma']
    E_inc   = phys['E_inc']

    def curl(ux, uy, uz):
        uxr, uxs, uxt = ux@DrT, ux@DsT, ux@DtT
        uyr, uys, uyt = uy@DrT, uy@DsT, uy@DtT
        uzr, uzs, uzt = uz@DrT, uz@DsT, uz@DtT
        cx = (ry*uzr+sy*uzs+ty*uzt) - (rz*uyr+sz*uys+tz*uyt)
        cy = (rz*uxr+sz*uxs+tz*uxt) - (rx*uzr+sx*uzs+tx*uzt)
        cz = (rx*uyr+sx*uys+tx*uyt) - (ry*uxr+sy*uxs+ty*uxt)
        return cx, cy, cz

    def rhs(Hx,Hy,Hz, Ex,Ey,Ez, Jx,Jy,Jz, t_sim):
        cHx,cHy,cHz = curl(Hx,Hy,Hz)
        cEx,cEy,cEz = curl(Ex,Ey,Ez)

        def jump(u):
            uf = u.reshape(-1)
            return (uf[vmapM]-uf[vmapP]).reshape(K, 4*Nfp)
        dHx,dHy,dHz = jump(Hx), jump(Hy), jump(Hz)
        dEx_,dEy_,dEz_ = jump(Ex), jump(Ey), jump(Ez)

        # PEC exterior wall: [E]=2E^-, [H]=0
        ExM = Ex.reshape(-1)[vmapM].reshape(K,4*Nfp)
        EyM = Ey.reshape(-1)[vmapM].reshape(K,4*Nfp)
        EzM = Ez.reshape(-1)[vmapM].reshape(K,4*Nfp)
        dEx_ = xp.where(mapB, 2.0*ExM, dEx_)
        dEy_ = xp.where(mapB, 2.0*EyM, dEy_)
        dEz_ = xp.where(mapB, 2.0*EzM, dEz_)
        dHx  = xp.where(mapB, 0.0, dHx)
        dHy  = xp.where(mapB, 0.0, dHy)
        dHz  = xp.where(mapB, 0.0, dHz)

        ndH = nx*dHx + ny*dHy + nz*dHz
        ndE = nx*dEx_ + ny*dEy_ + nz*dEz_
        nxEx = ny*dEz_-nz*dEy_; nxEy = nz*dEx_-nx*dEz_; nxEz = nx*dEy_-ny*dEx_
        nxHx = ny*dHz -nz*dHy ; nxHy = nz*dHx -nx*dHz ; nxHz = nx*dHy -ny*dHx

        fHx = -nxEx - (dHx-ndH*nx)
        fHy = -nxEy - (dHy-ndH*ny)
        fHz = -nxEz - (dHz-ndH*nz)
        fEx =  nxHx - (dEx_-ndE*nx)
        fEy =  nxHy - (dEy_-ndE*ny)
        fEz =  nxHz - (dEz_-ndE*nz)

        half = 0.5*Fscale
        dHx_ =  cEx + (half*fHx)@LIFTT
        dHy_ =  cEy + (half*fHy)@LIFTT
        dHz_ =  cEz + (half*fHz)@LIFTT
        dEx  = -cHx + (half*fEx)@LIFTT
        dEy  = -cHy + (half*fEy)@LIFTT
        dEz  = -cHz + (half*fEz)@LIFTT

        # Drude ADE on metal ELEMENTS, driven by the total in-metal field
        Eix, Eiy, Eiz = E_inc(t_sim)
        dJx = xp.zeros_like(Jx); dJy = xp.zeros_like(Jy); dJz = xp.zeros_like(Jz)
        dJx[m] = wp2*(Ex[m]+Eix) - gam*Jx[m]
        dJy[m] = wp2*(Ey[m]+Eiy) - gam*Jy[m]
        dJz[m] = wp2*(Ez[m]+Eiz) - gam*Jz[m]
        dEx[m] -= Jx[m]; dEy[m] -= Jy[m]; dEz[m] -= Jz[m]

        dEx -= sigma*Ex; dEy -= sigma*Ey; dEz -= sigma*Ez
        dHx_ -= sigma*Hx; dHy_ -= sigma*Hy; dHz_ -= sigma*Hz
        return dHx_,dHy_,dHz_, dEx,dEy,dEz, dJx,dJy,dJz
    return rhs


# ---------------------------------------------------------------- main
def main(mesh_path, total_fs, outdir, snap_every_fs, cfl, clip_pct):
    os.makedirs(outdir, exist_ok=True)
    print(f"backend: {'CuPy/GPU' if GPU else 'NumPy/CPU'}; order p={ORDER}; "
          f"dtype={'fp32' if FP32 else 'fp64'}", flush=True)

    ref = build_reference_full(ORDER)
    Np, Nfp = ref['Np'], ref['Nfp']
    Dr, Ds, Dt = ref['Dr'], ref['Ds'], ref['Dt']

    VX, EToV, metal = read_tets(mesh_path)
    K = EToV.shape[0]
    print(f"mesh: {K} tets, Np={Np} -> {K*Np} nodes; metal elems {metal.sum()}", flush=True)

    x, y, z = build_nodes_highorder(VX, EToV, ref)
    rx,ry,rz,sx,sy,sz,tx,ty,tz,J = geometric_factors(x,y,z,Dr,Ds,Dt)
    nx,ny,nz,sJ,Fscale,_ = normals_surface(x,y,z,Dr,Ds,Dt,ref['fmask'],Nfp)
    vmapM,vmapP,EToE,EToF = build_face_connectivity(EToV,x,y,z,ref['fmask'],Nfp)
    mapB = (vmapM == vmapP)

    # CFL from true geometric insphere
    v = VX[EToV]
    v0,v1,v2,v3 = v[:,0],v[:,1],v[:,2],v[:,3]
    vol  = np.abs(np.einsum('ij,ij->i', np.cross(v1-v0,v2-v0), v3-v0))/6.0
    ta   = lambda a,b,c: 0.5*np.linalg.norm(np.cross(b-a,c-a),axis=1)
    Atot = ta(v0,v1,v2)+ta(v0,v1,v3)+ta(v1,v2,v3)+ta(v0,v2,v3)
    r_in = 3.0*vol/Atot
    # The explicit step is set by the SMALLEST element. A handful of sliver
    # tets at the curved metal surface can dictate a step 5x smaller than the
    # bulk needs. clip_pct>0 ignores that worst tail (the upwind flux is most
    # dissipative exactly there); set clip_pct=0 for the strict bound.
    if clip_pct > 0:
        h_cfl = float(np.percentile(r_in, clip_pct))
    else:
        h_cfl = float(r_in.min())
    dt   = cfl * h_cfl/(ORDER+1)**2
    nsteps = int(total_fs*c_phys/dt)
    print(f"r_in min {r_in.min():.3f} / p{clip_pct} {h_cfl:.3f} / median "
          f"{np.median(r_in):.3f} nm", flush=True)
    print(f"dt={dt/c_phys:.3e} fs; steps={nsteps}", flush=True)

    # sponge
    dxp = np.maximum(0, np.abs(x)-900.)/300.
    dyp = np.maximum(0, np.abs(y)-300.)/300.
    dzp = np.maximum(0, np.abs(z)-300.)/300.
    sigma_np = 0.15*(dxp**3+dyp**3+dzp**3)

    # ---- move everything to device
    DT = _dtype()
    def dev(a):
        a = xp.asarray(a)
        return a.astype(DT) if a.dtype.kind == 'f' else a
    op = dict(Dr=dev(Dr),Ds=dev(Ds),Dt=dev(Dt),LIFT=dev(ref['LIFT']),
              rx=dev(rx),ry=dev(ry),rz=dev(rz),sx=dev(sx),sy=dev(sy),sz=dev(sz),
              tx=dev(tx),ty=dev(ty),tz=dev(tz),
              nx=dev(nx),ny=dev(ny),nz=dev(nz),Fscale=dev(Fscale),
              vmapM=dev(vmapM.ravel()),vmapP=dev(vmapP.ravel()),
              mapB=dev(mapB),K=K,Np=Np,Nfp=Nfp)

    m_dev = dev(metal)
    xm = dev(x[metal]); ym = dev(y[metal]); zm = dev(z[metal])
    gL = 1.0/np.sqrt(1.0-SRC_V**2)
    def E_inc(t_sim):
        z_e = SRC_V*(t_sim - T_CENTER_FS*c_phys)
        dx = xm-SRC_X0; dy = ym-SRC_Y0; dz = zm-z_e
        Rd = (dx*dx + dy*dy + (gL*dz)**2 + SIGMA_E**2)**1.5
        return (Q_AMP*gL*dx/Rd, Q_AMP*gL*dy/Rd, Q_AMP*gL*dz/Rd)

    phys = dict(metal_elems=m_dev, wp2=WP2, gamma=GAMMA,
                sigma=dev(sigma_np), E_inc=E_inc)
    rhs = make_rhs(op, phys)

    F   = [xp.zeros((K,Np), dtype=DT) for _ in range(9)]
    res = [xp.zeros((K,Np), dtype=DT) for _ in range(9)]

    rk4a = xp.asarray([0.0,-0.4178904745,-1.192151694,-1.697784692,-1.514183444], dtype=DT)
    rk4b = xp.asarray([0.1496590219,0.3792103129,0.8229550293,0.6994504559,0.1530572479], dtype=DT)

    # ---- EELS: Ez along the ELECTRON TRAJECTORY LINE (paper Eq. 3/5)
    # trajectory is the line (x0, y0, z) for z in the physical domain
    NZ = 512
    z_line = np.linspace(-900.0, 900.0, NZ)
    traj = np.stack([np.full(NZ,SRC_X0), np.full(NZ,SRC_Y0), z_line], axis=1)
    cent = np.stack([x.mean(1), y.mean(1), z.mean(1)], axis=1)
    # nearest element to each trajectory sample (built once on CPU)
    from scipy.spatial import cKDTree
    tree = cKDTree(cent)
    _, traj_elem = tree.query(traj)
    traj_elem_dev = dev(np.asarray(traj_elem, dtype=np.int64))
    nrec    = nsteps//REC_EVERY + 1
    Ez_line = np.zeros((nrec, NZ), dtype=np.float32)

    # ---- WIRE-AXIS probe (paper Fig. 2f): a line PARALLEL to the nanowire,
    # just outside its surface. SPP wave packets propagate along x, so this is
    # the cut that reveals the group velocity and the cap reflections. The
    # trajectory probe above runs along z and cannot show this.
    NXW = 512
    x_line = np.linspace(-750.0, 750.0, NXW)
    wire_pts = np.stack([x_line, np.full(NXW, R_WIRE + 5.0),
                         np.zeros(NXW)], axis=1)
    _, wire_elem = tree.query(wire_pts)
    wire_elem_dev = dev(np.asarray(wire_elem, dtype=np.int64))
    Ew_line = np.zeros((nrec, NXW), dtype=np.float32)
    tarr    = np.zeros(nrec)
    print(f"EELS recording every {REC_EVERY} steps -> {nrec} samples "
          f"({Ez_line.nbytes/1e6:.0f} MB)", flush=True)

    # ---- y~0 slice indices for snapshots (TRUE slice, not a projection)
    ytol = SLICE_YTOL
    for _ in range(8):
        sl = np.abs(y) < ytol
        if sl.sum() > 5000:
            break
        ytol *= 2.0
    if sl.sum() == 0:
        raise RuntimeError("empty y~0 slice — check mesh units/geometry")
    if ytol != SLICE_YTOL:
        print(f"NOTE: widened slice tolerance to {ytol:.1f} nm to capture nodes")
    sl_flat = np.where(sl.ravel())[0]
    x_sl = x.ravel()[sl_flat]; z_sl = z.ravel()[sl_flat]
    np.savez_compressed(os.path.join(outdir,"slice_coords.npz"),
                        x=x_sl, z=z_sl, ytol=ytol)
    sl_dev = dev(sl_flat)
    print(f"slice nodes (|y|<{ytol:.1f} nm): {sl_flat.size}", flush=True)

    snap_every = max(1, int(snap_every_fs*c_phys/dt))
    t0 = time.time()
    for step in range(nsteps):
        tsim = step*dt
        for i in range(5):
            rh = rhs(F[0],F[1],F[2],F[3],F[4],F[5],F[6],F[7],F[8], tsim)
            for q in range(9):
                res[q] = rk4a[i]*res[q] + dt*rh[q]
                F[q]   = F[q] + rk4b[i]*res[q]

        if step % REC_EVERY == 0:
            ir = step//REC_EVERY
            tarr[ir] = tsim
            Ez_line[ir] = to_cpu(F[5][traj_elem_dev, 0]).astype(np.float32)
            Ew = xp.sqrt(F[3][wire_elem_dev,0]**2 + F[4][wire_elem_dev,0]**2
                         + F[5][wire_elem_dev,0]**2)
            Ew_line[ir] = to_cpu(Ew).astype(np.float32)

        if step % snap_every == 0:
            Emag = xp.sqrt(F[3]**2+F[4]**2+F[5]**2).reshape(-1)[sl_dev]
            np.savez_compressed(
                os.path.join(outdir, f"slice_{step:07d}.npz"),
                E=to_cpu(Emag).astype(np.float32), t_fs=tsim/c_phys)
            mE = float(to_cpu(xp.max(xp.abs(F[5]))))
            el = time.time()-t0
            rate = (step+1)/el
            eta = (nsteps-step)/max(rate,1e-9)/3600
            print(f"step {step}/{nsteps} t={tsim/c_phys:7.3f} fs "
                  f"max|Ez|={mE:.3e} {rate:.1f} step/s ETA {eta:.1f} h", flush=True)

    np.savez_compressed(os.path.join(outdir,"eels_line.npz"),
                        t=tarr, z=z_line, Ez=Ez_line, v=SRC_V, x0=SRC_X0,
                        t_center=T_CENTER_FS*c_phys)
    np.savez_compressed(os.path.join(outdir,"wire_line.npz"),
                        t=tarr, x=x_line, E=Ew_line,
                        t_center=T_CENTER_FS*c_phys)
    print(f"done in {(time.time()-t0)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh", nargs="?", default="paper_mesh.msh")
    ap.add_argument("--total-fs", type=float, default=25.0)
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--snap-fs", type=float, default=0.25)
    ap.add_argument("--cfl", type=float, default=C_CFL)
    ap.add_argument("--clip-pct", type=float, default=1.0,
                    help="ignore this %% of worst sliver elements in the CFL "
                         "(0 = strict minimum)")
    a = ap.parse_args()
    main(a.mesh, a.total_fs, a.outdir, a.snap_fs, a.cfl, a.clip_pct)
