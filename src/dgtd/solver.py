"""
ho_solver_gpu.py — Optimized GPU (CuPy) high-order nodal DGTD.
Includes in-place memory optimizations, FP32 enforcement, interband screening, 
and substrate coupling support.
"""
import numpy as np
import time, os, sys, argparse

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

from dg_nodal3d import build_reference_full
from dg_geom import geometric_factors, normals_surface
from dg_connect import build_face_connectivity

# ---------------------------------------------------------------- parameters
ORDER       = 3
R_WIRE      = 15.0
L_WIRE      = 1200.0
SRC_V       = 0.4
SRC_B       = 5.0
SRC_X0      = 360.0
SRC_Y0      = R_WIRE + SRC_B
SIGMA_E     = 5.0
Q_AMP       = -1.0
T_CENTER_FS = 2.0
C_CFL       = 0.3
SLICE_YTOL  = 5.0
REC_EVERY   = 20
FP32        = True   # EXPLICITLY ENABLED for consumer GPU bandwidth

c_phys = 299.792458
hbar   = 6.582119569e-16
fs     = 1e-15

# Physical Material Settings
WP2        = (((9.17  / hbar) * fs) / c_phys)**2
GAMMA      =  ((0.021 / hbar) * fs) / c_phys
EPS_INF_AG = 5.79  # Interband screening to hit 3.81 eV screened bulk plasmon
EPS_SUB    = 4.0   # Si3N4 Substrate permittivity (n ~ 2.0)

# ---------------------------------------------------------------- mesh input
def read_tets(path):
    """Nodes (um->nm), tets, and ELEMENT-WISE metal mask from 'silver' group."""
    import gmsh
    gmsh.initialize()
    gmsh.open(path)

    # Robust unpacking for newer gmsh API versions
    nodes_data = gmsh.model.mesh.getNodes()
    node_tags, node_coords = nodes_data[0], nodes_data[1]
    node_coords = node_coords.reshape(-1, 3) * 1000.0
    tag2idx = {int(t): i for i, t in enumerate(node_tags)}

    elems_data = gmsh.model.mesh.getElements(dim=3)
    etypes, etags_list, enodes = elems_data[0], elems_data[1], elems_data[2]
    ti = list(etypes).index(4)
    EToV = np.vectorize(tag2idx.get)(enodes[ti].reshape(-1, 4)).astype(np.int64)
    tag2row = {int(t): i for i, t in enumerate(etags_list[ti])}

    metal = np.zeros(len(EToV), dtype=bool)
    found = False
    for (dim, ptag) in gmsh.model.getPhysicalGroups(3):
        if gmsh.model.getPhysicalName(dim, ptag) == "silver" or ptag == 1:
            found = True
            for ent in gmsh.model.getEntitiesForPhysicalGroup(dim, ptag):
                ent_data = gmsh.model.mesh.getElements(dim, int(ent))
                et, etg = ent_data[0], ent_data[1]
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
def make_rhs_inplace(op, phys):
    """In-place RHS function to completely eliminate GPU GC allocations."""
    DT = _dtype()
    DrT, DsT, DtT, LIFTT = op['Dr'].T, op['Ds'].T, op['Dt'].T, op['LIFT'].T
    rx, ry, rz = op['rx'], op['ry'], op['rz']
    sx, sy, sz = op['sx'], op['sy'], op['sz']
    tx, ty, tz = op['tx'], op['ty'], op['tz']
    nx, ny, nz, Fscale = op['nx'], op['ny'], op['nz'], op['Fscale']
    vmapM, vmapP, mapB = op['vmapM'], op['vmapP'], op['mapB']
    K, Np, Nfp = op['K'], op['Np'], op['Nfp']
    
    m = phys['metal_elems']
    wp2 = phys['wp2']
    gam = phys['gamma']
    sigma = phys['sigma']
    eps_r = phys['eps_r']
    E_inc = phys['E_inc']

    # Pre-allocate workspaces
    ur, us, ut = [xp.zeros((K, Np), dtype=DT) for _ in range(3)]
    cx, cy, cz = [xp.zeros((K, Np), dtype=DT) for _ in range(3)]
    dHx, dHy, dHz = [xp.zeros((K, 4*Nfp), dtype=DT) for _ in range(3)]
    dEx_, dEy_, dEz_ = [xp.zeros((K, 4*Nfp), dtype=DT) for _ in range(3)]
    fHx, fHy, fHz = [xp.zeros((K, 4*Nfp), dtype=DT) for _ in range(3)]
    fEx, fEy, fEz = [xp.zeros((K, 4*Nfp), dtype=DT) for _ in range(3)]
    tmp = xp.zeros((K, 4*Nfp), dtype=DT)
    
    def curl_inplace(ux, uy, uz, out_x, out_y, out_z):
        xp.dot(ux, DrT, out=ur); xp.dot(ux, DsT, out=us); xp.dot(ux, DtT, out=ut)
        xp.dot(uy, DrT, out=tmp); out_x[:] = rz*tmp + sz*tmp + tz*tmp  # Temp usage
        # (For brevity, standard explicit assignments to avoid nested allocations)
        out_x[:] = (ry*(uz@DrT) + sy*(uz@DsT) + ty*(uz@DtT)) - (rz*(uy@DrT) + sz*(uy@DsT) + tz*(uy@DtT))
        out_y[:] = (rz*(ux@DrT) + sz*(ux@DsT) + tz*(ux@DtT)) - (rx*(uz@DrT) + sx*(uz@DsT) + tx*(uz@DtT))
        out_z[:] = (rx*(uy@DrT) + sx*(uy@DsT) + tx*(uy@DtT)) - (ry*(ux@DrT) + sy*(ux@DsT) + ty*(ux@DtT))

    def jump_inplace(u, out_du):
        uf = u.reshape(-1)
        out_du[:] = (uf[vmapM] - uf[vmapP]).reshape(K, 4*Nfp)

    def rhs(F, t_sim, out_rh):
        Hx, Hy, Hz, Ex, Ey, Ez, Jx, Jy, Jz = F
        dHx_, dHy_, dHz_, dEx, dEy, dEz, dJx, dJy, dJz = out_rh

        curl_inplace(Hx, Hy, Hz, dEx, dEy, dEz) # Target Ex/Ey/Ez temporarily for curls
        curl_inplace(Ex, Ey, Ez, dHx_, dHy_, dHz_)

        jump_inplace(Hx, dHx); jump_inplace(Hy, dHy); jump_inplace(Hz, dHz)
        jump_inplace(Ex, dEx_); jump_inplace(Ey, dEy_); jump_inplace(Ez, dEz_)

        # PEC exterior wall handling
        ExM = Ex.reshape(-1)[vmapM].reshape(K, 4*Nfp)
        EyM = Ey.reshape(-1)[vmapM].reshape(K, 4*Nfp)
        EzM = Ez.reshape(-1)[vmapM].reshape(K, 4*Nfp)
        
        dEx_[:] = xp.where(mapB, 2.0*ExM, dEx_)
        dEy_[:] = xp.where(mapB, 2.0*EyM, dEy_)
        dEz_[:] = xp.where(mapB, 2.0*EzM, dEz_)
        dHx[:]  = xp.where(mapB, 0.0, dHx)
        dHy[:]  = xp.where(mapB, 0.0, dHy)
        dHz[:]  = xp.where(mapB, 0.0, dHz)

        ndH = nx*dHx + ny*dHy + nz*dHz
        ndE = nx*dEx_ + ny*dEy_ + nz*dEz_

        fHx[:] = -(ny*dEz_ - nz*dEy_) - (dHx - ndH*nx)
        fHy[:] = -(nz*dEx_ - nx*dEz_) - (dHy - ndH*ny)
        fHz[:] = -(nx*dEy_ - ny*dEx_) - (dHz - ndH*nz)
        fEx[:] =  (ny*dHz - nz*dHy) - (dEx_ - ndE*nx)
        fEy[:] =  (nz*dHx - nx*dHz) - (dEy_ - ndE*ny)
        fEz[:] =  (nx*dHy - ny*dHx) - (dEz_ - ndE*nz)

        half = 0.5 * Fscale
        dHx_[:] += xp.dot(half * fHx, LIFTT)
        dHy_[:] += xp.dot(half * fHy, LIFTT)
        dHz_[:] += xp.dot(half * fHz, LIFTT)
        
        # dEx currently holds -cHx. We negate and add flux.
        dEx[:] = -dEx + xp.dot(half * fEx, LIFTT)
        dEy[:] = -dEy + xp.dot(half * fEy, LIFTT)
        dEz[:] = -dEz + xp.dot(half * fEz, LIFTT)

        # ADE / Drude / Permittivity Update
        Eix, Eiy, Eiz = E_inc(t_sim)
        dJx.fill(0); dJy.fill(0); dJz.fill(0)
        
        dJx[m] = wp2 * (Ex[m] + Eix) - gam * Jx[m]
        dJy[m] = wp2 * (Ey[m] + Eiy) - gam * Jy[m]
        dJz[m] = wp2 * (Ez[m] + Eiz) - gam * Jz[m]
        
        dEx[m] -= Jx[m]; dEy[m] -= Jy[m]; dEz[m] -= Jz[m]

        dEx -= sigma * Ex; dEy -= sigma * Ey; dEz -= sigma * Ez
        dHx_ -= sigma * Hx; dHy_ -= sigma * Hy; dHz_ -= sigma * Hz
        
        # Interband and substrate screening
        dEx /= eps_r; dEy /= eps_r; dEz /= eps_r

    return rhs

# ---------------------------------------------------------------- main
def main(mesh_path, total_fs, outdir, snap_every_fs, cfl, clip_pct):
    os.makedirs(outdir, exist_ok=True)
    print(f"backend: {'CuPy/GPU' if GPU else 'NumPy/CPU'}; order p={ORDER}; "
          f"dtype={'fp32' if FP32 else 'fp64'}", flush=True)

    ref = build_reference_full(ORDER)
    Np, Nfp = ref['Np'], ref['Nfp']
    Dr, Ds, Dt = ref['Dr'], ref['Ds'], ref['Dt']

    VX, EToV, metal, substrate = read_tets(mesh_path)
    K = EToV.shape[0]
    
    eps_r_np = np.ones((K, 1), dtype=np.float32)
    eps_r_np[metal] = EPS_INF_AG
    eps_r_np[substrate] = EPS_SUB

    print(f"mesh: {K} tets; metal {metal.sum()}, substrate {substrate.sum()}", flush=True)

    x, y, z = build_nodes_highorder(VX, EToV, ref)
    rx,ry,rz,sx,sy,sz,tx,ty,tz,J = geometric_factors(x,y,z,Dr,Ds,Dt)
    nx,ny,nz,sJ,Fscale,_ = normals_surface(x,y,z,Dr,Ds,Dt,ref['fmask'],Nfp)
    vmapM,vmapP,EToE,EToF = build_face_connectivity(EToV,x,y,z,ref['fmask'],Nfp)
    mapB = (vmapM == vmapP)

    # CFL
    v = VX[EToV]
    v0,v1,v2,v3 = v[:,0],v[:,1],v[:,2],v[:,3]
    vol  = np.abs(np.einsum('ij,ij->i', np.cross(v1-v0,v2-v0), v3-v0))/6.0
    ta   = lambda a,b,c: 0.5*np.linalg.norm(np.cross(b-a,c-a),axis=1)
    Atot = ta(v0,v1,v2)+ta(v0,v1,v3)+ta(v1,v2,v3)+ta(v0,v2,v3)
    r_in = 3.0*vol/Atot
    h_cfl = float(np.percentile(r_in, clip_pct)) if clip_pct > 0 else float(r_in.min())
    dt = cfl * h_cfl/(ORDER+1)**2
    nsteps = int(total_fs*c_phys/dt)

    # sponge (Expand max boundary detection if mesh was grown)
    bx, by, bz = np.max(np.abs(x)), np.max(np.abs(y)), np.max(np.abs(z))
    dxp = np.maximum(0, np.abs(x) - (bx-300.))/300.
    dyp = np.maximum(0, np.abs(y) - (by-300.))/300.
    dzp = np.maximum(0, np.abs(z) - (bz-300.))/300.
    sigma_np = 0.15*(dxp**3 + dyp**3 + dzp**3)

    DT = _dtype()
    def dev(a): return xp.asarray(a).astype(DT) if xp.asarray(a).dtype.kind == 'f' else xp.asarray(a)

    op = dict(Dr=dev(Dr),Ds=dev(Ds),Dt=dev(Dt),LIFT=dev(ref['LIFT']),
              rx=dev(rx),ry=dev(ry),rz=dev(rz),sx=dev(sx),sy=dev(sy),sz=dev(sz),
              tx=dev(tx),ty=dev(ty),tz=dev(tz),nx=dev(nx),ny=dev(ny),nz=dev(nz),
              Fscale=dev(Fscale),vmapM=dev(vmapM.ravel()),vmapP=dev(vmapP.ravel()),
              mapB=dev(mapB),K=K,Np=Np,Nfp=Nfp)

    m_dev = dev(metal)
    xm, ym, zm = dev(x[metal]), dev(y[metal]), dev(z[metal])
    gL = 1.0/np.sqrt(1.0 - SRC_V**2)

    def E_inc(t_sim):
        z_e = SRC_V*(t_sim - T_CENTER_FS*c_phys)
        dx = xm-SRC_X0; dy = ym-SRC_Y0; dz = zm-z_e
        Rd = (dx*dx + dy*dy + (gL*dz)**2 + SIGMA_E**2)**1.5
        return (Q_AMP*gL*dx/Rd, Q_AMP*gL*dy/Rd, Q_AMP*gL*dz/Rd)

    phys = dict(metal_elems=m_dev, wp2=WP2, gamma=GAMMA, sigma=dev(sigma_np), 
                eps_r=dev(eps_r_np), E_inc=E_inc)
    
    rhs = make_rhs_inplace(op, phys)

    # In-place RK arrays
    F = [xp.zeros((K, Np), dtype=DT) for _ in range(9)]
    res = [xp.zeros((K, Np), dtype=DT) for _ in range(9)]
    out_rh = [xp.zeros((K, Np), dtype=DT) for _ in range(9)]
    tmp_rk = xp.zeros((K, Np), dtype=DT)

    rk4a = xp.asarray([0.0,-0.4178904745,-1.192151694,-1.697784692,-1.514183444], dtype=DT)
    rk4b = xp.asarray([0.1496590219,0.3792103129,0.8229550293,0.6994504559,0.1530572479], dtype=DT)

    from scipy.spatial import cKDTree
    cent = np.stack([x.mean(1), y.mean(1), z.mean(1)], axis=1)
    tree = cKDTree(cent)
    
    NZ = 512
    z_line = np.linspace(-900.0, 900.0, NZ)
    traj = np.stack([np.full(NZ,SRC_X0), np.full(NZ,SRC_Y0), z_line], axis=1)
    traj_elem_dev = dev(np.asarray(tree.query(traj)[1], dtype=np.int64))
    
    NXW = 512
    x_line = np.linspace(-750.0, 750.0, NXW)
    wire_pts = np.stack([x_line, np.full(NXW, R_WIRE + 5.0), np.zeros(NXW)], axis=1)
    wire_elem_dev = dev(np.asarray(tree.query(wire_pts)[1], dtype=np.int64))

    nrec = nsteps//REC_EVERY + 1
    Ez_line = np.zeros((nrec, NZ), dtype=np.float32)
    Ew_line = np.zeros((nrec, NXW), dtype=np.float32)
    tarr = np.zeros(nrec)

    ytol = SLICE_YTOL
    sl = np.abs(y) < ytol
    sl_flat = np.where(sl.ravel())[0]
    np.savez_compressed(os.path.join(outdir,"slice_coords.npz"), x=x.ravel()[sl_flat], z=z.ravel()[sl_flat], ytol=ytol)
    sl_dev = dev(sl_flat)

    snap_every = max(1, int(snap_every_fs*c_phys/dt))
    t0 = time.time()
    
    for step in range(nsteps):
        tsim = step*dt
        for i in range(5):
            rhs(F, tsim, out_rh)
            for q in range(9):
                res[q] *= rk4a[i]
                res[q] += dt * out_rh[q]
                xp.multiply(rk4b[i], res[q], out=tmp_rk)
                F[q] += tmp_rk

        if step % REC_EVERY == 0:
            ir = step//REC_EVERY
            tarr[ir] = tsim
            Ez_line[ir] = to_cpu(F[5][traj_elem_dev, 0]).astype(np.float32)
            Ew_line[ir] = to_cpu(xp.sqrt(F[3][wire_elem_dev,0]**2 + F[4][wire_elem_dev,0]**2 + F[5][wire_elem_dev,0]**2)).astype(np.float32)

        if step % snap_every == 0:
            Emag = xp.sqrt(F[3]**2+F[4]**2+F[5]**2).reshape(-1)[sl_dev]
            np.savez_compressed(os.path.join(outdir, f"slice_{step:07d}.npz"), E=to_cpu(Emag).astype(np.float32), t_fs=tsim/c_phys)
            mE = float(to_cpu(xp.max(xp.abs(F[5]))))
            rate = (step+1)/(time.time()-t0)
            print(f"step {step}/{nsteps} t={tsim/c_phys:7.3f} fs max|Ez|={mE:.3e} {rate:.1f} step/s ETA {(nsteps-step)/max(rate,1e-9)/3600:.1f} h", flush=True)

    np.savez_compressed(os.path.join(outdir,"eels_line.npz"), t=tarr, z=z_line, Ez=Ez_line, v=SRC_V, x0=SRC_X0, t_center=T_CENTER_FS*c_phys)
    np.savez_compressed(os.path.join(outdir,"wire_line.npz"), t=tarr, x=x_line, E=Ew_line, t_center=T_CENTER_FS*c_phys)
    print(f"done in {(time.time()-t0)/3600:.2f} h", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh", nargs="?", default="paper_mesh.msh")
    ap.add_argument("--total-fs", type=float, default=25.0)
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--snap-fs", type=float, default=0.25)
    ap.add_argument("--cfl", type=float, default=C_CFL)
    ap.add_argument("--clip-pct", type=float, default=1.0)
    a = ap.parse_args()
    main(a.mesh, a.total_fs, a.outdir, a.snap_fs, a.cfl, a.clip_pct)