"""
Operator-spectrum stability test.

This is the check that originally exposed a flux/curl sign error: the scheme
passed flux-consistency and a single-sample energy check while still being
unstable. Only the eigenvalue spectrum revealed it.

A correct semidiscrete DGTD Maxwell operator with an upwind flux is
dissipative, so every eigenvalue must satisfy Re(lambda) <= 0. Any positive
real part means the scheme injects energy and will blow up regardless of how
small the timestep is.
"""
import numpy as np
import pytest

from dgtd import build_reference_full, geometric_factors, normals_surface
from dgtd.dg_connect import build_face_connectivity


def _assemble_operator(N=2, pec_boundary=True):
    """Build the dense semidiscrete Maxwell operator on a small 2-tet mesh."""
    ref = build_reference_full(N)
    r, s, t = ref['r'], ref['s'], ref['t']
    Np, Nfp = ref['Np'], ref['Nfp']
    Dr, Ds, Dt, LIFT = ref['Dr'], ref['Ds'], ref['Dt'], ref['LIFT']

    P = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                  [0, 0, 1], [1, 1, 1.]], float) * 40
    EToV = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    K = 2
    x = np.zeros((K, Np)); y = np.zeros((K, Np)); z = np.zeros((K, Np))
    for k in range(K):
        V0, V1, V2, V3 = P[EToV[k]]
        X = (V0[None] + 0.5 * (1 + r)[:, None] * (V1 - V0)[None]
             + 0.5 * (1 + s)[:, None] * (V2 - V0)[None]
             + 0.5 * (1 + t)[:, None] * (V3 - V0)[None])
        x[k], y[k], z[k] = X[:, 0], X[:, 1], X[:, 2]

    rx, ry, rz, sx, sy, sz, tx, ty, tz, J = geometric_factors(x, y, z, Dr, Ds, Dt)
    nx, ny, nz, sJ, Fscale, _ = normals_surface(x, y, z, Dr, Ds, Dt,
                                                ref['fmask'], Nfp)
    vmapM, vmapP, EToE, EToF = build_face_connectivity(
        EToV, x, y, z, ref['fmask'], Nfp)
    vmapM, vmapP = vmapM.ravel(), vmapP.ravel()
    mapB = (vmapM.reshape(K, 4 * Nfp) == vmapP.reshape(K, 4 * Nfp))

    def curl(ux, uy, uz):
        uxr, uxs, uxt = ux @ Dr.T, ux @ Ds.T, ux @ Dt.T
        uyr, uys, uyt = uy @ Dr.T, uy @ Ds.T, uy @ Dt.T
        uzr, uzs, uzt = uz @ Dr.T, uz @ Ds.T, uz @ Dt.T
        cx = (ry * uzr + sy * uzs + ty * uzt) - (rz * uyr + sz * uys + tz * uyt)
        cy = (rz * uxr + sz * uxs + tz * uxt) - (rx * uzr + sx * uzs + tx * uzt)
        cz = (rx * uyr + sx * uys + tx * uyt) - (ry * uxr + sy * uxs + ty * uxt)
        return cx, cy, cz

    def rhs(F):
        Hx, Hy, Hz, Ex, Ey, Ez = F
        cHx, cHy, cHz = curl(Hx, Hy, Hz)
        cEx, cEy, cEz = curl(Ex, Ey, Ez)

        def jump(u):
            uf = u.reshape(-1)
            return (uf[vmapM] - uf[vmapP]).reshape(K, 4 * Nfp)
        dHx, dHy, dHz = jump(Hx), jump(Hy), jump(Hz)
        dEx, dEy, dEz = jump(Ex), jump(Ey), jump(Ez)

        if pec_boundary:
            ExM = Ex.reshape(-1)[vmapM].reshape(K, 4 * Nfp)
            EyM = Ey.reshape(-1)[vmapM].reshape(K, 4 * Nfp)
            EzM = Ez.reshape(-1)[vmapM].reshape(K, 4 * Nfp)
            dEx = np.where(mapB, 2.0 * ExM, dEx)
            dEy = np.where(mapB, 2.0 * EyM, dEy)
            dEz = np.where(mapB, 2.0 * EzM, dEz)
            dHx = np.where(mapB, 0.0, dHx)
            dHy = np.where(mapB, 0.0, dHy)
            dHz = np.where(mapB, 0.0, dHz)

        ndH = nx * dHx + ny * dHy + nz * dHz
        ndE = nx * dEx + ny * dEy + nz * dEz
        nxEx = ny * dEz - nz * dEy
        nxEy = nz * dEx - nx * dEz
        nxEz = nx * dEy - ny * dEx
        nxHx = ny * dHz - nz * dHy
        nxHy = nz * dHx - nx * dHz
        nxHz = nx * dHy - ny * dHx

        fHx = -nxEx - (dHx - ndH * nx)
        fHy = -nxEy - (dHy - ndH * ny)
        fHz = -nxEz - (dHz - ndH * nz)
        fEx = nxHx - (dEx - ndE * nx)
        fEy = nxHy - (dEy - ndE * ny)
        fEz = nxHz - (dEz - ndE * nz)

        half = 0.5 * Fscale
        return [cEx + (half * fHx) @ LIFT.T,
                cEy + (half * fHy) @ LIFT.T,
                cEz + (half * fHz) @ LIFT.T,
                -cHx + (half * fEx) @ LIFT.T,
                -cHy + (half * fEy) @ LIFT.T,
                -cHz + (half * fEz) @ LIFT.T]

    n = 6 * K * Np

    def unpack(v):
        out, o = [], 0
        for _ in range(6):
            out.append(v[o:o + K * Np].reshape(K, Np)); o += K * Np
        return out

    A = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n); e[j] = 1.0
        A[:, j] = np.concatenate([f.ravel() for f in rhs(unpack(e))])
    return A


@pytest.mark.parametrize("pec", [True, False])
def test_operator_is_dissipative(pec):
    """
    max Re(eig) must be <= 0 up to round-off. A genuinely wrong flux sign gives
    max Re(eig) of order the spectral radius (we measured +0.17 and +2.2 for
    the two wrong sign combinations), so the tolerance is scaled to rho: real
    errors exceed it by five orders of magnitude, while the eigenvalue
    round-off of this non-normal operator sits around 1e-6*rho.
    """
    A = _assemble_operator(N=2, pec_boundary=pec)
    ev = np.linalg.eigvals(A)
    max_re = ev.real.max()
    rho = np.abs(ev).max()
    assert max_re < 1e-4 * rho, (
        f"operator injects energy: max Re(eig) = {max_re:.3e} "
        f"(spectral radius {rho:.3e}). Check the upwind flux signs and the "
        "curl convention."
    )


def test_spectral_radius_finite():
    """The spectral radius sets the CFL limit; it must be finite and positive."""
    A = _assemble_operator(N=2)
    rho = np.abs(np.linalg.eigvals(A)).max()
    assert np.isfinite(rho) and rho > 0
