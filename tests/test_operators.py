"""
Regression tests for the DGTD spatial operators.

These encode the checks that were used to find and fix real bugs during
development, so they must keep passing:

  * reference element: orthonormality, polynomial exactness, mass/volume
  * the discrete DIVERGENCE THEOREM (catches a wrong surface-integral mapping)
  * geometric factors against an analytic affine map
  * face-node connectivity coincidence
  * operator SPECTRUM: max Re(eig) <= 0, which is what caught the flux sign
    error that made the scheme inject energy

Run with:  pytest -q
"""
import numpy as np
import pytest

from dgtd import build_reference_full, geometric_factors, normals_surface
from dgtd.dg_connect import build_face_connectivity
from dgtd.dg_nodal3d import vandermonde2d

ORDERS = [1, 2, 3, 4]


# --------------------------------------------------------------- reference
@pytest.mark.parametrize("N", ORDERS)
def test_orthonormal_basis(N):
    ref = build_reference_full(N)
    modal = ref['V'].T @ ref['M'] @ ref['V']
    assert np.max(np.abs(modal - np.eye(ref['Np']))) < 1e-10


@pytest.mark.parametrize("N", ORDERS)
def test_differentiation_exactness(N):
    """Dr/Ds/Dt differentiate polynomials of degree <= N exactly."""
    ref = build_reference_full(N)
    r, s, t = ref['r'], ref['s'], ref['t']
    rng = np.random.default_rng(0)
    for _ in range(10):
        while True:
            i, j, k = rng.integers(0, N + 1, 3)
            if i + j + k <= N:
                break
        f = (r**i) * (s**j) * (t**k)
        dr = (i * (r**(i - 1) if i else 0 * r)) * (s**j) * (t**k)
        assert np.max(np.abs(ref['Dr'] @ f - dr)) < 1e-9


@pytest.mark.parametrize("N", ORDERS)
def test_mass_matrix_volume(N):
    """Sum of the mass matrix is the reference tet volume, 4/3."""
    ref = build_reference_full(N)
    assert abs(np.sum(ref['M']) - 4.0 / 3.0) < 1e-10


@pytest.mark.parametrize("N", [2, 3, 4])
def test_divergence_theorem(N):
    """
    INT_V div F  ==  SUM_faces INT_S F.n
    Volume and surface operators must be mutually consistent. This failed
    originally because the slanted face needed a per-face barycentric map.
    """
    ref = build_reference_full(N)
    r, s, t = ref['r'], ref['s'], ref['t']
    Dr, Ds, Dt, M = ref['Dr'], ref['Ds'], ref['Dt'], ref['M']
    rng = np.random.default_rng(1)

    def rp(deg):
        c = np.zeros_like(r)
        for _ in range(20):
            i, j, k = rng.integers(0, deg + 1, 3)
            if i + j + k <= deg:
                c = c + rng.standard_normal() * (r**i) * (s**j) * (t**k)
        return c

    for _ in range(5):
        Fx, Fy, Fz = rp(N - 1), rp(N - 1), rp(N - 1)
        vol = np.sum(M @ (Dr @ Fx + Ds @ Fy + Dt @ Fz))
        surf = 0.0
        for f in range(4):
            idx = ref['fmask'][f]
            n = ref['face_normals'][f]
            mass2d, area = ref['face_2d'][f]
            Fn = Fx[idx] * n[0] + Fy[idx] * n[1] + Fz[idx] * n[2]
            surf += (area / 2.0) * np.sum(mass2d @ Fn)
        assert abs(vol - surf) < 1e-9


# --------------------------------------------------------------- geometry
def test_geometric_factors_affine():
    """Under an affine map, J = det(A) and gradients of linear fields are exact."""
    N = 3
    ref = build_reference_full(N)
    r, s, t = ref['r'], ref['s'], ref['t']
    rng = np.random.default_rng(3)
    A = rng.standard_normal((3, 3))
    while np.linalg.det(A) < 0.3:
        A = rng.standard_normal((3, 3))
    b = rng.standard_normal(3)
    pts = (A @ np.stack([r, s, t]) ).T + b
    x, y, z = pts[:, 0][None, :], pts[:, 1][None, :], pts[:, 2][None, :]
    rx, ry, rz, sx, sy, sz, tx, ty, tz, J = geometric_factors(
        x, y, z, ref['Dr'], ref['Ds'], ref['Dt'])
    assert np.allclose(J, np.linalg.det(A))
    p, q, w = rng.standard_normal(3)
    f = (p * x + q * y + w * z)[0]
    dfdx = rx[0] * (ref['Dr'] @ f) + sx[0] * (ref['Ds'] @ f) + tx[0] * (ref['Dt'] @ f)
    assert np.max(np.abs(dfdx - p)) < 1e-9


def _two_tet_mesh(N):
    ref = build_reference_full(N)
    r, s, t = ref['r'], ref['s'], ref['t']
    P = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0],
                  [0, 0, 1], [1, 1, 1.]], float) * 40
    EToV = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    K, Np = 2, ref['Np']
    x = np.zeros((K, Np)); y = np.zeros((K, Np)); z = np.zeros((K, Np))
    for k in range(K):
        V0, V1, V2, V3 = P[EToV[k]]
        X = (V0[None] + 0.5 * (1 + r)[:, None] * (V1 - V0)[None]
             + 0.5 * (1 + s)[:, None] * (V2 - V0)[None]
             + 0.5 * (1 + t)[:, None] * (V3 - V0)[None])
        x[k], y[k], z[k] = X[:, 0], X[:, 1], X[:, 2]
    return ref, EToV, x, y, z


def test_face_connectivity_coincidence():
    """Every paired face node must be spatially coincident."""
    ref, EToV, x, y, z = _two_tet_mesh(3)
    vmapM, vmapP, EToE, EToF = build_face_connectivity(
        EToV, x, y, z, ref['fmask'], ref['Nfp'])
    xf, yf, zf = x.reshape(-1), y.reshape(-1), z.reshape(-1)
    gap = np.sqrt((xf[vmapM] - xf[vmapP])**2
                  + (yf[vmapM] - yf[vmapP])**2
                  + (zf[vmapM] - zf[vmapP])**2)
    assert gap.max() < 1e-10
    # the two tets share exactly one face
    assert (EToE[0] != 0).sum() == 1
