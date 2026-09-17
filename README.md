# DGTD Nanowire SPP

High-order nodal **discontinuous Galerkin time-domain** (DGTD) solver for
Maxwell's equations with **Drude** dispersive media, used to simulate the
excitation of **surface plasmon polaritons** (SPPs) on a silver nanowire by a
swift electron in an aloof (non-penetrating) trajectory — the
electron-energy-loss spectroscopy (EELS) configuration.

Reproduces the numerical experiment of

> W. Zhao *et al.*, *Time-domain study of surface plasmon polariton propagation
> in silver nanowires*, **Phys. Rev. B 113**, 085425 (2026).
> [doi:10.1103/gpqj-4v29](https://doi.org/10.1103/gpqj-4v29)

Runs on CUDA via CuPy, with an automatic NumPy fallback.

---

## Physics

| quantity | value |
|---|---|
| wire | silver, radius **15 nm**, length **1.2 µm**, hemispherical caps |
| permittivity | Drude, ħω_p = 9.17 eV, ħγ = 21 meV, ε_∞ = 1 |
| electron | v = 0.4c (≈47 keV), impact parameter b = 5 nm, x₀ = 0.36 µm |
| source | Gaussian charge, σ_e = 5 nm, **scattered-field** formulation |
| basis | nodal DG, polynomial order p = 3 (4 supported) |
| integrator | 5-stage low-storage Runge–Kutta (Carpenter–Kennedy) |
| flux | upwind, vacuum impedances (ε_∞ = 1 in both media) |

The grid carries only the **induced (scattered)** field. The electron's
analytic incident field never enters the grid; it drives the Drude
polarisation current inside the metal, which is the physically correct
coupling and keeps the vacuum background at exactly zero.

Materials are assigned **element-wise** from the mesh's `silver` physical
group. This matters: a per-node material mask puts a discontinuity *inside*
elements, which a nodal basis cannot represent, and radiates spurious current
from the metal surface.

---

## Install

```bash
conda env create -f environment.yml
conda activate dgtd
pip install -e .
```

`environment.yml` installs CuPy and a matching CUDA runtime from conda-forge,
so a system CUDA module is usually unnecessary. Set `cuda-version` to match
your driver (`nvidia-smi`, top right). Without CuPy everything still runs on
the CPU, just slower.

## Quick start

```bash
python scripts/check_setup.py          # verify imports, GPU, mesh
python scripts/make_mesh.py            # build paper_mesh.msh (needs gmsh)
python scripts/validate_run.py --fs 8  # short run + automatic physics checks
bash   scripts/run.sh                  # full run (TOTAL_FS=80 recommended)
python -m dgtd.postprocess out_DIR     # frames, spectrum, space-time maps
```

`validate_run.py` exits non-zero unless the field is finite, bounded, peaks
near the electron transit, does not grow afterwards, and stays localised on
the wire. Run it before committing to a long job.

## Choosing the run length

Two timescales set the useful window:

* **Plasmon ring-down** ħ/γ ≈ **31 fs**. Truncating earlier leaves the mode
  oscillating at the cut, producing Gibbs ringing and negative excursions in
  Γ_EELS.
* **Fabry–Pérot mode spacing.** With k_j = jπ/L and the cylinder dispersion,
  consecutive modes near 2.5 eV are separated by ≈ 0.17 eV. Resolving them
  needs 2πħ/T ≪ 0.17 eV, i.e. **T ≳ 80 fs** (resolution 0.05 eV).

A 25 fs run has 0.165 eV resolution — the same as the mode spacing — so it
cannot distinguish real modes from ringing. Use `TOTAL_FS=80`.

Predicted mode energies for this geometry (fundamental n = 0 branch):

| j | λ_SPP (nm) | E_j (eV) |
|---|---|---|
| 8 | 300 | 2.08 |
| 9 | 267 | 2.27 |
| 10 | 240 | 2.45 |
| 11 | 218 | 2.62 |
| 12 | 200 | 2.78 |

## Reading the output

* `frames/` — |E_ind| on the y ≈ 0 slice, fixed colour scale, time measured
  from closest approach (the paper's convention).
* `wire_spacetime.png` — |E| on a line beside the wire vs time, the analogue of
  the paper's Fig. 2(f). **This is the cut that shows SPP propagation**:
  diagonal streaks at ≈ 0.5c, bending back at the end caps.
* `eels_spacetime.png` — field along the electron trajectory. Runs along **z**,
  so a vertical stripe near z = 0 is geometric and says nothing about
  propagation.
* `eels_spectrum.png` — Γ_EELS(ω). It is a loss probability, so it must be
  **non-negative**; a symmetric negative swing means the coupling efficiency
  has been phase-rotated (usually a time-origin error).

## Tests

```bash
pytest -q
```

Twenty regression tests covering the reference element (orthonormality,
polynomial exactness, mass/volume, the **discrete divergence theorem**),
geometric factors against an analytic affine map, face-node coincidence, and
the **operator spectrum**. The spectral test is the important one: it asserts
max Re(λ) ≤ 0, and it is what originally exposed a flux sign error that passed
both flux-consistency and single-sample energy checks while still injecting
energy.

## Performance

| lever | effect |
|---|---|
| `--clip-pct 1.0` | timestep from the 1st-percentile insphere, not the worst sliver |
| `REC_EVERY = 20` | avoids a GPU→CPU sync every step |
| `FP32 = True` | large speedup on RTX cards (fp64 runs at 1/32 rate) |
| mesh optimisation | `Mesh.OptimizeThreshold` + repeated Netgen passes |

The explicit timestep is set by the **smallest** element, so mesh quality
translates directly into wall-clock time. `make_mesh.py` reports the achieved
insphere statistics; aim for a minimum above ~1.5 nm.

## Layout

```
src/dgtd/
  dg_nodal3d.py   reference tetrahedron: Jacobi basis, Vandermonde, mass, LIFT
  dg_geom.py      geometric factors, face normals, surface Jacobians
  dg_connect.py   face-node connectivity (vmapM / vmapP)
  solver.py       time integration, Drude ADE, electron source, recording
  postprocess.py  frames, EELS spectrum, space-time maps
scripts/          mesh generation, validation, setup check, run wrappers
tests/            regression tests
```

## Licence

MIT — see [LICENSE](LICENSE).

## Citing

If this code contributes to published work, please cite the paper above for
the physics, and note the numerical method follows

> K. Busch, M. König, J. Niegemann, *Discontinuous Galerkin methods in
> nanophotonics*, Laser & Photonics Reviews **5**, 773 (2011).
