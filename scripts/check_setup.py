"""
check_setup.py — verify everything is in place before a long run.

    python check_setup.py [paper_mesh.msh]

Checks imports, the GPU backend, the module files, and (if the mesh is
present) that it loads with sane units and a plausible metal-element count.
Exits non-zero if something is wrong.
"""
import sys, os, importlib

ok = True
def good(m): print(f"  [ok]   {m}")
def bad(m):
    global ok; ok = False
    print(f"  [FAIL] {m}")

print("\n--- python & core packages ---")
print(f"  python {sys.version.split()[0]}  ({sys.executable})")
for mod in ("numpy", "scipy", "matplotlib"):
    try:
        m = importlib.import_module(mod)
        good(f"{mod} {getattr(m,'__version__','?')}")
    except Exception as e:
        bad(f"{mod}: {e}")

print("\n--- scipy pieces the solver actually uses ---")
try:
    from scipy.spatial import cKDTree           # trajectory sampling
    from scipy.interpolate import griddata      # plotting
    good("scipy.spatial.cKDTree + scipy.interpolate.griddata (needs qhull)")
except Exception as e:
    bad(f"scipy qhull-backed pieces: {e}")

print("\n--- GPU backend ---")
try:
    import cupy
    n = cupy.cuda.runtime.getDeviceCount()
    a = cupy.arange(10); s = int(a.sum())     # actually exercise the runtime
    good(f"cupy {cupy.__version__}, {n} device(s), test sum={s}")
except Exception as e:
    print(f"  [note] cupy unavailable -> solver falls back to NumPy/CPU: {e}")

print("\n--- solver modules ---")
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
for mod in ("dgtd.dg_nodal3d", "dgtd.dg_geom", "dgtd.dg_connect"):
    try:
        importlib.import_module(mod); good(f"{mod}.py")
    except Exception as e:
        bad(f"{mod}.py: {e}")
try:
    from dgtd import build_reference_full
    r = build_reference_full(3)
    good(f"reference element p=3: Np={r['Np']}, Nfp={r['Nfp']}")
except Exception as e:
    bad(f"reference element: {e}")

print("\n--- gmsh (needed only to read/build the mesh) ---")
try:
    import gmsh
    gmsh.initialize(); v = gmsh.option.getString("General.Version"); gmsh.finalize()
    good(f"gmsh {v}")
except Exception as e:
    bad(f"gmsh: {e}")

mesh = sys.argv[1] if len(sys.argv) > 1 else "paper_mesh.msh"
print(f"\n--- mesh: {mesh} ---")
if not os.path.exists(mesh):
    print(f"  [note] not found; skipping (generate with: python paper_mesh.py)")
else:
    try:
        from dgtd.solver import read_tets
        VX, EToV, metal = read_tets(mesh)
        good(f"{len(EToV)} tets, {len(VX)} vertices")
        xr = (VX[:,0].min(), VX[:,0].max())
        yr = (VX[:,1].min(), VX[:,1].max())
        print(f"         x range {xr[0]:.0f} .. {xr[1]:.0f} nm")
        print(f"         y range {yr[0]:.0f} .. {yr[1]:.0f} nm")
        if max(abs(xr[0]), abs(xr[1])) < 10:
            bad("coordinates look like micrometres, not nm — unit mismatch!")
        else:
            good("units look like nm")
        nm_ = int(metal.sum())
        if nm_ == 0:
            bad("zero metal elements — check the 'silver' physical group")
        else:
            good(f"{nm_} metal elements ({100*nm_/len(EToV):.1f}%)")
    except Exception as e:
        bad(f"mesh read: {e}")

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
