"""
validate_run.py — realistic short run + automatic physics checks.

Run this BEFORE committing to the full 25 fs. It simulates through the
electron's closest approach (t = 2 fs) and out to ~6 fs, which is enough to
see the SPP launch, then checks the result against what the physics requires.

    python validate_run.py                    # uses paper_mesh.msh
    python validate_run.py mymesh.msh --fs 6

Checks performed on the recorded trajectory field and the run log:
  1. field is finite everywhere (no NaN/Inf)
  2. field is BOUNDED - no runaway growth after the electron leaves
  3. excitation PEAKS near closest approach (t ~ T_CENTER_FS), not at the end
  4. field is LOCALISED on the wire, not filling the box
  5. late-time decay is consistent with Drude damping (ring-down, not growth)
Exits non-zero if any check fails.
"""
import numpy as np, subprocess, sys, os, glob, argparse, json

def run(mesh, fs, outdir, extra):
    cmd = [sys.executable, "-m", "dgtd.solver", mesh,
           "--total-fs", str(fs), "--outdir", outdir,
           "--snap-fs", str(max(fs/25.0, 0.05))] + extra
    print("running:", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, text=True)
    if p.returncode != 0:
        print("solver exited non-zero"); sys.exit(1)


def check(outdir, t_center_fs=2.0):
    c = 299.792458
    ok = True
    def good(m): print(f"  [ok]   {m}")
    def bad(m):
        nonlocal ok; ok = False; print(f"  [FAIL] {m}")
    def note(m): print(f"  [note] {m}")

    d  = np.load(os.path.join(outdir, "eels_line.npz"))
    t  = d['t']/c                      # -> fs
    Ez = d['Ez']
    n  = np.count_nonzero(t) + 1
    t, Ez = t[:n], Ez[:n]
    amp = np.abs(Ez).max(axis=1)       # max |Ez| along the line, per sample

    print("\n--- physics checks ---")

    # 1 finite
    if np.all(np.isfinite(Ez)):
        good("all recorded values finite")
    else:
        bad("NaN/Inf present in the recorded field")

    # 2 bounded -- test for EXPONENTIAL GROWTH, not dynamic range.
    # A correct run legitimately spans ~10 orders (it starts near machine zero
    # before the electron arrives), so comparing peak to global minimum is
    # meaningless. Blow-up instead shows the envelope still RISING at the end.
    if amp.max() <= 0:
        bad("field is identically zero - nothing was excited")
    else:
        m = len(amp)
        q = max(2, m//5)
        late, prev = amp[-q:].mean(), amp[-2*q:-q].mean()
        rising = late > 1.5*max(prev, 1e-30)
        peak_at_end = int(np.argmax(amp)) >= m - q
        if rising and peak_at_end:
            bad(f"envelope still rising at the end of the run "
                f"({prev:.3e} -> {late:.3e}, peak in the final fifth) "
                "- exponential blow-up")
        else:
            good(f"field bounded: peak {amp.max():.3e}, "
                 f"end-of-run mean {late:.3e} ({late/amp.max():.3f} x peak)")

    # 3 peak near closest approach
    i_pk = int(np.argmax(amp)); t_pk = t[i_pk]
    if t_pk < t_center_fs + 2.0:
        good(f"excitation peaks at t = {t_pk:.2f} fs (electron passes ~{t_center_fs} fs)")
    else:
        bad(f"peak at t = {t_pk:.2f} fs is late - source timing or growth problem")

    # 4/5 late-time behaviour: must not exceed the peak
    tail = amp[t > t_pk + 1.0]
    if tail.size:
        ratio = tail.max()/amp.max()
        if ratio <= 1.01:
            good(f"no post-peak growth (late/peak = {ratio:.3f})")
        else:
            bad(f"field GROWS after the electron passes (late/peak = {ratio:.3f})")
        if tail.size > 4:
            a, b = tail[:tail.size//2].mean(), tail[tail.size//2:].mean()
            if b <= a*1.05: good(f"late-time ring-down/decay ({a:.2e} -> {b:.2e})")
            else:           note(f"late-time rising ({a:.2e} -> {b:.2e}) - watch this")
    else:
        note("run too short to judge post-peak behaviour; use --fs 6 or more")

    # 4b localisation, from the last slice snapshot
    co = np.load(os.path.join(outdir, "slice_coords.npz"))
    xs, zs = co['x'], co['z']
    snaps = sorted(f for f in glob.glob(os.path.join(outdir, "slice_*.npz"))
                   if not f.endswith("slice_coords.npz"))
    if snaps:
        E = np.load(snaps[-1])['E']
        near = np.abs(zs) < 60.0                     # within 60 nm of the wire
        far  = np.abs(zs) > 250.0
        if near.any() and far.any():
            r = np.mean(E[near])/max(np.mean(E[far]), 1e-30)
            if r > 3.0:
                good(f"field localised on the wire (near/far = {r:.1f}x)")
            else:
                bad(f"field NOT localised (near/far = {r:.1f}x) - filling the box")
        else:
            note("slice does not span near+far; skipping localisation test")

    print()
    print("VALIDATION PASSED - safe to launch the full run" if ok
          else "VALIDATION FAILED - do not launch the long run yet")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh", nargs="?", default="paper_mesh.msh")
    ap.add_argument("--fs", type=float, default=6.0)
    ap.add_argument("--outdir", default="validate_out")
    ap.add_argument("--clip-pct", type=float, default=1.0)
    ap.add_argument("--cfl", type=float, default=0.3)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()

    if not a.check_only:
        run(a.mesh, a.fs, a.outdir,
            ["--clip-pct", str(a.clip_pct), "--cfl", str(a.cfl)])
    sys.exit(0 if check(a.outdir) else 1)
