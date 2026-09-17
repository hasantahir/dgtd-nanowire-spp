"""
postprocess.py — make movie frames and the EELS spectrum from solver output.
Runs on the CPU (login node) after the GPU job finishes; no re-simulation.

  python3 postprocess.py out/            # frames + spectrum
  python3 postprocess.py out/ --frames   # frames only
"""
import numpy as np, os, sys, glob, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

hbar = 6.582119569e-16
fs   = 1e-15
c    = 299.792458


def make_frames(outdir, zoom=0.35, nx=700, nz=320, t_center_fs=2.0,
                dyn_range=3.0, fixed_vmax=None):
    """
    Frames on the y~0 slice.

    Two conventions matched to the paper:
      * time origin at CLOSEST APPROACH (paper: "t = 0 when the electron is at
        the closest approach"), so we subtract t_center_fs;
      * a FIXED colour range taken from the global maximum over all frames.
        Per-frame autoscaling makes the pre-arrival frames (where the field is
        ~1e-13, i.e. numerically zero) show pure round-off noise at full
        contrast, which looks like spurious field on the wire but is not.
    """
    co = np.load(os.path.join(outdir, "slice_coords.npz"))
    xs, zs = co['x']/1000.0, co['z']/1000.0        # nm -> um
    files = sorted(f for f in glob.glob(os.path.join(outdir, "slice_*.npz"))
                   if not f.endswith("slice_coords.npz"))
    print(f"{len(files)} snapshots; {xs.size} slice nodes")
    os.makedirs(os.path.join(outdir, "frames"), exist_ok=True)

    # --- pass 1: robust reference level for a FIXED colour scale.
    # The single global max is a poor anchor: at closest approach the field at
    # the electron is a brief, very intense spike, and anchoring there pushes
    # the (much weaker) SPP on the wire into the dark. Instead use the MEDIAN
    # over frames of each frame's 99.9th percentile, which tracks the level of
    # the structure you actually want to see.
    if fixed_vmax is None:
        levels = []
        for f in files:
            e = np.asarray(np.load(f)['E'], float)
            e = e[np.isfinite(e) & (e > 0)]
            if e.size:
                levels.append(np.log10(np.percentile(e, 99.9)))
        if levels:
            gmax = float(np.median(levels))
            # never clip the brightest structure entirely
            gmax = max(gmax, float(np.median(levels)))
        else:
            gmax = -3.0
    else:
        gmax = fixed_vmax
    vmin, vmax = gmax - dyn_range, gmax
    print(f"fixed colour range: [{vmin:.2f}, {vmax:.2f}] (log10)  "
          f"[override with --vmax / --dyn-range]")

    gx, gz = np.mgrid[-0.78:0.78:complex(0,nx), -zoom:zoom:complex(0,nz)]
    for f in files:
        d = np.load(f)
        E = np.asarray(d['E'], float); t_fs = float(d['t_fs']) - t_center_fs
        L = np.log10(E + 1e-30)
        g = griddata((xs, zs), L, (gx, gz), method='linear')
        fig, ax = plt.subplots(figsize=(10, 4.4))
        pc = ax.pcolormesh(gx, gz, g, shading='gouraud', cmap='viridis',
                           vmin=vmin, vmax=vmax)
        ax.add_patch(plt.Rectangle((-0.6, -0.015), 1.2, 0.03, fill=False,
                                   edgecolor='w', ls='--', lw=0.8, alpha=0.8))
        # electron position (moves in z at v = 0.4c, closest approach at t=0)
        z_e = 0.4*299.792458*(t_fs)/1000.0
        if abs(z_e) <= zoom:
            ax.plot(0.36, z_e, 'o', color='red', ms=5, zorder=5)
        ax.set_xlabel('x (µm)'); ax.set_ylabel('z (µm)')
        ax.set_title(f't = {t_fs:+.2f} fs')
        cb = fig.colorbar(pc, ax=ax, pad=0.02, fraction=0.03)
        cb.set_label(r'$\log_{10}|E^{\rm ind}|$ (arb.)')
        tag = os.path.basename(f).replace("slice_", "").replace(".npz", "")
        fig.savefig(os.path.join(outdir, "frames", f"f_{tag}.png"),
                    dpi=130, bbox_inches='tight')
        plt.close(fig)
    print("frames written to", os.path.join(outdir, "frames"))


def eels_spectrum(outdir, emin=0.5, emax=4.0, ne=800, taper_frac=0.35):
    """
    Paper Eq. (3)/(5):
        eta(w) = (e/2 pi w) INT dz Ez_ind(z,w) exp(-i w (z - z0)/v)
        Gamma_EELS(w) = (2/hbar) Re[eta(w)]

    TWO corrections over the naive transform:

    * TIME ORIGIN. The paper sets t = 0 at the electron's closest approach and
      z0 = its position then. The solver records t from the START of the run,
      with closest approach at t_center. Using the raw t in the time-FT leaves
      a phase exp(i w t_center) that ROTATES the complex eta, mixing Im into
      Re -- which makes Gamma swing negative. Gamma_EELS is a loss probability
      and must be non-negative, so a sign-alternating spectrum is the signature
      of exactly this error.

    * TAIL TAPER. A finite time window truncates the still-ringing plasmon,
      producing Gibbs ringing at roughly the resolution scale
      (2 pi hbar / T ~ 0.17 eV for T = 25 fs). A one-sided taper on the tail
      suppresses it; taper_frac=0 disables.
    """
    d = np.load(os.path.join(outdir, "eels_line.npz"))
    t = d['t']; zl = d['z']; Ez = d['Ez']; v = float(d['v'])
    n = np.count_nonzero(t) + 1
    t, Ez = t[:n], Ez[:n]
    t_c = float(d['t_center']) if 't_center' in d.files else 0.0
    if t_c == 0.0:
        print("  NOTE: no t_center in file (older run) -- spectrum phase may be "
              "rotated; re-run the solver to record it.")
    tp = t - t_c                                  # time from closest approach
    dt = tp[1]-tp[0]; dz = zl[1]-zl[0]

    W = np.ones(len(tp))
    if taper_frac > 0:
        k = int(taper_frac*len(tp))
        if k > 1:
            W[-k:] = 0.5*(1+np.cos(np.pi*np.arange(k)/k))
    Ezw_in = Ez*W[:, None]

    E_eV = np.linspace(emin, emax, ne)
    om = (E_eV/hbar)*fs/c
    spec = np.zeros(ne)
    for i, w in enumerate(om):
        Ezw = (Ezw_in*np.exp(1j*w*tp)[:, None]).sum(axis=0)*dt
        eta = (Ezw*np.exp(-1j*w*zl/v)).sum()*dz
        spec[i] = np.real(eta)/max(w, 1e-12)
    m = np.max(np.abs(spec)) or 1.0
    spec = spec/m

    neg = np.mean(spec < 0)
    print(f"EELS: {neg*100:.0f}% of the spectrum is negative "
          f"({'OK - residual ringing' if neg < 0.15 else 'HIGH - see notes'})")

    np.savez(os.path.join(outdir, "eels_spectrum.npz"), E_eV=E_eV, spec=spec)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(E_eV, spec, lw=1.8)
    ax.axhline(0, color='k', lw=0.6, alpha=0.5)
    ax.set_xlabel('Energy loss (eV)'); ax.set_ylabel(r'$\Gamma_{\rm EELS}$ (arb.)')
    ax.set_xlim(emin, emax); ax.grid(alpha=0.3)
    fig.savefig(os.path.join(outdir, "eels_spectrum.png"),
                dpi=200, bbox_inches='tight')
    plt.close(fig)

    # --- trajectory space-time map (along z). NOTE: the induced field is
    # geometrically peaked where the trajectory passes the wire (z ~ 0), so a
    # vertical stripe here is EXPECTED and says nothing about propagation.
    fig, ax = plt.subplots(figsize=(8, 5))
    ex = [zl.min()/1000, zl.max()/1000, tp.min()/c, tp.max()/c]
    ax.imshow(np.abs(Ez), aspect='auto', origin='lower', extent=ex, cmap='inferno')
    ax.set_xlabel('z (µm)'); ax.set_ylabel('t (fs, from closest approach)')
    ax.set_title(r'$|E_z^{\rm ind}|$ along the electron trajectory'
                 '\n(vertical stripe at z~0 is geometric, not a defect)')
    fig.savefig(os.path.join(outdir, "eels_spacetime.png"), dpi=180,
                bbox_inches='tight')
    plt.close(fig)


def wire_spacetime(outdir, vg_ref=0.5):
    """
    Paper Fig. 2(f): |E| on a line PARALLEL to the wire, vs time. This is the
    cut that reveals SPP propagation -- wave packets leave the excitation point
    and travel along x at the group velocity, reflecting off the end caps.
    A reference line at vg_ref*c is overlaid for slope comparison.
    """
    f = os.path.join(outdir, "wire_line.npz")
    if not os.path.exists(f):
        print("  (no wire_line.npz -- re-run the solver to record the "
              "wire-axis probe)")
        return
    d = np.load(f)
    t = d['t']; x = d['x']; E = d['E']
    n = np.count_nonzero(t) + 1
    t, E = t[:n], E[:n]
    t_c = float(d['t_center']) if 't_center' in d.files else 0.0
    tp = (t - t_c)/c

    L = np.log10(np.abs(E) + 1e-18)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ex = [x.min()/1000, x.max()/1000, tp.min(), tp.max()]
    vmax = np.nanpercentile(L, 99.5)
    im = ax.imshow(L, aspect='auto', origin='lower', extent=ex,
                   cmap='inferno', vmin=vmax-3.0, vmax=vmax)
    # group-velocity reference from the excitation point
    x0 = 0.36
    tt = np.linspace(0, tp.max(), 50)
    for sgn in (+1, -1):
        ax.plot(x0 + sgn*vg_ref*c*tt/1000.0, tt, 'w--', lw=1.0, alpha=0.7)
    ax.axvline(-0.6, color='cyan', lw=0.8, alpha=0.6)
    ax.axvline( 0.6, color='cyan', lw=0.8, alpha=0.6)
    ax.set_xlim(x.min()/1000, x.max()/1000)
    ax.set_xlabel('x along the wire (µm)')
    ax.set_ylabel('t (fs, from closest approach)')
    ax.set_title(f'|E| beside the wire — dashed = {vg_ref}c reference, '
                 'cyan = end caps')
    fig.colorbar(im, ax=ax, pad=0.02, fraction=0.03,
                 label=r'$\log_{10}|E|$')
    fig.savefig(os.path.join(outdir, "wire_spacetime.png"), dpi=180,
                bbox_inches='tight')
    plt.close(fig)
    print("wire-axis space-time map written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--frames", action="store_true")
    ap.add_argument("--spectrum", action="store_true")
    ap.add_argument("--zoom", type=float, default=0.35)
    ap.add_argument("--t-center", type=float, default=2.0,
                    help="fs of closest approach; frames are labelled "
                         "relative to it, as in the paper")
    ap.add_argument("--dyn-range", type=float, default=3.0,
                    help="decades shown below the reference level")
    ap.add_argument("--vmax", type=float, default=None,
                    help="set the top of the colour scale explicitly (log10); "
                         "overrides the automatic reference level")
    a = ap.parse_args()
    do_all = not (a.frames or a.spectrum)
    if a.frames or do_all:
        make_frames(a.outdir, zoom=a.zoom, t_center_fs=a.t_center,
                    dyn_range=a.dyn_range, fixed_vmax=a.vmax)
    if a.spectrum or do_all:
        eels_spectrum(a.outdir)
        wire_spacetime(a.outdir)
