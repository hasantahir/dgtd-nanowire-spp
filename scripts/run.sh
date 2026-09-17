#!/usr/bin/env bash
# run_dgtd.sh — run the DGTD nanowire job directly on a CUDA workstation
# (no scheduler needed).  Survives logout via nohup; logs to a timestamped file.
#
#   conda activate dgtd
#   bash run_dgtd.sh                 # foreground, short test
#   nohup bash run_dgtd.sh &         # background, full run
#
set -euo pipefail

TOTAL_FS=${TOTAL_FS:-25.0}
SNAP_FS=${SNAP_FS:-0.25}
CFL=${CFL:-0.3}
MESH=${MESH:-paper_mesh.msh}

STAMP=$(date +%Y%m%d_%H%M%S)
OUT="out_${STAMP}"
LOG="dgtd_${STAMP}.log"
mkdir -p "$OUT"

{
  echo "host      : $(hostname)"
  echo "date      : $(date)"
  echo "conda env : ${CONDA_DEFAULT_ENV:-<none>}"
  echo "python    : $(which python)"
  nvidia-smi --query-gpu=name,memory.total,driver_version \
             --format=csv,noheader 2>/dev/null || echo "nvidia-smi not found"
  python - <<'PY'
try:
    import cupy
    print("cupy      :", cupy.__version__,
          cupy.cuda.runtime.getDeviceCount(), "device(s)")
except Exception as e:
    print("cupy      : NOT AVAILABLE ->", e, "(will fall back to NumPy/CPU)")
PY
  echo "outdir    : $OUT"
  echo "----------------------------------------"

  python ho_solver_gpu.py "$MESH" \
      --total-fs "$TOTAL_FS" \
      --outdir   "$OUT" \
      --snap-fs  "$SNAP_FS" \
      --cfl      "$CFL"

  echo "--- post-processing ---"
  python postprocess.py "$OUT"
  echo "results in $OUT"
} 2>&1 | tee "$LOG"
