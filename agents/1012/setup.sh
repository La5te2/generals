#!/usr/bin/env bash
# Install in the selected Python environment; no system package manager is run.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python_bin="${PYTHON:-python}"
backend="${1:-cuda12}"
case "$backend" in
    cuda12|cuda13) jax_package="jax[$backend]==0.11.2" ;;
    cpu) jax_package="jax==0.11.2" ;;
    *) echo 'Usage: bash setup.sh [cuda12|cuda13|cpu]' >&2; exit 2 ;;
esac
command -v node >/dev/null || { echo 'Install Node.js before setup (historical replay conversion).' >&2; exit 1; }
command -v git >/dev/null || { echo 'Install Git before setup (pinned environment dependency).' >&2; exit 1; }
"$python_bin" -B -c 'import sys; assert sys.version_info >= (3, 12), "Python 3.12+ required"'
"$python_bin" -m pip install -r requirements.txt "$jax_package"
"$python_bin" -B - "$backend" <<'PY'
import sys
import jax
devices = jax.devices()
print({"jax": jax.__version__, "devices": [str(d) for d in devices]}, flush=True)
if sys.argv[1] != "cpu" and not any(d.platform == "gpu" for d in devices):
    raise SystemExit("GPU JAX is unavailable; fix the driver/runtime before training.")
PY
