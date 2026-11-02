#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python_bin="${PYTHON:-python}"
backend="${1:-cuda12}"
case "$backend" in
    cuda12|cuda13) jax_package="jax[$backend]==0.11.2" ;;
    cpu) jax_package="jax==0.11.2" ;;
    *) echo 'Usage: bash setup.sh [cuda12|cuda13|cpu]' >&2; exit 2 ;;
esac
"$python_bin" -B -c 'import sys; assert sys.version_info >= (3, 12), "Python 3.12+ required"'
"$python_bin" -m pip install -r requirements.txt "$jax_package"
"$python_bin" -B - "$backend" <<'PY'
import sys
import jax
print({"jax": jax.__version__, "devices": [str(x) for x in jax.devices()]}, flush=True)
if sys.argv[1] != "cpu" and not any(x.platform == "gpu" for x in jax.devices()):
    raise SystemExit("GPU JAX is unavailable; check the driver/runtime.")
PY
