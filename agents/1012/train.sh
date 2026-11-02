#!/usr/bin/env bash
# Supervised training only. Resume explicitly to avoid silently reusing weights.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python_bin="${PYTHON:-python}"
export PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
if [[ "${ALLOW_CPU:-0}" != 1 ]]; then
    "$python_bin" -B -c 'import jax; assert any(d.platform == "gpu" for d in jax.devices()), "No JAX GPU: run setup.sh, or set ALLOW_CPU=1 for a local check"'
fi
exec "$python_bin" -u -B train.py --data data/processed --min-stars 70 \
    --batch 8 --epochs 1 --save-every 100 --log-every 25 "$@"
