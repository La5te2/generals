#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python_bin="${PYTHON:-python}"
export PYTHONDONTWRITEBYTECODE=1
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$PWD/.jax-cache}"
if [[ "${ALLOW_CPU:-0}" != 1 ]]; then
    "$python_bin" -B -c 'import jax; assert any(d.platform == "gpu" for d in jax.devices()), "GPU required; ALLOW_CPU=1 enables a small CPU check"'
fi
exec "$python_bin" -u -B ppo.py "$@"
