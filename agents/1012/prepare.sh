#!/usr/bin/env bash
# Convert bounded portions of BOTH sources, deduplicating into the same split.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python_bin="${PYTHON:-python}"
limit="${1:-1000}"
[[ "$limit" =~ ^[0-9]+$ ]] || { echo 'Usage: bash prepare.sh [new_games_per_source; 0=all]' >&2; exit 2; }
export JAX_PLATFORMS=cpu
export PYTHONDONTWRITEBYTECODE=1
mkdir -p logs
"$python_bin" -u -B data.py data/raw/high_rank.parquet --limit "$limit" \
    --output data/processed --report data/high-rank-conversion.json 2>&1 | tee -a logs/prepare.log
"$python_bin" -u -B data.py data/raw/replays_prod.tar.gz --limit "$limit" \
    --output data/processed --report data/archive-conversion.json 2>&1 | tee -a logs/prepare.log
