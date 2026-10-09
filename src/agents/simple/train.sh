#!/usr/bin/env bash
# launch this experiment with the active Python environment; training arguments pass through unchanged.
set -euo pipefail

scriptDir="${BASH_SOURCE[0]%/*}"
if [[ "$scriptDir" == "${BASH_SOURCE[0]}" ]]; then scriptDir=.; fi
scriptDir="$(cd -- "$scriptDir" && pwd -P)"
python="${PYTHON:-python3}"

if ! command -v "$python" >/dev/null 2>&1; then
    printf 'Python not found: %s. Activate your environment or set PYTHON to its executable.\n' "$python" >&2
    exit 127
fi

# keep sampling and optimization on the GPU; trailing arguments override these defaults.
# exec preserves direct signal delivery; -u streams cloud logs immediately and -B avoids bytecode caches.
exec "$python" -B -u "$scriptDir/train.py" --device cuda:0 --microbatch 128 --inference-batch 256 "$@"
