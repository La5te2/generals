#!/usr/bin/env bash
set -euo pipefail
scriptDir="${BASH_SOURCE[0]%/*}"
if [[ "$scriptDir" == "${BASH_SOURCE[0]}" ]]; then scriptDir=.; fi
scriptDir="$(cd -- "$scriptDir" && pwd -P)"
python="${PYTHON:-python3}"
if ! command -v "$python" >/dev/null 2>&1; then
    printf 'Python not found: %s. Activate your environment or set PYTHON to its executable.\n' "$python" >&2
    exit 127
fi
# Keep user-supplied output/resume paths relative to the calling directory.
exec "$python" -B -u "$scriptDir/train.py" "$@"
