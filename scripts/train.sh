#!/usr/bin/env bash
set -euo pipefail
scriptDir="${BASH_SOURCE[0]%/*}"
if [[ "$scriptDir" == "${BASH_SOURCE[0]}" ]]; then scriptDir=.; fi
cd -- "$scriptDir/.."
"${PYTHON:-python3}" -B -u src/agents/nebula/train.py "$@"
