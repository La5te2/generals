#!/usr/bin/env bash
set -euo pipefail

scriptDir="${BASH_SOURCE[0]%/*}"
if [[ "$scriptDir" == "${BASH_SOURCE[0]}" ]]; then scriptDir=.; fi
cd -- "$scriptDir/.."

# Git Bash and MSYS2 share the Windows tool discovery in build.bat.
if [[ "$OSTYPE" == msys* ]]; then
    MSYS_NO_PATHCONV=1 cmd.exe /d /c scripts\\build.bat "$@"
    exit 0
fi

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DVCPKG_MANIFEST_INSTALL=OFF "$@"
cmake --build build --parallel
