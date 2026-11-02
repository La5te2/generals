"""Create a new standalone upload directory, omitting local runtimes and weights."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    destination = args.destination.resolve()
    if destination.exists() or destination.is_relative_to(root):
        parser.error("Choose a new destination outside the source directory")
    files = [
        *root.glob("*.py"),
        *root.glob("*.sh"),
        root / "replay.js",
        root / "requirements.txt",
        root / "README.md",
        root / "CLOUD.md",
    ]
    for folder in ("vendor", "data/raw", "data/pilot"):
        files.extend(p for p in (root / folder).rglob("*") if p.is_file())
    files.append(root / "data/inventory.json")
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = []
    for path in sorted(files):
        relative = path.relative_to(root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        with path.open("rb") as stream:
            expected = hashlib.file_digest(stream, "sha256").hexdigest()
        with target.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise OSError(f"Copy verification failed: {relative}")
        manifest.append(
            {
                "file": relative.as_posix(),
                "bytes": target.stat().st_size,
                "sha256": actual,
            }
        )
    (destination / "bundle.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "folder": str(destination),
                "files": len(manifest),
                "bytes": sum(row["bytes"] for row in manifest),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
