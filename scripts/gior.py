"""Download a generals.io replay-page URL or a direct .gior URL without changing its bytes."""
# python tools/gior.py https://generals.io/replays/bLG7W2qA7
import argparse
import os
import re
import sys
import tempfile
from pathlib import Path
from urllib.error import URLError
from urllib.parse import unquote, urlsplit, urlunsplit
from urllib.request import Request, urlopen

LIMIT = 32 * 1024 * 1024


def source(url):
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
        raise ValueError("Use an HTTP(S) replay URL without embedded credentials")
    path = unquote(parts.path)
    page = re.fullmatch(r"/replays/([A-Za-z0-9_-]{1,128})/?", path)
    if page and parts.hostname in ("generals.io", "www.generals.io", "bot.generals.io"):
        region = "bot" if parts.hostname == "bot.generals.io" else "na"
        name = page[1] + ".gior"
        return f"https://generalsio-replays-{region}.s3.amazonaws.com/{name}", name
    name = path.rsplit("/", 1)[-1]
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}\.gior", name, re.IGNORECASE):
        raise ValueError("Expected a generals.io/replays/ID page or a direct .gior URL")
    return urlunsplit(parts._replace(fragment="")), name


def download(url, output=None, *, timeout=30, force=False):
    address, name = source(url)
    if timeout <= 0:
        raise ValueError("Timeout must be positive")
    if re.fullmatch(r"(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])\.gior", name, re.IGNORECASE):
        name = "replay-" + name
    destination = Path(output) if output else Path(name)
    if destination.suffix.lower() != ".gior":
        raise ValueError("Output filename must end in .gior")
    if destination.exists() and not force:
        raise FileExistsError(f"Already exists: {destination}; choose another path or use --force")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        request = Request(address, headers={"User-Agent": "Generals-Replay-Downloader", "Accept-Encoding": "identity"})
        with urlopen(request, timeout=timeout) as response:
            if response.headers.get_content_type() in ("text/html", "application/xhtml+xml", "application/json", "application/xml", "text/xml"):
                raise ValueError("Server returned a page or error document instead of a .gior file")
            length = response.headers.get("Content-Length")
            if length is not None and (int(length) <= 0 or int(length) > LIMIT):
                raise ValueError("Replay is empty or exceeds 32 MiB")
            with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".gior-", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                size = 0
                while chunk := response.read(64 * 1024):
                    if size == 0 and chunk.lstrip().startswith((b"<", b"{")):
                        raise ValueError("Server returned text instead of compressed replay bytes")
                    size += len(chunk)
                    if size > LIMIT:
                        raise ValueError("Replay exceeds 32 MiB")
                    stream.write(chunk)
            if size == 0 or size % 2 or (length is not None and size != int(length)):
                raise ValueError("Empty, truncated or invalid .gior byte length")
        if force:
            os.replace(temporary, destination)
        else:
            # create the name atomically without overwriting a file created during the download.
            os.link(temporary, destination)
        return destination, size
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--force", action="store_true", help="Replace an existing output file")
    args = parser.parse_args()
    try:
        path, size = download(args.url, args.output, timeout=args.timeout, force=args.force)
    except (OSError, URLError, ValueError) as error:
        print(f"Download failed: {error}", file=sys.stderr)
        return 1
    print(f"{path.resolve()} ({size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
