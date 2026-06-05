"""Fetch + unpack the BIRD-SQL dev set into data/bird/ (spec §6).

Downloads the dev zip, extracts it, and extracts the nested dev_databases.zip
that BIRD ships inside. The loader (src.tasks) discovers dev.json / dev_databases
by globbing, so we don't depend on the exact top-level folder name (which BIRD
has changed across releases).

The full dev set is several GB. For first runs, restrict to a few databases via
config.BIRD_DB_SUBSET / the run_experiment --db-subset flag rather than re-downloading.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

from src.config import BIRD_DEV_URL, BIRD_DIR


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"already downloaded: {dest} ({dest.stat().st_size/1e6:.1f} MB)")
        return
    print(f"downloading {url}\n  -> {dest}")
    try:
        import requests  # type: ignore

        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            done = 0
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = 100 * done / total
                        print(f"\r  {done/1e6:7.1f} / {total/1e6:.1f} MB ({pct:5.1f}%)",
                              end="", flush=True)
            print()
    except ImportError:
        import urllib.request

        urllib.request.urlretrieve(url, dest)  # noqa: S310 — known benchmark URL


def _unzip(zip_path: Path, out_dir: Path) -> None:
    print(f"unzipping {zip_path.name} -> {out_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download + unpack the BIRD dev set.")
    ap.add_argument("--url", default=BIRD_DEV_URL)
    ap.add_argument("--dir", default=str(BIRD_DIR))
    ap.add_argument("--keep-zip", action="store_true", help="don't delete the downloaded zip")
    args = ap.parse_args()

    bird_dir = Path(args.dir)
    zip_path = bird_dir / "dev.zip"

    _download(args.url, zip_path)
    _unzip(zip_path, bird_dir)

    # BIRD nests the databases inside a second zip — extract any we find.
    for nested in bird_dir.rglob("*.zip"):
        if nested == zip_path or "__MACOSX" in nested.parts:
            continue
        _unzip(nested, nested.parent)

    # Remove macOS resource-fork junk so discovery can't grab the wrong dir.
    import shutil

    for junk in bird_dir.rglob("__MACOSX"):
        shutil.rmtree(junk, ignore_errors=True)

    def _ok(p: Path) -> bool:
        return "__MACOSX" not in p.parts

    dev_json = next((p for p in sorted(bird_dir.rglob("dev.json")) if _ok(p)), None)
    db_root = next((p for p in sorted(bird_dir.rglob("dev_databases")) if _ok(p)), None)
    if not dev_json or not db_root:
        print(
            "WARNING: could not locate dev.json and/or dev_databases/ after extraction.\n"
            f"  dev.json: {dev_json}\n  dev_databases: {db_root}\n"
            "Inspect the extracted tree under "
            f"{bird_dir} and adjust if BIRD changed its layout.",
            file=sys.stderr,
        )
        sys.exit(1)

    n_dbs = len(list(db_root.iterdir()))
    print(f"\nOK. dev.json: {dev_json}\n    dev_databases: {db_root} ({n_dbs} databases)")
    if not args.keep_zip:
        zip_path.unlink(missing_ok=True)
    print("Next: python -m scripts.smoke_test")


if __name__ == "__main__":
    main()
