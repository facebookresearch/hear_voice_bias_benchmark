#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Download the HEAR Voice Bias Benchmark audio from GitHub Releases.

The parquet shards are attached to a GitHub Release rather than committed to the
repository, so `git clone` stays small and you fetch the ~13 GiB of audio only if you
want it — or just the config you need.

    python download.py                       # both configs -> ./data
    python download.py --config openqa       # one config only
    python download.py --out /path/to/dir    # somewhere else
    python download.py --limit 3             # first 3 shards of each config, to try it
    python download.py --verify-only         # re-check what is already on disk

Release assets are a flat namespace, so they are named `<config>-train-XXXXX-of-XXXXX`
and written back into `data/<config>/` so the two configs stay loadable separately.

Every asset is checked against the SHA256 in release_assets.json. A shard already
present and matching is skipped, so an interrupted run resumes cleanly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ASSET_LIST = Path(__file__).with_name("release_assets.json")
RELEASE_URL = "https://github.com/{repo}/releases/download/{tag}/{name}"
CHUNK = 1 << 20


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        while chunk := resp.read(CHUNK):
            out.write(chunk)
    # Rename only after a complete read, so a killed run never leaves a valid-looking file.
    tmp.replace(dest)


def inspect(dest: Path, asset: dict) -> str:
    """'ok', or why the local copy cannot be trusted."""
    if not dest.exists():
        return "missing"
    actual = dest.stat().st_size
    if actual != asset["size_bytes"]:
        return f"SIZE MISMATCH ({actual:,} != {asset['size_bytes']:,} expected)"
    if sha256(dest) != asset["sha256"]:
        return "CHECKSUM MISMATCH"
    return "ok"


def handle(asset: dict, repo: str, tag: str, out_dir: Path, verify_only: bool) -> tuple[str, str]:
    # asset["path"] is data/<config>/<file>; keep that layout under --out.
    dest = out_dir / Path(asset["path"]).relative_to("data")

    state = inspect(dest, asset)
    if state == "ok":
        return asset["name"], "ok (cached)"
    if verify_only:
        return asset["name"], state

    url = RELEASE_URL.format(repo=repo, tag=tag, name=asset["name"])
    try:
        fetch(url, dest)
    except urllib.error.HTTPError as e:
        return asset["name"], f"HTTP {e.code}"
    except Exception as e:  # network failures should not kill the whole run
        return asset["name"], f"{type(e).__name__}: {e}"

    got = sha256(dest)
    if got != asset["sha256"]:
        return asset["name"], f"CHECKSUM MISMATCH (got {got[:12]}...)"
    return asset["name"], "ok"


def main() -> int:
    spec = json.loads(ASSET_LIST.read_text())
    configs = sorted(spec["configs"])

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data"), help="output directory")
    ap.add_argument("--config", choices=configs, action="append", help="repeatable; default all")
    ap.add_argument("--limit", type=int, help="first N shards of each config")
    ap.add_argument("--workers", type=int, default=4, help="parallel downloads")
    ap.add_argument("--verify-only", action="store_true", help="check local files, download nothing")
    args = ap.parse_args()

    wanted = set(args.config or configs)
    by_config: dict[str, list[dict]] = defaultdict(list)
    for a in spec["assets"]:
        if a["config"] in wanted:
            by_config[a["config"]].append(a)

    assets = []
    for cfg in sorted(by_config):
        shards = sorted(by_config[cfg], key=lambda a: a["name"])
        assets.extend(shards[: args.limit] if args.limit else shards)

    if not assets:
        print("nothing selected", file=sys.stderr)
        return 2

    total_gb = sum(a["size_bytes"] for a in assets) / 1024**3
    action = "verifying" if args.verify_only else "downloading"
    print(f"{action} {len(assets)} shard(s) across {len(wanted)} config(s), "
          f"{total_gb:.2f} GiB -> {args.out}")

    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [
            pool.submit(handle, a, spec["repo"], spec["tag"], args.out, args.verify_only)
            for a in assets
        ]
        for i, fut in enumerate(as_completed(futs), 1):
            name, status = fut.result()
            if not status.startswith("ok"):
                failures.append((name, status))
            print(f"  [{i}/{len(assets)}] {name}  {status}")

    if failures:
        print(f"\n{len(failures)} failed:", file=sys.stderr)
        for name, status in failures:
            print(f"  {name}: {status}", file=sys.stderr)
        return 1

    print(f"\nall {len(assets)} shard(s) verified")
    if not args.limit:
        print("\nLoad with:")
        print("    from datasets import load_dataset")
        for cfg in sorted(wanted):
            print(f"    {cfg:6} = load_dataset('parquet', "
                  f"data_files='{args.out}/{cfg}/*.parquet', split='train')")
    return 0


if __name__ == "__main__":
    sys.exit(main())
