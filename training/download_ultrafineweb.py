#!/usr/bin/env python3
"""Download only Ultra-FineWeb Chinese, with resumption and upstream SHA-256 checks."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
import fcntl
import hashlib
from itertools import count
import json
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
REPOSITORY = "openbmb/Ultra-FineWeb"
REVISION = "02c85641e3d19a854be2e09139c25adaa9518063"
PREFIX = "data/ultrafineweb_zh/"
DEFAULT_OUTPUT = ROOT / "training/data/raw/ultra-fineweb-zh"


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def fetch_manifest(output, revision):
    path = output / "download-manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest["repository"] != REPOSITORY or manifest["revision"] != revision:
            raise ValueError("Existing download belongs to another repository revision")
        return manifest
    url = f"https://huggingface.co/api/datasets/{REPOSITORY}/tree/{revision}/{PREFIX.rstrip('/')}?limit=1000"
    with urllib.request.urlopen(url, timeout=60) as response:
        files = json.load(response)
        if response.headers.get("Link"):
            raise ValueError("Unexpected pagination: refusing an incomplete manifest")
    entries = []
    for item in files:
        if item["type"] != "file":
            continue
        if not item["path"].startswith(PREFIX) or not item["path"].endswith(".parquet"):
            raise ValueError(f"Unexpected source file: {item['path']}")
        entries.append({"path": item["path"], "filename": Path(item["path"]).name,
                        "bytes": item["size"], "sha256": item["lfs"]["oid"]})
    if len(entries) != 256 or len({item["filename"] for item in entries}) != 256:
        raise ValueError("Expected exactly 256 unique Chinese source files")
    manifest = {"repository": REPOSITORY, "revision": revision, "split": "zh",
                "created_at": now(), "total_bytes": sum(x["bytes"] for x in entries), "files": entries}
    write_json(path, manifest)
    # Preserve the upstream data documentation and license alongside the files.
    for filename in ("README.md", "README_ZH.md", "LICENSE"):
        url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{revision}/{filename}"
        with urllib.request.urlopen(url, timeout=60) as response:
            (output / filename).write_bytes(response.read())
    return manifest


def download_file(entry, output, url, stop, reserve_bytes):
    target = output / entry["filename"]
    partial = target.with_suffix(target.suffix + ".part")
    if target.exists():
        if target.stat().st_size == entry["bytes"] and digest(target) == entry["sha256"]:
            return entry["filename"]
        raise ValueError(f"Existing completed file failed verification: {target}")
    if partial.exists() and partial.stat().st_size > entry["bytes"]:
        raise ValueError(f"Oversized partial download: {partial}")
    if partial.exists() and partial.stat().st_size == entry["bytes"]:
        if digest(partial) != entry["sha256"]:
            raise ValueError(f"Partial file failed verification: {partial}")
        partial.replace(target)
        return entry["filename"]
    error_path = output / (entry["filename"] + ".curl.log")
    for attempt in count():
        if stop.is_set():
            raise InterruptedError("Download stopped")
        if shutil.disk_usage(output).free < reserve_bytes + entry["bytes"]:
            raise OSError("Disk reserve reached; partial download retained")
        # Re-resolve the pinned file on retries to refresh expiring CDN links.
        request_url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(
            {"download": "true", "attempt": time.time_ns()})
        with error_path.open("ab") as errors:
            process = subprocess.Popen([
                "curl", "--location", "--fail", "--silent", "--show-error",
                "--connect-timeout", "30", "--speed-time", "180", "--speed-limit", "1024",
                "--continue-at", "-", "--output", str(partial), request_url,
            ], stdout=subprocess.DEVNULL, stderr=errors)
            while process.poll() is None:
                if stop.wait(1):
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise InterruptedError("Download stopped; partial retained")
        if process.returncode == 0:
            if partial.stat().st_size != entry["bytes"]:
                raise ValueError(f"Downloaded size mismatch: {partial}")
            if digest(partial) != entry["sha256"]:
                raise ValueError(f"Downloaded checksum mismatch: {partial}")
            partial.replace(target)
            return entry["filename"]
        # Overnight connection outages must not abandon an otherwise resumable
        # corpus. Keep reconnecting for transport errors; other failures retain
        # the bounded retry policy. SIGINT/SIGTERM still interrupt the backoff.
        transport_errors = {5, 6, 7, 16, 18, 28, 35, 52, 55, 56, 92}
        if process.returncode not in transport_errors and attempt >= 7:
            raise RuntimeError(f"Download failed after retries; see {error_path}")
        with error_path.open("a") as errors:
            errors.write(f"{now()} retry={attempt + 1} curl_exit={process.returncode}\n")
        if stop.wait(min(60, 2 ** min(attempt, 6))):
            raise InterruptedError("Download stopped; partial retained")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--reserve-gib", type=float, default=32)
    args = parser.parse_args()
    if args.workers < 1 or args.reserve_gib < 0:
        parser.error("workers must be positive and reserve nonnegative")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".download.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stop = threading.Event()
    for name in (signal.SIGINT, signal.SIGTERM):
        signal.signal(name, lambda *_: stop.set())
    manifest = fetch_manifest(output, args.revision)
    entries = manifest["files"]
    reserve_bytes = int(args.reserve_gib * 2 ** 30)
    def present_bytes():
        total = 0
        for item in entries:
            target = output / item["filename"]
            partial = target.with_suffix(target.suffix + ".part")
            try:
                total += min(item["bytes"], (target if target.exists() else partial).stat().st_size)
            except FileNotFoundError:
                pass
        return total
    initial_bytes = present_bytes()
    remaining = manifest["total_bytes"] - initial_bytes
    if shutil.disk_usage(output).free < remaining + reserve_bytes:
        raise OSError(f"Need {remaining / 2**30:.1f} GiB plus {args.reserve_gib:g} GiB reserve")
    verified_path = output / "verified-files.json"
    verified = {}
    started = time.monotonic()
    def status(stage, **details):
        total = present_bytes()
        seconds = time.monotonic() - started
        value = {"stage": stage, "updated_at": now(), "repository": REPOSITORY,
                 "revision": args.revision, "verified_files": len(verified), "total_files": len(entries),
                 "downloaded_bytes": total, "total_bytes": manifest["total_bytes"],
                 "session_bytes_per_second": (total - initial_bytes) / max(1, seconds),
                 "session_seconds": seconds, **details}
        write_json(output / "download-status.json", value)
        print(json.dumps(value), flush=True)
    status("downloading")
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            pending = {}
            for item in entries:
                url = f"https://huggingface.co/datasets/{REPOSITORY}/resolve/{args.revision}/{item['path']}"
                pending[executor.submit(download_file, item, output, url, stop, reserve_bytes)] = item
            last_status = time.monotonic()
            try:
                while pending:
                    done, _ = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
                    for future in done:
                        item = pending.pop(future)
                        future.result()
                        verified[item["filename"]] = {"bytes": item["bytes"], "sha256": item["sha256"]}
                        write_json(verified_path, {"revision": args.revision, "files": verified})
                    if done or time.monotonic() - last_status >= 30:
                        status("downloading")
                        last_status = time.monotonic()
            except BaseException:
                stop.set()
                for future in pending:
                    future.cancel()
                raise
        status("complete")
    except BaseException as error:
        status("stopped" if isinstance(error, InterruptedError) else "failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
