"""Read-only, reproducible corpus sampling and raw-web provenance recovery.

Uniform byte proposals accepted with probability 8/line_bytes give every JSONL
row the same inclusion probability, avoiding the usual byte-seek length bias.
Each compact training row has at least five JSON values and is longer than 8 bytes.
"""
from __future__ import annotations

import argparse
import bisect
import collections
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "training/artifacts/corpus-quality-20260916"
MANIFEST = ROOT / "training/data/processed/web-mix-40m-192ch-16conv-20260916/manifest.json"
GROUPS = [("base", 0, 256, 42217584), ("synthetic", 256, 384, 31904356),
          ("web_first", 384, 512, 20000000), ("web_added", 512, 640, 20000000)]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def flags(text, target):
    gap = target + 1
    left, right = text[:gap], text[gap:]
    out = []
    if min(len(left), len(right)) <= 2:
        out.append("short_side")
    if len(text) > 128:
        out.append("long_input")
    if not (0 < gap < len(text)):
        out.append("invalid_target")
    if re.search(r"https?://|www\.|</?\w+|&(?:nbsp|amp|lt|gt);|\ufffd", text):
        out.append("markup_or_encoding")
    if re.search(r"点击|登录|注册|版权|转载请|联系电话|咨询热线|加微信|免费咨询|立即购买|阅读原文|相关阅读|上一篇|下一篇", text):
        out.append("web_boilerplate_cue")
    if re.search(r"(.)\1{4,}", text):
        out.append("repeated_character")
    if re.search(r"[A-Za-z]$", left) and re.match(r"[A-Za-z]", right):
        out.append("latin_boundary")
    if left[-1:].isdecimal() and right[:1].isdecimal():
        out.append("numeric_boundary")
    return out


def containing_row(handle, pos):
    width = 1024
    while True:
        start = max(0, pos - width)
        handle.seek(start)
        prefix = handle.read(pos - start)
        newline = prefix.rfind(b"\n")
        if newline >= 0 or start == 0:
            offset = start + newline + 1 if newline >= 0 else 0
            handle.seek(offset)
            return offset, handle.readline()
        width *= 2


def sample(per_group=5000, review_per_group=250):
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_bytes = MANIFEST.read_bytes()
    manifest = json.loads(manifest_bytes)
    rng = random.Random(2026091601)
    rows, review, inventory = [], [], []
    started = time.time()
    for group, first, last, population in GROUPS:
        shards = manifest["shards"][first:last]
        bounds, total = [], 0
        for shard in shards:
            path = ROOT / shard["path"]
            assert path.stat().st_size == shard["bytes"]
            total += shard["bytes"]
            bounds.append(total)
        handles = collections.OrderedDict()
        selected, attempts, accepted = set(), 0, []
        while len(accepted) < per_group:
            attempts += 1
            ticket = rng.randrange(total)
            index = bisect.bisect_right(bounds, ticket)
            position = ticket - (bounds[index - 1] if index else 0)
            if index not in handles:
                if len(handles) >= 24:
                    _, old = handles.popitem(last=False)
                    old.close()
                handles[index] = (ROOT / shards[index]["path"]).open("rb")
            handle = handles[index]
            handles.move_to_end(index)
            offset, line = containing_row(handle, position)
            assert len(line) >= 8
            if rng.random() >= 8 / len(line) or (index, offset) in selected:
                continue
            record = json.loads(line)
            assert len(record) == 5
            text, target, domain, bucket, bin_index = record
            assert isinstance(text, str) and 0 <= target < len(text) - 1
            selected.add((index, offset))
            row = {"id": f"{group}-{len(accepted)+1:04d}", "stratum": group,
                   "domain": manifest["domains"][domain], "text": text,
                   "target_index": target, "length_bucket": bucket, "position_bin": bin_index,
                   "path": str((ROOT / shards[index]["path"]).relative_to(ROOT)), "offset": offset,
                   "row_sha256": hashlib.sha256(line).hexdigest(), "flags": flags(text, target),
                   "review_sample": len(accepted) < review_per_group}
            accepted.append(row)
        for handle in handles.values():
            handle.close()
        rows.extend(accepted)
        review.extend(accepted[:review_per_group])
        inventory.append({"stratum": group, "population": population,
                          "automated_n": per_group, "review_n": review_per_group,
                          "proposals": attempts, "bytes": total})
        print(json.dumps({"stage": "sampled", **inventory[-1], "seconds": time.time()-started}), flush=True)
    with (OUT / "sample.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    save(OUT / "review-sample.json", review)
    save(OUT / "sampling.json", {"seed": 2026091601, "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                               "method": "uniform bytes across each stratum; accept 8/row_bytes; reject repeated row locations",
                               "strata": inventory, "total_population": sum(x[3] for x in GROUPS),
                               "seconds": time.time()-started})


def consumed_documents(run):
    ends = {}
    for line in (ROOT / f"training/artifacts/{run}/prepare.log").open():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        cursor = row.get("cursor", {})
        if row.get("stage") == "preparing_web" and cursor.get("file", 0) > 0 and cursor.get("group") == 0 and cursor.get("row") == 0:
            ends[cursor["file"]] = row["documents"]
    assert len(ends) == 256
    return [ends[i] - ends.get(i-1, 0) for i in range(1, 257)]


def trace_web():
    # This audit describes a fixed historical corpus. Later default policy
    # changes must not change its reconstruction of the removed proxy labels.
    run = ROOT / "training/artifacts" / MANIFEST.parent.name
    snapshot = run / "source"
    plan = json.loads((run / "run.json").read_text())
    for name in ("prepare_web_data.py", "prepare_synthetic_data.py", "text_policy.py", "text-policy.json"):
        relative = "training/" + name
        assert hashlib.sha256((snapshot / relative).read_bytes()).hexdigest() == plan["source_hashes"][relative]
    sys.path[:0] = [str(snapshot / "training"), str(ROOT / "training/.deps")]
    import pyarrow.parquet as pq
    from prepare_web_data import labeled_samples
    from prepare_synthetic_data import normalize_document, document_signature
    from text_policy import PROXY_PUNCTUATION
    review = json.loads((OUT / "review-sample.json").read_text())
    # Also recover a bounded risk sample, separate from the probability sample.
    extras = []
    counts = collections.Counter()
    for line in (OUT / "sample.jsonl").open():
        row = json.loads(line)
        if row["review_sample"] or not row["stratum"].startswith("web"):
            continue
        for flag in row["flags"]:
            if counts[flag] < 12:
                extras.append(row)
                for f in row["flags"]:
                    counts[f] += 1
                break
    save(OUT / "risk-sample.json", extras)
    targets = collections.defaultdict(list)
    for row in review + extras:
        if row["stratum"].startswith("web"):
            targets[(row["text"], row["target_index"])].append(row["id"])
    runs = ["web-mix-20m-192ch-16conv-20260915", "web-mix-40m-192ch-16conv-20260916"]
    limits = [max(a, b) for a, b in zip(*(consumed_documents(r) for r in runs))]
    raw = ROOT / "training/data/raw/ultra-fineweb-zh"
    found, documents, pairs = {}, 0, 0
    started = time.time()
    output = OUT / "web-provenance.jsonl"
    with output.open("w") as destination:
        for file_index, limit in enumerate(limits):
            path = raw / f"ultrafineweb-zh-part-{file_index+1:03d}-of-256.parquet"
            parquet = pq.ParquetFile(path)
            groups = list(range(parquet.num_row_groups))
            random.Random(20260915 + file_index).shuffle(groups)
            consumed = 0
            for group in groups:
                group_row = 0
                for batch in parquet.iter_batches(batch_size=128, row_groups=[group], columns=["content", "source", "score"], use_threads=False):
                    for item in batch.to_pylist():
                        if consumed >= limit:
                            break
                        content = item["content"] or ""
                        consumed += 1
                        documents += 1
                        group_row += 1
                        normalized = None
                        for text, target, punctuation in labeled_samples(content):
                            pairs += 1
                            ids = targets.get((text, target))
                            if not ids:
                                continue
                            if normalized is None:
                                normalized = normalize_document(content)
                            left, right = text[:target+1], text[target+1:]
                            pattern = re.compile(re.escape(left) + r"\s*([" + re.escape("".join(PROXY_PUNCTUATION)) + r"\s]+)" + re.escape(right))
                            match = pattern.search(normalized)
                            context = normalized[max(0, match.start()-150):match.end()+150] if match else normalized[:500]
                            for ident in ids:
                                if ident in found:
                                    continue
                                result = {"id": ident, "source": item["source"], "score": item["score"],
                                          "raw_file": path.name, "row_group": group, "row_in_group": group_row-1,
                                          "document_sha256": document_signature(content), "document_characters": len(normalized),
                                          "punctuation": punctuation, "context": context,
                                          "context_matched": bool(match), "text": text, "target_index": target}
                                found[ident] = result
                                destination.write(json.dumps(result, ensure_ascii=False) + "\n")
                        if consumed >= limit:
                            break
                    if consumed >= limit:
                        break
                if consumed >= limit:
                    break
            destination.flush()
            if file_index % 8 == 7 or file_index == 255:
                print(json.dumps({"stage": "tracing", "files": file_index+1, "documents": documents,
                                  "pairs": pairs, "found": len(found), "wanted": sum(map(len, targets.values())),
                                  "seconds": time.time()-started}), flush=True)
    save(OUT / "trace-summary.json", {"documents": documents, "pairs": pairs, "found": len(found),
                                     "wanted": sum(map(len, targets.values())),
                                     "missing": [i for ids in targets.values() for i in ids if i not in found],
                                     "seconds": time.time()-started})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["sample", "trace-web"])
    args = parser.parse_args()
    sample() if args.action == "sample" else trace_web()
