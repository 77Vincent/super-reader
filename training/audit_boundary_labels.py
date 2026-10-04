"""Sample punctuation-derived labels without loading model predictions.

The review is an AI first pass, not an independently adjudicated benchmark.
Run with Python 3 from any working directory. No training corpus is modified.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "training/artifacts/position-ablation-1m-20261004/data/validation.jsonl"
OUT = ROOT / "training/artifacts/label-quality-20261004"
REVIEW = ROOT / "training/reviews/boundary-quality-20261004"
SEED = 2026100417
EXPECTED_SHA = "740072c0a4658e03e6a06133403555e2ab367edca6dddf548a96bf90074bbe6a"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sample():
    assert sha(VALIDATION) == EXPECTED_SHA, "Validation snapshot changed"
    groups = {"single": [], "ordinary": []}
    for line_no, line in enumerate(VALIDATION.open(), 1):
        row = json.loads(line)
        text = "".join(row["tokens"])
        cut = row["target_index"] + 1
        assert 0 < cut < len(text)
        cohort = "single" if min(cut, len(text) - cut) == 1 else "ordinary"
        groups[cohort].append(dict(
            validation_row_1based=line_no, id=row["id"],
            document_id=row["document_id"], source=row.get("source", row["domain"]),
            domain=row["domain"], text=text, gold_cut=cut,
            punctuation=row.get("punctuation"), cohort=cohort,
            row_sha256=hashlib.sha256(line.encode()).hexdigest(),
        ))
    assert [len(groups[g]) for g in ["single", "ordinary"]] == [252, 99748]
    chosen = {"single": groups["single"], "ordinary": sorted(
        random.Random(SEED).sample(groups["ordinary"], 200),
        key=lambda r: r["validation_row_1based"])}
    rows = []
    for cohort, prefix in [("single", "S"), ("ordinary", "N")]:
        rows.extend(dict(review_id=f"{prefix}{i:03d}", **r)
                    for i, r in enumerate(chosen[cohort], 1))
    REVIEW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path = REVIEW / "sample.jsonl"
    if path.exists():
        assert path.read_text() == payload, "Refusing to replace a different review sample"
    else:
        path.write_text(payload)
    write_json(REVIEW / "protocol.json", {
        "validation": str(VALIDATION.relative_to(ROOT)), "validation_sha256": EXPECTED_SHA,
        "sample_sha256": sha(path), "seed": SEED,
        "population": {g: len(rs) for g, rs in groups.items()},
        "review_counts": {g: len(rs) for g, rs in chosen.items()},
        "selection": "All 252 single-character-side rows; random.sample without replacement of 200/99748 other rows; Python random.Random, sorted into validation order.",
        "reviewer": "Codex AI first-pass semantic review; human confirmation pending",
        "blinding": "No predictions in review input. Some examples were seen in previous discussion; not an independent blind human review.",
        "unit": "Acceptability of the marked boundary for reading in the prepared two-fragment input. Not article factual correctness or original-author correctness.",
        "verdicts": {
            "A": "Marked boundary defensible; no alternative enumerated. Does not imply uniqueness.",
            "M": "Marked boundary defensible and at least one explicitly enumerated alternative also defensible; alternatives are not exhaustive.",
            "E": "Clear lexical/name/structural defect at the target under natural reading segmentation.",
            "U": "Insufficient context, ambiguous extraction/structure, or unresolved boundary judgment.",
        },
        "cuts": "1-based character count to the LEFT of a cut; zero-based model gap index + 1. Measured on normalized prepared text.",
        "context": "No raw source document recovery in this audit; document_id/source identify prepared provenance only.",
        "constraints": ["Length, model disagreement, subject matter, and source are not sufficient grounds for E.",
                        "Input residue or missing outer context does not automatically invalidate a locally defensible boundary.",
                        "Dictionary/list/heading ambiguity is U unless the local boundary is clearly defensible or clearly defective.",
                        "Annotation frozen and hashed before joining predictions.",
                        "Do not pool the oversampled 452 rows into a corpus error rate.",
                        "Training/validation/test data and released model remain unchanged."]
    })
    return rows


def analyze():
    """Join frozen semantic judgments to four already-computed predictions."""
    rows = sample()
    freeze = json.loads((REVIEW / "freeze.json").read_text())
    for name in ["sample", "annotations"]:
        assert sha(REVIEW / f"{name}.jsonl") == freeze[f"{name}_sha256"], name
    annotations = [json.loads(l) for l in (REVIEW / "annotations.jsonl").open()]
    labels = {r["review_id"]: r for r in annotations}
    assert len(labels) == len(annotations) == len(rows) == 452
    assert labels.keys() == {r["review_id"] for r in rows}
    for r in rows:
        a = labels[r["review_id"]]
        assert a["verdict"] in {"A", "M", "E", "U"} and a["reason"]
        cuts = a["alternative_cuts"]
        assert (a["verdict"] == "M") == bool(cuts)
        assert len(cuts) == len(set(cuts))
        assert all(isinstance(c, int) and 0 < c < len(r["text"]) and c != r["gold_cut"] for c in cuts)
    sys.path.insert(0, str(ROOT / "training/.deps"))
    import numpy as np
    paths = {
        "before": ROOT / "training/artifacts/single-character-ablation-1m-20261004/control/paired-predictions.npz",
        "after": ROOT / "training/artifacts/single-character-ablation-1m-20261004/paired-predictions.npz",
    }
    gold = np.array([json.loads(l)["target_index"] for l in VALIDATION.open()])
    predictions = {}
    for stage, path in paths.items():
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data[k].copy() for k in data.files}
        assert arrays.keys() == {"targets", "inverse-cell", "none"}
        assert np.array_equal(arrays["targets"], gold)
        for mode in ["inverse-cell", "none"]:
            assert arrays[mode].shape == gold.shape
            predictions[f"{mode}_{stage}"] = arrays[mode] + 1
    reviewed = []
    for r in rows:
        a = labels[r["review_id"]]
        c = r["gold_cut"]
        rr = {**r, **a, "marked_text": r["text"][:c] + "｜" + r["text"][c:]}
        rr["predictions"] = {k: int(v[r["validation_row_1based"] - 1]) for k, v in predictions.items()}
        assert all(0 < c < len(r["text"]) for c in rr["predictions"].values())
        reviewed.append(rr)
    counts, scores = [], []
    for cohort in ["single", "ordinary"]:
        group = [r for r in reviewed if r["cohort"] == cohort]
        freq = collections.Counter(r["verdict"] for r in group)
        counts.append({"cohort": cohort, "n": len(group), "documents": len({r['document_id'] for r in group}),
                       **{k: freq[k] for k in ["A", "M", "E", "U"]}})
        for mode in ["inverse-cell", "none"]:
            for verdict in ["all", "A", "M", "E", "U"]:
                subset = [r for r in group if verdict == "all" or r["verdict"] == verdict]
                def hit(r, stage):
                    return r["predictions"][f"{mode}_{stage}"] == r["gold_cut"]
                lost = [r['review_id'] for r in subset if hit(r, "before") and not hit(r, "after")]
                gained = [r['review_id'] for r in subset if not hit(r, "before") and hit(r, "after")]
                scores.append({"cohort": cohort, "mode": mode, "verdict": verdict, "n": len(subset),
                               "before": sum(hit(r, 'before') for r in subset),
                               "after": sum(hit(r, 'after') for r in subset),
                               "newly_wrong": len(lost), "newly_right": len(gained),
                               "lost_ids": lost, "gained_ids": gained})
    for mode, expected in [("inverse-cell", (203, 164, 39)), ("none", (179, 153, 26))]:
        total = next(s for s in scores if s['cohort'] == 'single' and s['mode'] == mode and s['verdict'] == 'all')
        assert (total['before'], total['after'], total['newly_wrong']) == expected
        assert total['newly_right'] == 0
    multiple_scores = []
    for cohort in ["single", "ordinary"]:
        group = [r for r in reviewed if r['cohort'] == cohort and r['verdict'] == 'M']
        for model in predictions:
            alternative_hits = [r['review_id'] for r in group if r['predictions'][model] in r['alternative_cuts']]
            exact = sum(r['predictions'][model] == r['gold_cut'] for r in group)
            multiple_scores.append(dict(cohort=cohort, model=model, n=len(group), exact=exact,
                                        listed_acceptable=exact+len(alternative_hits), alternative_hit_ids=alternative_hits))
    doc_groups = collections.defaultdict(list)
    for r in reviewed:
        doc_groups[r['cohort'], r['document_id']].append(r)
    clusters = [{"cohort": cohort, "document_id": doc, "n": len(group),
                 "verdicts": dict(collections.Counter(r['verdict'] for r in group))}
                for (cohort, doc), group in doc_groups.items() if len(group) > 1]
    findings = {"review_counts": counts, "exact_gold_scores": scores, "multiple_boundary_scores": multiple_scores,
                "repeated_documents": sorted(clusters, key=lambda r: -r['n']),
                "freeze": freeze, "predictions_sha256": {k: sha(v) for k, v in paths.items()},
                "checks": {"validation_sha256_matches": True, "sample_selection_reproduced": True,
                           "annotation_ids_and_cuts_valid": True, "annotation_freeze_matches": True,
                           "four_prediction_arrays_match_validation_order": True, "prior_252_counts_reconciled": True},
                "limitations": ["AI first pass, not human gold; earlier partial exposure to examples.",
                                "Normalized pair text only, no raw source verification.",
                                "Multiple valid cuts are not exhaustively listed; A does not mean unique.",
                                "Exact-gold scores measure punctuation-label agreement, not semantic correctness.",
                                "No accuracy claim for clear-error E rows; moving away from bad gold is not automatically right.",
                                "252 is a census within reused 100k smoke validation; 200 ordinary rows are sampled, not whole corpus.",
                                "Single seed continued from pretrained checkpoint; retained rows change batches/update counts."]}
    followup = json.loads((REVIEW / "prediction-followup.json").read_text())
    assert followup['initial_annotations_sha256'] == freeze['annotations_sha256']
    by_id = {r['review_id']: r for r in reviewed}
    expected = {r['review_id'] for r in reviewed if r['cohort'] == 'single' and r['verdict'] in ['A', 'M']
                and any(r['predictions'][m+'_before'] == r['gold_cut'] and r['predictions'][m+'_after'] != r['gold_cut']
                        for m in ['inverse-cell', 'none'])}
    assert len(followup['rows']) == len(expected) == 29
    assert {r['review_id'] for r in followup['rows']} == expected
    for f in followup['rows']:
        r = by_id[f['review_id']]
        assert f['after_verdict'] in ['A', 'E', 'U'] and f['reason']
        modes = [m for m in ['inverse-cell', 'none'] if r['predictions'][m+'_before'] == r['gold_cut']
                 and r['predictions'][m+'_after'] != r['gold_cut']]
        assert f['modes'] == modes
        assert all(f['after_cut'] == r['predictions'][m+'_after'] for m in modes)
    findings['unblinded_followup'] = {
        'stage': followup['stage'], 'review_n': len(followup['rows']),
        'by_mode': [{'mode': m, **dict(collections.Counter(r['after_verdict'] for r in followup['rows'] if m in r['modes']))}
                    for m in ['inverse-cell', 'none']],
        'sha256': sha(REVIEW / 'prediction-followup.json'),
    }
    findings['checks']['posthoc_coverage_and_prediction_cuts_match'] = True
    write_json(OUT / "findings.json", findings)
    write_json(OUT / "reviewed.json", reviewed)
    with (OUT / "human-review.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["review_id", "validation_row_1based", "cohort", "marked_text", "punctuation", "document_id",
                  "human_verdict", "human_alternative_cuts", "human_reason"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(reviewed)
    with (OUT / "reviewed.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["review_id", "cohort", "validation_row_1based", "marked_text", "verdict", "reason", "alternative_cuts", "document_id"]
        writer = csv.DictWriter(f, fieldnames=fields + list(predictions), extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**r, **r["predictions"]} for r in reviewed)
    return findings


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--show", choices=["single", "ordinary"])
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--count", type=int, default=50)
    p.add_argument("--analyze", action="store_true")
    args = p.parse_args()
    if args.analyze:
        findings = analyze()
        print(json.dumps({"counts": findings["review_counts"], "checks": findings["checks"]}, ensure_ascii=False))
        raise SystemExit(0)
    rows = sample()
    if args.show:
        selected = [r for r in rows if r["cohort"] == args.show]
        for r in selected[args.start-1:args.start-1+args.count]:
            c = r["gold_cut"]
            print(f'{r["review_id"]} [{r["punctuation"]}] {r["text"][:c]}｜{r["text"][c:]}')
    else:
        print(json.dumps({"review_n": len(rows), "sample_sha256": sha(REVIEW / "sample.jsonl")}))
