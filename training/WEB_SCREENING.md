# Local web prose screening

This optional preparation stage selects continuous prose from cached Chinese
Ultra-FineWeb documents. It does not change punctuation proxies, generate
boundary labels, alter validation/test data, or launch training.

```bash
python3 training/screen_local_web.py --scope consumed \
  --output-dir training/artifacts/web-screen-new --workers 6
python3 training/select_web_prose_spans.py \
  --screen-dir training/artifacts/web-screen-new \
  --output-dir training/artifacts/web-spans-new --workers 6
```

`consumed` reconstructs the raw documents visited during the existing 20m/40m
web preparation, using the same shuffled row-group order and per-file limits.
It covers all 256 files but does **not** scan every downloaded document.
`--scope all` scans the entire download and is substantially larger.
The download manifest, verified-file registry, preparation logs, and inherited
holdout registries must be available locally. These commands do not download
missing prerequisites.

## Selection

The document stage checks Chinese prose density and quarantines conspicuous
catalogs, numbered structures, exercises, formulas/code, markup, tables,
boilerplate and fragmented/verse-like text. Quarantine means unsuitable for
intact prose training under these conservative rules; it does not mean every
sentence is wrong. Useful prose in quarantined documents is not recovered in
this first pass.

Admission is not based on a source blacklist or a high score cutoff. The existing
0.5 score floor is retained, and source/score breakdowns remain available.
Inspection uses NFKC for matching without changing the stored text. Rules and
thresholds live in `web_document_quality.py`, independently of `text-policy.json`.

The second stage selects physical source lines with at least 60 Han characters,
at least two clauses of eight or more Han characters, and at least 70% of Han
characters within such clauses. It removes lines with recognizable calls to
action, boilerplate, repeated keyword stuffing and the document-stage structural
problems. It keeps exact source substrings and never joins adjacent lines or
text across removed material. These thresholds favor sustained prose and
deliberately omit many valid short paragraphs.

## Outputs and safeguards

- Document `manifest.json` and per-file indices record every decision and reason.
- `train-candidates.parquet` contains unique training-only documents. Inherited
  holdouts and deterministic hash-reserved validation/test documents are excluded.
- Span `train-prose.parquet` contains exact, deduplicated source spans in `content`,
  with source, score, original file/row-group/row, code-point offsets, document
  fingerprints and span SHA-256. **Each row is an independent input document;
  never concatenate rows before generating labels.**
- Manifests include code/input hashes, policy snapshots, counts and limitations.
  Document screening can reuse completed per-file results when all identities
  match. Span extraction reruns under the same identity; changed rules or inputs
  require a new output directory.
- Exact span deduplication does not replace near-duplicate checks or the later
  sample-level overlap audit against old train/validation/test data.

The downstream web preparer has not been rewired to consume this selection.
Use these artifacts for review first; subsequently prepare fresh samples from
each selected span with the agreed text policy and existing overlap safeguards.
Do not reinterpret upstream scores or structural admission as measured label
accuracy or a guarantee of factual correctness.

## Recorded local run

The 2026-09-16 run screened 553,294 consumed source documents. The document stage
retained 265,145, excluded 10,692 admitted holdout documents and removed 469
duplicate training documents, leaving 253,984 for paragraph extraction.
The final counts, source breakdown, fresh sample review and executed analysis
notebook are in `artifacts/web-prose-spans-20260916/README.md`.
Earlier review samples used to refine the rules are development evidence, not
an independent quality benchmark. Small AI reviews cannot certify the corpus.

```bash
PYTHONPATH=training:training/.deps python3 -m unittest \
  training.test_web_document_quality training.test_local_web_screening \
  training.test_web_prose_spans
```
