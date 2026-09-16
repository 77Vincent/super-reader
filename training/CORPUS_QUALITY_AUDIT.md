# Corpus quality audit — 2026-09-16

The current training corpus shows a small but repeatable class of unsuitable
punctuation-derived targets. Most reviewed targets were acceptable; the evidence
points primarily to proxy/reading-task mismatches, rather than widespread poor
punctuation habits by authors. This is an **AI semantic first pass, not an
independently human-adjudicated quality benchmark**.

The audit did not change the running training job, its data, validation/test
sets, or the backend.

## Scope and results

Population: `web-mix-40m-192ch-16conv-20260916`, **114,121,940 training rows**.
This is a historical **unicode-context-v1** corpus. The subsequent v2 default
excludes ASCII periods; this audit's original findings remain unchanged, and
raw-web reconstruction uses the audited run's hash-verified v1 source snapshot.
Four strata each contribute 5,000 random rows to automatic checks, of which 250
were reviewed semantically. The 1,000 reviewed rows were not selected using model
predictions. A separate 73-row targeted risk inspection does not contribute to
prevalence estimates.

| Training stratum | Population | Reviewed | Clear unsuitable target (E) | Uncertain (U) | Severe input noise |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original corpus, mostly Wikipedia | 42,217,584 | 250 | 3 | 1 | 3 |
| Existing multi-style corpus | 31,904,356 | 250 | 2 | 0 | 0 |
| First web addition | 20,000,000 | 250 | 9 | 3 | 2 |
| Second web addition | 20,000,000 | 250 | 8 | 3 | 7 |

Weighting each stratum by its population share gives:

- Acceptable target: **97.5721%**.
- Clear unsuitable target: **1.8593%**; stratified bootstrap 95% interval
  **1.1010–2.7160%**, covering sampling variation only.
- Uncertain: **0.5686%**. Counting every U as a problem raises the estimate to
  **2.4279%**; this is a rubric sensitivity check, not an error-rate bound.
- Severe input noise: **1.0748%**; minor input noise: **6.6087%**.
  Input quality and target quality overlap and must not be added.

These are training-row proportions, not model precision, online bad-cut rates,
or loss-weighted exposure. Domain/position loss weights can change the effective
training influence. Per-source error counts are too small for a firm ranking.

## Observed failure modes

Of the 22 E judgments, 15 concern structural numbering, two names/initials,
three closing quotes, one an opening bracket, and one a coordinate comma.
Examples use `|` for the training target:

- `F | Billinghurst`: splits an initial from a surname.
- `玉石的选购 九 | 宝石`: splits a section number from its heading.
- `字体大小调整与阴影效果 2.3.2 | 项目符号调整 2.3.3`:
  a section-number period becomes the target.
- `y)关于(a | b)的对称点为(2a-x`: a coordinate comma becomes the target.
- `并用“不要打扰我的圈子 | ”一词激起了一名接近的士兵`:
  removing punctuation leaves the target before the closing quotation mark.

The numbering judgment assumes the product should preserve a heading/number
as a reading unit. Some original punctuation is perfectly valid; its use as a
natural-language gap label is the problem. The targeted inspection additionally
confirmed domain-name periods, the period in `unistd.h`, code-argument commas,
and Japanese-language material. Those deliberately selected examples do not
establish prevalence.

Nineteen of the 22 E positions do not have Han characters on both sides. The
current backend's adjacent-Han condition would reject those positions, but
training on them can still affect scores and other predictions. This is not
evidence that the remaining three equal the online failure rate.

## Template repetition

A bounded probe of 20,000 rows found **seven identical sentence-template
families after numeric normalization** and **25 similar pairs**, all involving
Wikipedia-style population/area descriptions. Their targets can be valid, but
the patterns reveal a diversity concern.

The probe uses NFKC, numeric normalization, rare character-three-gram candidates,
Jaccard >= 0.85 and sequence similarity >= 0.90. It is neither exhaustive within
the sample nor a whole-corpus deduplication measurement. Pair counts are not row
counts; no full-corpus duplicate rate is inferred.

## Method and limitations

`audit_corpus.py` uniformly proposes byte positions within each stratum, takes
the containing JSONL row, and accepts it with probability `8 / row_byte_length`.
This cancels byte-sampling length bias. Repeated row locations are rejected.
Seed: `2026091601`; each sampled row retains its path, byte offset and SHA-256.
All 20,000 row hashes were checked. The byte-to-row mapping was also checked
exhaustively on mixed-length UTF-8 examples.

All 500 random web rows and 73 targeted web rows have exact text/target matches
in the consumed raw web corpus. The random rows match 498 document signatures.
Compact training rows lack original document IDs; matching preserves the first
compatible source and does not prove unique provenance. Non-web review uses the
prepared input, without recovered full original context. All 250 reviewed base
rows happened to be Wikipedia, so small legacy source domains cannot be assessed
separately. Factual correctness was not audited.

`analyze_corpus_audit.py` summarizes the frozen annotations, verifies sampled
bytes, and computes a 50,000-replicate stratified bootstrap (seed `2026091602`).
The interval excludes reviewer disagreement and systematic misclassification.
An upstream quality score is not a guarantee of target quality: one corrupted
formula/table sample had a score of 0.9527. This single counterexample does not
establish a score/error correlation.

## Follow-up supported by this audit

First independently review E/U judgments and a sample of A judgments. Then
evaluate sample-level handling of numbering, names, domains, code/math, and quote
attachment; add provenance sidecars when preparing a new data version. Measure
template prevalence before introducing per-template caps. Do not automatically
remove all short fragments, ads, technical prose, or mixed-script inputs.

Keep full validation/test comparisons stable. A separate human-reviewed reading
check can expose proxy-label problems; it must not replace or subsample the
standard validation set. No cleanup or retraining change was applied by this
audit.

## Local evidence and reproduction

Generated files are local, ignored artifacts in
`training/artifacts/corpus-quality-20260916/`:

- [Interactive report](artifacts/corpus-quality-20260916/report.html)
- [All 1,000 reviewed rows with judgments and context](artifacts/corpus-quality-20260916/reviewed.csv)
- [Executed analysis notebook](artifacts/corpus-quality-20260916/audit.ipynb)
- `sampling.json`, `sample.jsonl`, `annotations.jsonl`, `rubric.json`,
  `web-provenance.jsonl`, `findings.json`, `near-duplicates.json` and `audit.sqlite`

With the corresponding local corpus and preparation logs available:

```bash
python3 training/audit_corpus.py sample
python3 training/audit_corpus.py trace-web
# Requires the frozen annotations.jsonl; semantic judgments are not auto-generated.
python3 training/analyze_corpus_audit.py
```

Sampling regenerates the local sample files. For ordinary review, open the CSV
or run the notebook against the frozen artifacts instead. The standalone HTML
is generated from `artifact.json` using the Data Analytics portable renderer;
the original source and calculation metadata are embedded in it.

The HTML passed canonical validation, packaging, and structural verification.
An additional check using installed Chrome timed out during chart extraction;
browser layout/interaction verification is therefore incomplete. The portable
file includes the semantic chart-data/table fallback and does not need a server.
