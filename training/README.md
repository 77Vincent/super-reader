# Boundary model training

The current data contract is **unicode-context-v2**, defined in
[text-policy.json](text-policy.json). Its model input representation remains
**unicode-context-v1**; v2 changes the punctuation-derived supervision, not the
CNN architecture or token IDs. Training, validation and test preparation share
this contract. Older corpora are retained for provenance and cannot be silently
mixed into a new run.

## Input and labels

Adjacent fragments A and B become the input AB. Only their separating proxy
punctuation is hidden; the target is the code-point gap between A and B.
The proxy set contains Chinese/ASCII commas, exclamation marks, question marks
and semicolons, plus the Chinese full stop `。` and ellipsis `…`. ASCII periods
`.` and their NFKC-compatible forms such as `．` remain context, including
numbering, initials, domains, filenames and runs of dots (`...`). Colons also
remain context.

Other punctuation, digits, letters, quotes and symbols remain in the input.
NFKC normalization standardizes compatible forms, except that `…` is preserved
so it cannot become an ASCII-dot proxy accidentally. Whitespace becomes a single
ASCII space and fragment edges are trimmed. Commas between digits stay intact;
periods are always retained. Both sides must contain a letter or number. Corpus-specific
Wikipedia markup and synthetic fenced-code/URL cleanup still happen during text
extraction, before sample generation.

```text
Source: 女：那可挺麻烦的，吃点儿治疗过敏的药吧。
Input:  女:那可挺麻烦的吃点儿治疗过敏的药吧
Target: 女:那可挺麻烦的 | 吃点儿治疗过敏的药吧

Source: 上午8:30至下午4:30；假日关门。
Input:  上午8:30至下午4:30假日关门
Target: 上午8:30至下午4:30 | 假日关门

Source: F. Billinghurst负责设计，团队实施。
Input:  F. Billinghurst负责设计团队实施
Target: F. Billinghurst负责设计 | 团队实施
```

The labels are punctuation-derived weak supervision. Accuracy measures recovery
of the hidden proxy; it does not establish human-rated reading chunk quality.

The [2026-09-16 corpus quality audit](CORPUS_QUALITY_AUDIT.md) records a stratified
1,000-row AI semantic review, 20,000-row automatic checks, proxy-label failure
examples, and limits of the resulting quality estimates.

The existing dated corpora, evaluation splits and running web-continuation job
were prepared under v1. Changing this default does not rewrite their samples or
alter their frozen source snapshots. Before a v2 run, regenerate train,
validation and test from the original documents into fresh directories while
preserving document ownership and overlap checks; dropping period-labeled rows
alone cannot restore periods previously removed from their inputs. Current
training/evaluation loaders reject v1 metadata. Historical jobs can continue
using their recorded v1 source snapshots. Accuracy across these label standards
must not be compared as if it were measured on the same holdout.

## Full local-corpus candidate

```bash
npm run model:retrain-context
# After an interruption:
npm run model:retrain-context -- --resume
```

The pipeline in [run_context_pipeline.py](run_context_pipeline.py) uses all cached
CLUE text entries, all extracted Chinese Wikipedia articles, and every row of the
cached Ultra-FineWeb Chinese multi-style Parquet. It does not cap documents,
pairs per document, sequence length, or the total synthetic sample count.
Deduplication, document holdouts, article extraction and the existing synthetic
Chinese-text quality checks still apply. It does not fetch additional corpora.

The default run is `training/artifacts/unicode-context-192ch-12conv-20260913/`.
Its stages are preflight, base regeneration, inherited-checkpoint overlap audit,
Wikipedia preparation, synthetic preparation, new-training overlap audit,
vocabulary construction, training, export and browser evaluation.

The initial checkpoint is the completed best epoch 2 of
`all-local-192ch-12conv-20260912`. A separate copy is recorded with its SHA-256.
The new candidate retains 192 channels and six residual blocks (12 kernel-3
convolutions). It starts two new epochs with learning rate 0.0003, domain-weight
power 0.65, equal overall/macro validation selection weights, gradient clipping
at 1.0, batches of at most 512 examples/8192 padded tokens, and five CPU threads.

The vocabulary preserves all existing IDs and extends to at most 8192 tokens
using training data only, with dedicated ASCII and common punctuation entries.
Embedding weights are copied by token identity; convolution and scoring weights
are inherited. New tokens are initialized separately and AdamW starts fresh.
The initialized model is evaluated on the new validation set before updates, so
checkpoint selection can also retain initialization (new-run epoch 0).

Original document-to-split assignments are preserved. Before establishing the
new holdout version, a conservative Han-projection comparison removes pairs seen
by inherited checkpoints. The new training shards must have zero exact input/gap
overlap with that evaluation set. Historical holdouts and reports are preserved;
old and new input-policy scores are not directly comparable.

`status.json`, `completed-stages.json`, stage logs and `corpus-coverage.json`
record progress. Preparation and training support restart; incomplete base/audit
outputs are archived before retry. A source hash change stops the next stage.
The trainer saves atomic checkpoints every four shards and at safe interrupt
boundaries. Completing the pipeline exports a separate candidate and does not
replace the extension's bundled weights.

## 16-layer continuation and Chinese web download

```bash
npm run model:download-web
npm run model:continue-depth
# After an interruption, repeat the download command and resume training:
npm run model:continue-depth -- --resume
```

These are independent jobs. The download fetches only the Chinese split of
`openbmb/Ultra-FineWeb`, pinned to revision
`02c85641e3d19a854be2e09139c25adaa9518063`: 256 Parquet files totaling
324,321,485,291 bytes (302.05 GiB). It uses three concurrent transfers, resumes
partial files, verifies every file against the upstream SHA-256, and reserves
32 GiB of free disk space. Connection outages are retried with up to 60 seconds
between attempts until connectivity returns or the job is stopped. Other errors
retain bounded retries; size and checksum mismatches stop the download.
Its manifest, verified-file list, progress and upstream
documentation are stored in `training/data/raw/ultra-fineweb-zh/`.
The upstream license also requires checking the original component datasets'
terms. Downloaded text is not automatically added to a running epoch; the separate
web continuation below prepares samples and protects the holdouts first.

The continuation run is
`training/artifacts/unicode-context-192ch-16conv-20260914/`. It inherits the
completed best epoch 2 from `unicode-context-192ch-12conv-20260913/candidate/`
and trains two new epochs on the same 74,121,940 examples, with all 727,578
validation and 766,489 test examples. Channels (192), vocabulary (8192), learning
rate (0.0003), domain weighting, batch limits and selection criteria remain the
same. AdamW starts fresh for the expanded architecture; new-run epoch 1 follows
the inherited epoch 2.

Eight residual blocks give 16 convolutions and 3,496,329 parameters. All inherited
tensors are copied exactly. The two added blocks start with zero residual scales,
so initialization preserves the old logits; the scales and convolution branches
then learn during optimization. The structural gap context can grow to 34
characters, up to 17 on each side. The preflight verifies weight/output equality,
nonzero learning gradients and exported JavaScript inference. Its probe updates
are discarded before formal training, which evaluates the complete validation
split before making updates and includes initialization in checkpoint selection.

The run snapshots its training/export code and initialization checkpoint so later
workspace edits do not change a running experiment. It saves atomic checkpoints
every four shards and at safe interrupt boundaries. `status.json`, `training.log`,
`preflight/verification.json` and the candidate checkpoints record progress.
Completion exports a separate candidate; the bundled backend stays on its
currently released model until explicitly promoted.

## Web data continuation

`npm run model:continue-web` prepares 20 million new training pairs across all
256 downloaded Chinese web files, combines them with the existing 74,121,940
pairs, and continues the best 16-layer epoch-1 weights for two epochs. It keeps
the vocabulary and architecture fixed. Whole-document web holdouts supplement
the unchanged original validation/test sets; no evaluation subset is used.
Historical input deduplication runs before sampling. Preparation and training
resume with `npm run model:continue-web -- --resume`.
See [the run design and overlap controls](WEB_CONTINUATION.md).

After the first mixed epoch improved full validation accuracy to 87.4859%,
`npm run model:expand-web` starts a separate continuation from that best epoch.
It retains the existing data, adds another 20 million distinct web training
pairs (40 million web / 114,121,940 total), and keeps both complete evaluation
splits byte-identical. `npm run model:expand-web -- --resume` resumes this run.

## Evaluation

```bash
npm run model:baseline
```

The real defaults are the new run's candidate model and
`training/data/processed/unicode-context-192ch-12conv-20260913-eval/validation.jsonl`.
Both must have been prepared/exported before this command can run. `--model`,
`--input` and `--output` are optional explicit experiment paths; they cannot bypass
input-policy checks. The evaluator rejects mismatched model/data representations,
invalid proxy labels anywhere in the input, and recorded checksum mismatches.
It uses the same deterministic sample of 500 examples per domain (seed 20260911).
Keep input and sample hashes fixed for within-version comparisons.

The bundled model is the completed epoch-1 best checkpoint from the 16-layer
web continuation run (`web-mix-20m-192ch-16conv-20260915`), trained on 94,121,940
pairs including 20 million web pairs;
see [BUNDLED_MODEL.md](BUNDLED_MODEL.md) for its provenance and validation results.
The backend first splits clauses using normalized proxy punctuation and enumeration
commas. All resulting clauses, including enumeration items, follow the same rule:
leave at most 12 visual units intact and send longer clauses with their
remaining Unicode context to the model. Numeric separators remain within numbers;
original text and UTF-16 offsets are preserved. Model cuts require immediately adjacent
Han characters on both sides in the original text, plus browser word protection.
The default [recursive softmax](RECURSIVE_SOFTMAX.md) caches original window logits
and recomputes softmax over each child's internal gaps. Only probabilities strictly
above 75% are eligible, with raw-logit ranking. An uncertain longer clause stays intact.
The [confidence audit](CONFIDENCE_THRESHOLD.md) measures original-input predictions
for the previous, pre-web model; its precision figures do not describe this new
checkpoint or establish the accuracy of recursive child decisions.

## Verification and small experiments

The [receptive-field analysis](RECEPTIVE_FIELD_ANALYSIS.md) scans all 74,121,940
training examples and stratifies the complete epoch-2 validation results by
context coverage. It records the coverage expected at larger CNN depths and the
limits of using those statistics to choose a model size.

The completed [CNN / Transformer experiment](CNN_TRANSFORMER_EXPERIMENT.md)
used identical training data and full validation/test splits. CNN reached 68.60%
test accuracy versus 60.85% for Transformer and led in all five test domains.
The experiment is closed; its dedicated scripts and generated artifacts were
removed after preserving the setup, results, limitations and audit fingerprints.

Recursive fragment rescoring without a confidence gate was evaluated and rejected: the example comparison
showed higher inference cost without a clear quality benefit. The backend retains
one-time clause scoring. Only the [conclusion](SCORING_COMPARISON.md) is retained;
the experimental option, scripts, tests and generated reports have been removed.

A separate [recursive >90% experiment](RECURSIVE_CONFIDENCE_EXPERIMENT.md)
compares fresh child inference with fixed original probabilities. Its
`scoringStrategy: "recursive-model"` option is opt-in; the default caches original
logits and recursively recomputes softmax. Reproduce the historical strategy comparison with
`node training/compare_recursive_confidence.mjs`.

```bash
npm test
npm run smoke:data
python3 training/run_smoke.py --epochs 2
# Extend an otherwise unchanged small run:
python3 training/run_smoke.py --epochs 6 --resume
```

`npm test` uses small temporary fixtures, not downloaded corpora or ignored model
artifacts. The real Parquet fixture installs pinned PyArrow on first use if it is
missing; Python 3.9+, pip and first-install network access are required. Full
training installs pinned PyTorch/tinygrad through the existing local bootstrap.

[preflight_context.py](preflight_context.py) verifies all inherited embeddings
with deliberately permuted IDs, checks unchanged non-embedding weights, runs an
actual optimization step involving a colon, exports it and compares browser and
PyTorch logits. Its tiny metrics are implementation checks, not performance claims.

See [BUNDLED_MODEL.md](BUNDLED_MODEL.md) for the currently shipped model and
[HISTORICAL_TRAINING.md](HISTORICAL_TRAINING.md) for previous experiment records.
