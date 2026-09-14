# Boundary model training

The current data contract is **unicode-context-v1**, defined in
[text-policy.json](text-policy.json). Training, validation, test data and exported
models identify their input representation explicitly. Older corpora are retained
for provenance and cannot be silently mixed into a new run.

## Input and labels

Adjacent fragments A and B become the input AB. Only their separating proxy
punctuation is hidden; the target is the code-point gap between A and B.
The proxy set contains Chinese/ASCII commas, periods, exclamation marks, question
marks, semicolons and ellipses. Colons are context rather than targets.

Other punctuation, digits, letters, quotes and symbols remain in the input.
NFKC normalization standardizes compatible forms; whitespace becomes a single
ASCII space and fragment edges are trimmed. Numeric periods and commas between
digits stay intact. Both sides must contain a letter or number. Corpus-specific
Wikipedia markup and synthetic fenced-code/URL cleanup still happen during text
extraction, before sample generation.

```text
Source: 女：那可挺麻烦的，吃点儿治疗过敏的药吧。
Input:  女:那可挺麻烦的吃点儿治疗过敏的药吧
Target: 女:那可挺麻烦的 | 吃点儿治疗过敏的药吧

Source: 上午8:30至下午4:30；假日关门。
Input:  上午8:30至下午4:30假日关门
Target: 上午8:30至下午4:30 | 假日关门
```

The labels are punctuation-derived weak supervision. Accuracy measures recovery
of the hidden proxy; it does not establish human-rated reading chunk quality.

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

The bundled model is the completed epoch-1 best checkpoint from this full run;
see [BUNDLED_MODEL.md](BUNDLED_MODEL.md) for its provenance and validation results.
The backend first splits clauses using normalized proxy punctuation and enumeration
commas. All resulting clauses, including enumeration items, follow the same rule:
leave at most 12 visual units intact and send longer clauses with their
remaining Unicode context to the model. Numeric separators remain within numbers;
original text and UTF-16 offsets are preserved. Opening/closing marks stay attached
to adjacent content when the model subdivides a clause.

## Verification and small experiments

The completed [CNN / Transformer experiment](CNN_TRANSFORMER_EXPERIMENT.md)
used identical training data and full validation/test splits. CNN reached 68.60%
test accuracy versus 60.85% for Transformer and led in all five test domains.
The experiment is closed; its dedicated scripts and generated artifacts were
removed after preserving the setup, results, limitations and audit fingerprints.

Recursive fragment rescoring was evaluated and rejected: the example comparison
showed higher inference cost without a clear quality benefit. The backend retains
one-time clause scoring. Only the [conclusion](SCORING_COMPARISON.md) is retained;
the experimental option, scripts, tests and generated reports have been removed.

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
