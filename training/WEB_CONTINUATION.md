# Local web continuation

## Expansion to 40 million web training pairs (2026-09-16)

The first mixed epoch completed all 94,121,940 training examples. On the same
1,158,011 validation examples, accuracy increased from 86.3990% to **87.4859%**,
domain macro accuracy from 85.3326% to **86.5490%**, and loss fell from 0.427472
to **0.392412**. The 20-million run was stopped safely during epoch 2 after
1,979,757 examples (shard index 10, next batch 879); its checkpoint is preserved.

The expansion inherits the **completed mixed epoch-1 best_state**. Its frozen
selection is in the previous artifact directory's `epoch-1-expansion/`, with
SHA-256 `0dc993a820dc151ef8bc0bf2232722179a253fec443e8a5a11411cc172f191d2`.
All 3,496,329 initialization parameters have been verified against that best_state.

```bash
npm run model:expand-web
npm run model:expand-web -- --resume
```

The new artifact/data directories are named `web-mix-40m-192ch-16conv-20260916`.
Preparation retains all existing shards and adds another **20,000,000 unique
projected-input training pairs**, giving **40,000,000 web pairs** and
**114,121,940 total training pairs** per epoch. The web share becomes **35.05%**
by sample count. Two new epochs are planned with the same architecture,
vocabulary and training settings, starting a fresh optimizer and epoch counter.

Both complete holdouts are **byte-identical** to the 20-million run:
validation 1,158,011 examples, test 1,183,228 examples. The preparation keeps the
same document-partition seed and raw-data revision, skips every validation/test
partition document (including previously unused holdout documents), and indexes
all inherited training data before accepting additions. The existing web domain
ID is reused. Historical corpus and holdout-document registries are resolved
through the original manifests/summaries; no old training or evaluation file is
rewritten. The same validation population now supports direct before/after
comparison. Preparation automatically leads into training; the bundled backend
is not replaced by this operation.

## Initial 20-million run

2026-09-15: the original 16-layer run was stopped safely in epoch 2 after
15,034,179 / 74,121,940 examples (20.28%). Its resumable checkpoint retains
shard index 76 and next batch 1133. The best completed epoch is still **epoch 1**,
with 89.1495% accuracy on the frozen 727,578-example validation set.

The new run adds **20,000,000** training pairs from the downloaded Chinese
Ultra-FineWeb corpus to all **74,121,940** existing training pairs: **94,121,940**
per epoch, with 21.25% new examples by count. Existing domain and position loss
weighting still applies, so this percentage is not the new domain's loss weight.
Two epochs are planned. The run starts from the frozen best epoch-1 weights;
the unfinished epoch-2 state remains available in its original run directory.

```bash
npm run model:continue-web
# Resume either data preparation or training after an interruption:
npm run model:continue-web -- --resume
# Preparation tests use the local training dependencies:
python3 training/test_web_preparation.py
```

The coordinator freezes its source files and checkpoint under
`training/artifacts/web-mix-20m-192ch-16conv-20260915/`. It prepares data under
`training/data/processed/web-mix-20m-192ch-16conv-20260915/`, verifies initialization,
then starts training automatically. `status.json`, `prepare.log`, `training.log`,
and the data directory's `status.json` show the current stage. A graceful SIGTERM
to the coordinator forwards one stop request to its active child. Preparation
resumes from committed document/output offsets; training uses its batch checkpoint.

## Sampling and supervision

- Use only the 256 locally downloaded Parquet files, pinned to
  `02c85641e3d19a854be2e09139c25adaa9518063`. Require the download verification
  registry and matching sizes; record the source SHA-256 values and file mtimes.
- Give every source file an equal training-pair quota (78,125 each), using a
  seeded shuffle of its row groups. This covers every file rather than taking
  only the first files; it is not a uniform sample of every document in the corpus.
- Apply the existing document filter (at least 80 Han characters, at least 70%
  Han among Han/ASCII letters), normalization, punctuation targets and adjacent
  fragment concatenation. Retain actual punctuation labels for evaluation.
  Add no sequence-length or per-document pair cap.
- Assign whole documents by a stable hash: 96% train, 2% validation, 2% test.
  Pair counts need not have precisely these proportions. Keep every resulting
  holdout example; there is no small validation subset.
- Preserve the original validation and test files unchanged as the prefixes of
  the combined splits. New examples use the `web` domain, with component-source
  metadata retained. Full validation reports old domains and the new web domain;
  test is used after checkpoint selection at the end of the run.

## Overlap controls

Index all training shards referenced by the current and inherited manifests,
plus the original validation and test inputs. Reject new candidate inputs that
match their **NFKC Han projection**, regardless of target position. Also reject
matches across new training, validation and test samples, and exclude original
holdout document fingerprints. This conservative rule removes some otherwise
distinct examples (for example, differences only in numbers or punctuation).

Bloom filters have no false negatives for inserted keys; false positives only
discard extra candidates. Historical indexing records each scanned file's digest.
After an interrupted write, new-data deduplication is reconstructed from committed
output bytes so discarded partial writes cannot change the resumed sample set.
These checks do not establish absence of paraphrases or other near duplicates.

## Training

Architecture remains 16 convolution layers, 192 channels and 8,192 vocabulary
entries (3,496,329 parameters). Initialization checks every parameter against the
frozen `best_state`. The changed data starts a new AdamW optimizer and new epoch
counter. Learning rate (0.0003), batch limits (512 examples / 8192 tokens), domain
weight power (0.65), selection macro weight (0.5), clipping (1.0) and CPU threads
(5) match the previous run. All combined validation samples participate in selection.

The mixed validation population differs from the old run's population. Compare
old-domain results separately rather than interpreting the change in combined
accuracy as a model improvement. The final export stays in the candidate directory;
the extension's bundled model is not automatically replaced.
