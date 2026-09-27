# Padding correction and larger continuation — 2026-09-25

The confirmed padding defect is fixed. A detached pipeline is preparing
**150,000,000 additional web pairs** and will train one epoch on
**396,266,210 pairs** (350 million web plus 46,266,210 inherited local pairs).
This is 1.609 times the preceding 246,266,210-pair corpus. The run is in progress;
there is no new full-validation or test result yet.

## Correctness change

Each residual block now clears padded activations after LayerNorm and after
the first convolution/GELU, as well as at the block output. Consequently the
second convolution cannot return artificial padded activations to real tokens.
The weights, parameter names, dimensions, token IDs and unpadded computation
remain compatible. The training execution contract is now
`masked-residual-convolutions-v1`, recorded in configuration/checkpoints so
old optimizer runs cannot silently resume with changed forward semantics.
Old checkpoints remain usable for explicit weight initialization.

Twenty Metal/device, checkpoint, data preparation and coordinator tests passed.
The new regression checks real-gap logits and gradients with 0, 1, 2 and 17
extra padding positions and with a longer companion sequence. It deliberately
uses nonzero LayerNorm biases. The existing integration test also covers
interruption, exact next-batch resume, worker refresh, full fixture evaluation
and browser-export parity.

The frozen production best weights were independently checked on 500 recorded
inputs: all 1,500 padding comparisons preserved top-1, with maximum absolute
logit difference **0.0000114441**. Unpadded predictions also matched all 500
previously recorded JavaScript results (maximum logit difference 0.000457764).
This is correctness evidence, not a new accuracy measurement or a guarantee
of reaching 90%.

## Training and evaluation

- Start from the current production best state, full-data low-rate epoch 1:
  `learning-rate-192ch-full-246m-v7-20260923/epoch-1-backend/training-state.pt`.
  SHA-256: `5169707d95d51fe2515496f372f8ad52197f531f37d84f4b6775d32a5b9e7fca`.
- Keep 192 channels, eight residual blocks / 16 convolutions, 8,192 characters,
  3,496,329 parameters, learning rate 0.00003 and fresh AdamW.
- Keep existing position and domain weighting (domain exponent 0.65), clipping
  at 1, batch limits 512 examples / 8,192 tokens and seed 2026090405. They are
  training choices whose harm has not been established; no untested weighting
  changes are bundled into this continuation.
- Retain all inherited shards; prepare new data from all 256 downloaded files
  using unchanged v7 cleaning and document split seed 20260915. Reject inputs
  already in the historical corpus/holdouts, and reserve holdout documents.
  No new downloads, score filter or semantic repair.
- Keep the **entire** validation (2,522,347) and test (2,569,996) files
  byte-identical. Measure the starting weights on full validation again with
  corrected padding, then train. Select by **overall validation top-1**, retaining
  initialization as an eligible fallback. Report per-domain accuracy as well.
  Evaluate the validation-selected checkpoint on full test after training.
- Historical batched scores used the old padding semantics. The valid
  before/after training comparison is the freshly measured initial validation
  against the new endpoint, both with the fix. The run does not separately
  remeasure the old weights' full test score if the new epoch wins.

This is a larger continuation, not an experiment isolating data quantity or
padding: it adds training exposure and changes checkpoint selection as described.
The extension's published weights and rules are not replaced by this job.

## Operation and completion

```sh
npm run model:expand-padding-fixed
# After a graceful stop, process failure or reboot:
npm run model:expand-padding-fixed -- --resume
```

Run directory: `training/artifacts/padding-fixed-web-350m-v7-20260925/`.
Data directory: `training/data/processed/padding-fixed-web-350m-v7-20260925/`.

The coordinator freezes code, initialization and input hashes. Preparation is
resumable, with committed shard sizes and document cursors. Training saves
atomically every two shards, at requested stops and before automatic Metal
worker refreshes. The detached process survives closing the terminal/Codex;
`caffeinate -is` prevents idle sleep. Manual sleep pauses progress and shutdown
requires the resume command above. Arbitrary failures are recorded, not retried
indefinitely; MPS memory refresh exit 75 resumes only from an advanced checkpoint.

Follow `status.json`, `prepare.log`, `training.log` and data `status.json`.
At completion, the pipeline writes `result.json`, `candidate/smoke-metrics.json`
and a separate `candidate/boundary-model-data.js`. `result.json` includes the
corrected initial validation, epoch endpoint, selected checkpoint's validation
and test, and validation gain in percentage points. A failed process records
`stage: failed` and its error; it never reports success before evaluation/export.

Estimated preparation, training and evaluation: **44–50 hours**, conditional
on sustained power and available compute. This is an estimate, not a deadline
guarantee. The initial 200-million-additional plan was reduced to 150 million
to aim near the user's two-day return. The unchanged historical 246m run took
22.55 training hours; a discarded-weights timing probe found corrected per-step
time 10–13% higher on three fixed batch shapes. The probe excludes data loading,
evaluation, worker refreshes and thermal changes. Initial free disk space was
87 GiB; added compact data is expected to consume roughly 14–16 GiB plus
indices, frozen evaluation copies and checkpoints.

Preflight evidence: `preflight.json` in the run directory. Reproduce against
the retained historical checkpoint with:

```sh
PYTHONPATH=training/.deps:training python3 training/verify_padding_contract.py \
  --output /tmp/haodu-padding-preflight.json
PYTHONPATH=training/.deps:training python3 -m unittest \
  training.test_training_device training.test_web_preparation training.test_web_continuation
```
