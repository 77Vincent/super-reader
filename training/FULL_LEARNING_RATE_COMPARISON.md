# Full-data learning-rate comparison — 2026-09-23

Replay the training run that produced the currently bundled model, changing
only the learning rate from **0.0003 to 0.00003**. Reuse the completed control;
train only the new lower-rate arm. Results are pending.

Started on 2026-09-23 at 00:23 China time. Exact initialization transfer and input
verification passed for all 2,432 shards; the Metal worker is preparing training
weights and initial validation. Coordinator PID at launch: 67607 (use the current
`status.json` after any resume).

| Setting | Existing backend / control | New candidate |
| --- | --- | --- |
| Learning rate | 0.0003 | 0.00003 |
| Architecture | 192 channels, 8 residual blocks / 16 convolutions | Same |
| Parameters / vocabulary | 3,496,329 / 8,192 | Same |
| Training pairs | 246,266,210 | Same |
| Web / other pairs | 200,000,000 / 46,266,210 | Same |
| Epochs on this manifest | 1 | 1 |
| Shards | 2,432 | Same files, manifest bytes and order |
| Seed | 2026090405 | Same; not the newer pilot seed |
| Batch limits | 512 samples / 8,192 tokens | Same |
| Optimizer | Fresh AdamW, weight decay 0.0001, foreach enabled | Same |
| Weighting / clipping | Domain exponent 0.65, existing position weights, gradient clip 1 | Same |
| Device | Metal float32, one CPU host thread / interop thread | Same |
| Regular checkpoints | Every 4 shards | Same |
| Full validation / test | 2,522,347 / 2,569,996 | Same fixed splits |

## Matched starting point

The reference is
`training/artifacts/chinese-line-web-200m-16conv-v7-20260920/epoch-1-backend/`.
Its validation accuracy is **88.7786%** and test accuracy **88.8945%**.
The full-data run started from the preceding 100-million-web run's epoch-2
weights, and trained one new epoch on the expanded dataset. That complete
historical initialization is retained in the reference run's `initialization.pt`:

```text
eaa1fecea56261d85a55664eb2e2ceefa18d00cd270d2d3ed90ea741eb33df5c
```

The new arm independently starts from these **same pre-expansion weights**
with a fresh optimizer. Starting from the already bundled weights, or the newer
10-million-pair pilot winner, would add a different amount of training and
would not isolate learning rate. The smaller pilot provides evidence for the
rate choice; its numerical gain is not a prediction for this different start.

The launcher checks the current backend against the reference release hash,
verifies the control checkpoint is the completed epoch-1 endpoint, and verifies
every inherited parameter exactly. It copies the original frozen dependencies
and checks that the trainer's optimization code is semantically identical.
The only reporting addition is the previously tested `--defer-test` mode.

The manifest and holdout metadata must match the control checkpoint's identity.
All shard byte sizes are checked; manifest-provided historical hashes are checked
where available. Fresh SHA-256 hashes of all 2,432 shard contents, both holdouts,
the vocabulary, initialization and source snapshot are saved and rechecked on
resume. This reuses the existing cleaned corpus without generating more data.

## Evaluation

Evaluate the complete validation split at initialization and epoch end. Retain
the one-epoch endpoint metrics even if validation prefers the starting weights.
Compare the endpoint with the reused backend control using the existing score:
half overall validation accuracy plus half mean domain accuracy. Freeze this
selection before running the selected new candidate on the complete test split.
If the control wins, reuse its already measured test result. No test metric
selects the winner. These fixed holdouts have been used in previous experiments;
they are not a fresh untouched test set.

`comparison.json` contains the endpoint comparison, selected full-test result
and reused control test result. No production bundle is automatically replaced.

## Execution and recovery

```sh
python3 training/run_full_learning_rate_comparison.py
python3 training/run_full_learning_rate_comparison.py --resume
```

Artifacts: `training/artifacts/learning-rate-192ch-full-246m-v7-20260923/`.

The launcher prepares the frozen run, then uses the existing learning-rate
coordinator for locking, source/input checks, Metal worker recovery, graceful
signals and completion. Inspect `status.json`, `train-lr-3e-5.log` and
`lr-3e-5/training-state.pt` for progress. Send SIGTERM to the coordinator PID
and wait for `stage: stopped` before shutting down when possible. A crash can
resume the latest checkpoint; work since that checkpoint may be repeated.

The original control took 22.28 hours of recorded training at about 3,070
samples/second. Budget roughly **23–25 hours** for this run and evaluation on
the same machine, excluding sleep, shutdown and substantial competing workloads.
