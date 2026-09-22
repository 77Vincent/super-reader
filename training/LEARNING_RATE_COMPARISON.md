# Lower-learning-rate continuation pilot — 2026-09-22

Test fixed learning rates **0.0001** and **0.00003** against the completed
**0.0003 / 192-channel** arm of the [width comparison](WIDTH_COMPARISON.md).
This experiment changes only learning rate. It does not add data, change the
model architecture, or introduce a learning-rate schedule.

Each new arm independently inherits the same best weights from
`chinese-line-web-200m-16conv-v7-20260920/epoch-1-backend` and starts a fresh
AdamW optimizer, matching the control. Both train one epoch on the exact same
**10,015,214 pairs / 99 shards**, with the same seed, order, batches, sample
weights and gradient clipping. The model remains 192 channels, 16 convolutions,
8,192 vocabulary entries and 3,496,329 parameters. The existing 0.0003 endpoint
is reused rather than retrained. Its validation accuracy is 88.7894% and
selection score is 88.6793%; the inherited weights' score is 88.8306%.

## Evaluation and selection

All **2,522,347 validation pairs** remain fixed and are evaluated for both
candidates. Record both the one-epoch endpoint and whether it beats the inherited
weights; do not silently replace an unsuccessful endpoint with its initialization
in the comparison table. Select with the existing score:

```text
0.5 * overall validation accuracy + 0.5 * macro-domain validation accuracy
```

The inherited weights are also eligible; ties retain the existing winner.
Write `selection.json` before evaluating the selected new candidate on all
**2,569,996 test pairs**. If the inherited weights or the existing control win,
reuse their already measured test result. No test accuracy participates in
selection. These holdouts have been evaluated in earlier experiments and are
not a newly untouched test set.

`train_sharded.py --defer-test` saves `validation-only.json` after the ordinary
optimization/checkpoint loop, without loading test records. Resume the selected
arm without this flag to run its normal validation-selected final evaluation.
The flag is deliberately not part of checkpoint optimization configuration.
The coordinator verifies that the training code before finalization is
semantically identical to the frozen control, and hashes the initialization,
vocabulary, all selected shards, holdouts and source snapshot.

## Run and resume

```sh
python3 training/run_learning_rate_comparison.py
python3 training/run_learning_rate_comparison.py --resume
```

Outputs are under `training/artifacts/learning-rate-192ch-10m-v7-20260922/`.
Training runs sequentially on Metal, with no CPU training fallback. Each arm
saves after every shard; memory-pressure worker refreshes resume the saved next
batch. Send SIGTERM to the coordinator PID recorded in `status.json` for a
graceful pause, and wait for `stage: stopped` before shutting down. A crash may
replay work since the latest checkpoint; interrupted validation restarts.

Monitor `train-lr-1e-4.log`, `train-lr-3e-5.log` and `validation-progress.json`.
The final `comparison.json` records both endpoints, the chosen checkpoint and
its full test result. The experiment never promotes weights to the backend.

The reference 192-channel arm took 57.5 minutes of pure training. Two new arms,
full validation and the selected test are estimated at roughly 2.5–3 hours on
the same machine, excluding sleep and substantial competing workloads.

This is a single-seed continuation on previously seen training data. An apparent
gain should be checked in a longer continuation before claiming a durable
improvement or changing the production model. No results are available yet.
