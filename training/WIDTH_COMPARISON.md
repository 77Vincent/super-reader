# 192 versus 256 channels: continuation pilot

The September 22 pilot tests whether widening the current best CNN improves
boundary prediction enough to justify its extra training and inference cost.
The production bundle is not changed by this experiment.

Both arms start with the function learned by the completed 200-million-web
continuation (`chinese-line-web-200m-16conv-v7-20260920/epoch-1-backend`). They
keep 16 convolutions, kernel size 3, 8 residual blocks, the same 8,192-character
vocabulary and 34-character gap receptive field. The parameter counts are
3,496,329 and 5,513,737, an increase of 57.7%.

## Fixed data and training

The selected pilot has **10,015,214 training pairs** in 99 existing shards.
Shards were originally assigned by content hash; seed `2026092201` selects
shards within each source directory, including both web-data generations.
This avoids taking a Wikipedia-only prefix. Counts are approximate rather
than exactly 10 million because whole shards are retained unchanged.

| Source | Training pairs |
| --- | ---: |
| News | 281 |
| Academic | 392 |
| Encyclopedia | 3,526 |
| Dialogue | 3,169 |
| Wikipedia | 835,641 |
| Synthetic multistyle | 967,947 |
| Web, both generations | 8,204,258 |
| Total | 10,015,214 |

The two arms use the exact same pilot manifest, shard bytes, token IDs, sample
order and batch construction. Each trains one complete pilot epoch, so they
receive the same number of optimizer updates. Both use fresh AdamW, learning
rate 0.0003, weight decay 0.0001, gradient clipping 1, batch cap 512, token cap
8,192, domain-weight power 0.65 and the same position-weight calculation.
Weight tables are computed from the pilot's actual counts in both arms.

Full validation **2,522,347** and test **2,569,996** remain byte-identical to the
existing fixed holdouts. Neither is capped or sampled. Validation's checkpoint
selection score remains half overall accuracy and half macro-domain accuracy.
Test data do not select checkpoints.

Training is sequential on Metal: 192 channels, then 256 channels. Checkpoints
are saved after each shard and on a graceful stop. Memory-pressure worker
refreshes resume the same next batch automatically.

## Preserving the initial predictions

Naively copying a narrow model into a wider model changes LayerNorm's mean
and variance. `widen_boundary.py` instead transforms each residual vector
`x` from width `C` to width `D` as:

```text
a = sqrt(D / C)
T(x) = a * [x, mean(x), ..., mean(x)]
```

Its mean is `a * mean(x)` and variance is exactly `variance(x)`. The original
LayerNorm gamma is divided by `a`, preserving its original output even with
epsilon. The first convolution preserves its original output features; the
second maps the residual update through `T`. The gap head compensates for
`a` on left/right/difference features and `a²` on the product features.
New intermediate features start independently randomized, with zero outgoing
weights. Gradient checks verify that their outgoing weights learn immediately
and their incoming weights learn after the first update. No layers are frozen.

Preflight checks masked logits, probabilities and best gaps on the current
reference texts, and discards all gradient-probe updates. Separate regression
tests cover learned LayerNorm affine parameters, padding, near-zero variance,
and exact loading through the existing trainer. The full initial validation
results are also recorded for both arms.

This is a controlled **warm-start widening experiment**, not a comparison of
architectures trained from scratch or given equal compute. The pilot samples
have already appeared in the inherited model's training corpus. A small or
negative gain in one pilot does not establish a universal capacity limit.

## Run, stop and resume

```sh
python3 training/run_width_comparison.py
python3 training/run_width_comparison.py --resume
```

The default run directory is
`training/artifacts/width-192-vs-256-10m-v7-20260922/`. Its `status.json` records
the coordinator PID and active child. Send **SIGTERM to the coordinator PID**
for a graceful pause; wait for `stage: stopped` before shutting down. An abrupt
shutdown can replay work since the last saved checkpoint. A paused evaluation
restarts that evaluation. Resume uses frozen source, initialization and data
hashes. Logs are `train-192.log` and `train-256.log`.

Each arm's normal outputs retain the validation-selected checkpoint. The
`ch192/final/` and `ch256/final/` exports instead contain each arm's **trained
one-epoch endpoint**, even if validation preferred the initialization. These
endpoints provide the primary equal-update comparison. Full validation/test
results for that endpoint are recomputed if needed.

On completion `comparison.json` records both endpoints' overall and per-domain
accuracy, loss, training speed and accuracy differences. `browser-benchmark.json`
checks the exported JavaScript logits against Metal reference predictions and
times the unchanged inference implementation on identical texts, with warmup
and alternating order. These Node/V8 timings exclude model loading, DOM work
and page rendering; they are not an end-to-end Chrome performance measurement.

Validation:

```sh
PYTHONPATH=training/.deps:training DEBUG=0 python3 -m unittest training/test_width_comparison.py training/test_web_continuation.py
node --check training/benchmark_width.mjs
```
