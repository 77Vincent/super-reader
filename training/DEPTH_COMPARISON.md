# 16/20-layer continuation comparison

Approved 2026-09-22. Reuse the completed **16-layer, 192-channel, learning-rate
0.00003** arm of the [learning-rate comparison](LEARNING_RATE_COMPARISON.md).
Only the new 20-layer arm was trained. Completed 2026-09-23 at 00:12 China time.
**Keep 16 layers for now:** the 20-layer candidate gained only 0.00148 percentage
points on the full test split while JS inference took 22.8% longer in this benchmark.

## Recorded decision — 2026-09-23

Retain **16 layers / 192 channels / a maximum 34-token gap receptive field**.
Increasing depth to 20 layers expanded the maximum gap context to 42 tokens,
but the additional context produced little measured accuracy gain in this
controlled 10,015,214-pair continuation. Long-input and newly covered-input
results also showed small gains. The extra 12.7% parameters and 22.8% JS
inference time do not justify adopting this candidate for production.

Although 20 layers won the prespecified validation score numerically, that
selection does not automatically make it the preferred deployment choice.
Do not promote this checkpoint or expand the 20-layer trial to the full corpus
on the strength of these results. The approved next experiment keeps 16 layers
and [changes only learning rate on the full dataset](FULL_LEARNING_RATE_COMPARISON.md).
This decision applies to the tested initialization, training budget and learning
rate; it does not establish that longer context can never help.

Launched at 22:46 China time. Metal identity/learnability checks and all 13 JS
reference cases passed before the training worker started at 22:47. The final
trained exports also passed all 13 JS reference checks.

| Setting | Reused control | New candidate |
| --- | --- | --- |
| Residual blocks / kernel-3 convolutions | 8 / 16 | 10 / 20 |
| Channels / vocabulary | 192 / 8,192 | Same |
| Parameters | 3,496,329 | 3,940,235 |
| Maximum context at a gap | 34 tokens, 17 per side | 42 tokens, 21 per side |
| Training pairs / epochs | 10,015,214 / 1 | Same |
| Learning rate | 0.00003 | Same |
| Validation pairs | 2,522,347 | Same complete split |
| Test pairs | 2,569,996 | Same complete split, selection protocol below |

Both arms start from the **pre-pilot initialization**, not from the improved
16-layer pilot endpoint. The manifest's 99 shards, shard order, vocabulary,
seed (2026092201), batch limits (512 examples / 8,192 tokens), fresh AdamW,
domain/position weighting and clipping are unchanged. The coordinator copies
the exact frozen trainer used by the control and hashes the inputs. It copies
the control result/checkpoint for analysis without retraining that arm.

Two extra residual blocks start with zero residual scales. A Metal preflight
checks exact copied tensors and identical starting logits on a padded batch
with short, long and mixed-script inputs. Two disposable optimization steps
check that the new scales and convolution branches can learn. All probe updates
are discarded. Both exports must match Metal reference predictions using the
unchanged JS backend before training starts.

## Evaluation and selection

### Completed results

| Metric | 16 layers | 20 layers | Change |
| --- | ---: | ---: | ---: |
| Full validation accuracy | 89.244699% | 89.255047% | +0.010348 percentage points |
| Full test accuracy | 89.361112% | 89.362590% | +0.001479 percentage points |
| Validation selection score | 0.892091706 | 0.892202000 | +0.000110294 |
| Validation loss | 0.3219593 | 0.3218314 | -0.0001279 |
| Test loss | 0.3176330 | 0.3174987 | -0.0001343 |
| Training time | 53.90 min | 62.55 min | +16.1% |
| JS 13-case suite median | 573.82 ms | 704.54 ms | +22.8% |
| Exported model size | 18,740,810 bytes | 21,109,355 bytes | +12.6% |

The validation-only selection chose 20 layers, then its complete test result
was measured. It answered 38 more labels correctly out of 2,569,996 test pairs.
The validation paired audit reproduced both reported accuracies exactly:
1,796 formerly incorrect labels became correct, and 1,535 formerly correct
labels became incorrect, for a net gain of 261 out of 2,522,347 pairs.

| Validation input length | Samples | 16-layer accuracy | 20-layer accuracy | Change (percentage points) |
| --- | ---: | ---: | ---: | ---: |
| 2–17 | 639,215 | 89.3831% | 89.3820% | -0.0011 |
| 18–34 | 1,216,083 | 89.8624% | 89.8693% | +0.0069 |
| 35–42 | 273,464 | 88.7481% | 88.7704% | +0.0223 |
| 43–64 | 270,127 | 87.5662% | 87.5973% | +0.0311 |
| 65–128 | 109,762 | 87.5941% | 87.6232% | +0.0292 |
| 129+ | 13,696 | 84.1925% | 84.2436% | +0.0511 |

Full-input coverage at the gold gap increased from 57.15% to 71.02% of validation
pairs. In the 349,884 newly covered pairs, accuracy rose from 89.6874% to
89.7043% (+0.0169 percentage points; 59 additional correct labels). Thus longer
inputs improved somewhat more, but the absolute benefit remained small.

This short low-LR continuation does not justify the measured inference cost
for production. It does not prove that larger context is never useful or that
the identity-initialized blocks could not benefit from longer/different training.
No production weights were changed, and no further 20-layer training was started.

### Prespecified protocol

The control endpoint reached 89.2447% full-validation accuracy and 89.3611%
full-test accuracy. Select between endpoints using the existing score:
half overall validation accuracy plus half macro-domain validation accuracy.
Ties keep the 16-layer control. Test scores do not select the winner. If the new
20-layer endpoint wins, evaluate its selected epoch-1 weights on the complete
test split; otherwise reuse the already measured control test result.

Additionally compare **all validation samples** in these predefined strata:

- Input lengths: 2–17, 18–34, 35–42, 43–64, 65–128 and 129+ tokens.
- Context at the gold gap: fully covered by 16 layers; newly covered by 20;
  still not fully covered by 20. Coverage requires both sides to fit independently,
  so a short sentence's edge gap can still lack distant context.
- Existing source domains, plus paired counts of corrected and regressed labels.

The paired analysis checks that total correct predictions reproduce each arm's
full-validation report. JS inference uses the same 13 texts, 3 warmups and
8 alternating-order repetitions in isolated Node/V8 contexts. It measures raw
CPU inference, excluding model loading, browser layout and DOM work.

Depth also increases parameter count by about 12.7%; this compares the complete
architecture change and cannot isolate receptive field from added capacity.
One seed and one short continuation are evidence for this setup, not a general
claim that a particular depth is optimal. Production weights remain unchanged.

## Local execution and recovery

Run: `python3 training/run_depth_comparison.py`

Resume: `python3 training/run_depth_comparison.py --resume`

Artifacts: `training/artifacts/depth-16-vs-20-10m-lr3e-5-v7-20260922/`

The coordinator snapshots source files, prevents duplicate runs with a lock,
and keeps the machine awake on power. SIGTERM/SIGINT is forwarded to the trainer,
which checkpoints the next batch at a safe boundary. Unexpected shutdown can
resume from the most recent saved checkpoint; completed stages are reused.
MPS worker refresh uses the existing checkpoint-advance safeguard.

Read `status.json`, `train-layers20.log` and the per-arm checkpoint for progress.
On completion, `comparison.json` combines selection, full validation/test,
`validation-strata.json` and `browser-benchmark.json`. No automatic backend
promotion is performed.
