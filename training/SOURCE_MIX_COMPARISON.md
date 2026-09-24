# Source composition pilot — 2026-09-24

The [expanded source review](SOURCE_TEACHER_AUDIT_R2.md) motivates a controlled
trial, not a conclusion that whole web sources are unusable. The user approved
comparing the existing mixture against the cleaner-looking candidate sources.
The pilot is complete. **Neither continuation beats the inherited model on the
precommitted validation selection score.** Removing the web/old small sources
does not improve overall accuracy in this run; the original backend is retained.
The background run started on 2026-09-24 at 12:46 China time. Matched data
preparation passed at 12:50 and the first Metal worker started immediately.
B was safely paused at 9,375,222 examples and resumed from the saved optimizer
and next batch. Final reports completed at **15:25:30 China time**. Both arms
trained exactly one pass; no second epoch was started.

## Results

All rows below use the same complete **2,522,347-example validation set**.
The A/B rows report the actual epoch-1 endpoints, not their selected epoch-0
checkpoints. Macro accuracy averages the six evaluation domains equally;
selection is half overall and half macro accuracy.

| Model | Overall accuracy | Macro accuracy | Selection score | Validation loss |
| --- | ---: | ---: | ---: | ---: |
| Inherited model | **89.2357%** | 89.2437% | **89.2397%** | 0.328032 |
| A: current mixture, epoch 1 | 89.1853% | **89.2553%** | 89.2203% | **0.325393** |
| B: Wikipedia + synthetic, epoch 1 | 89.1624% | 89.2083% | 89.1853% | 0.338851 |

A loses **0.0505 percentage points** of overall accuracy versus initialization;
B loses **0.0733 points**. B is **0.0229 points** below A (577 fewer exact matches).
These are small differences from one seed, not a statistical source ranking.
B's validation loss also increases versus initialization. Its lower training
loss cannot be compared directly with A's: the examples and effective source
loss weights differ.

| Validation domain | Inherited | A endpoint | B endpoint | B minus inherited, pp |
| --- | ---: | ---: | ---: | ---: |
| Academic | 91.2031% | 90.6856% | 90.5563% | -0.6468 |
| Dialogue | 93.5005% | 93.2259% | 93.4273% | -0.0732 |
| Encyclopedia | 89.1233% | 89.1233% | 89.6380% | +0.5147 |
| News | 82.5600% | 83.5200% | 82.4000% | -0.1600 |
| Web | 89.1084% | 89.0578% | 88.9757% | -0.1327 |
| Wikipedia | 89.9666% | 89.9191% | 90.2523% | +0.2857 |

B improves the Wikipedia and encyclopedia domains while the other four decline
versus initialization. This is consistent with specialization toward the
retained sources, not evidence of a broadly superior teacher mixture. This
pilot **does not justify dropping all web sources**. It also does not establish
that the observed damaged web examples are harmless, that every web source is
useful, or that Wikipedia/synthetic-only training from scratch would fail.
The audit and this experiment answer different questions.

Both arms select **epoch 0** under the frozen rule, and `selection.json` selects
`initial`. Accordingly, neither new endpoint is evaluated on test. The
**89.3738%** test accuracy recorded in `best_state_tests` is the existing
inherited-model result on all 2,569,996 test examples, reused for both selected
epoch-0 checkpoints. It is **not a measured test score for either new endpoint**.
No natural-reading accuracy improvement is claimed from this proxy-label test.

A records 69.24 training minutes and B 61.66 minutes, excluding pauses, data
preparation and validation. Both had identical 51,274 batch counts; these times
are not a controlled speed benchmark. Epoch validation and the final validation
recheck match exactly for both arms, initial validation matches the release,
and the production bundle hash is unchanged. The coordinator and worker exited
normally after writing `comparison.json`.

## Experimental controls

| Control | Both arms |
| --- | --- |
| Initialization | Current released best: `learning-rate-192ch-full-246m-v7-20260923/epoch-1-backend` |
| Model | 192 channels, 8 residual blocks / 16 convolutions, 8,192-character vocabulary |
| Training | 10,000,000 existing training pairs, one pass per arm |
| Optimizer | Fresh AdamW, lr 0.00003, weight decay 0.0001, gradient clip 1 |
| Weighting | Existing position weights and domain-frequency exponent 0.65 |
| Execution | Float32 Metal, sequential arms, 512 examples / 8,192 tokens per batch |
| Validation | All 2,522,347 fixed examples, all existing evaluation domains |
| Test | All 2,569,996 fixed examples; selection frozen before testing |
| Seed | 2026092407 |

**A — `current-mix`:** approximately the existing raw source proportions,
including both 100-million-row web generations and all old domains.

**B — `wiki-synthetic`:** only Wikipedia and synthetic multistyle pairs. These
sources are candidates suggested by the audit; they are not certified clean.
Their relative proportions within each matched stratum follow availability.

Preparation selects a seeded, source-directory-stratified pool of approximately
12 million A candidates and counts all existing Wikipedia/synthetic training
pairs for B. Capped proportional allocation gives both arms exactly 10 million
rows on their common support. Sampling is uniform without replacement within
each joint **exact character length / existing length bucket / target-position
decile**. If common support is insufficient, preparation fails instead of
silently changing the budget. Rare unmatched length/position cells are dropped;
the actual source proportions are recorded.

Each matched cell is distributed across the same 128 output shards by the same
deterministic rule. The arms therefore have identical per-shard length
histograms, examples, total tokens, position-decile counts, and optimizer-step
counts under the unchanged length-bucket batcher. Text and target indices stay
unchanged. B's compact domain IDs are remapped to its two-entry domain list.
No raw corpus is regenerated, repaired, or deleted.

The prepared arms each contain **10,000,000 pairs, 290,089,694 characters and
51,274 optimizer batches** across 128 shards. There were 11,827,156 A-pool
candidates on common support. The selected source counts are:

| Source | A: current mixture | B: Wikipedia + synthetic |
| --- | ---: | ---: |
| Web | 8,168,643 | 0 |
| Wikipedia | 845,450 | 4,717,260 |
| Synthetic multistyle | 978,350 | 5,282,740 |
| News / academic / encyclopedia / dialogue | 7,557 | 0 |

After the unchanged domain and position weighting, A's effective loss shares
are approximately 48.65% web, 20.41% Wikipedia, 22.61% synthetic and 8.34% small
old sources. B's are 48.42% Wikipedia and 51.58% synthetic. Raw row proportions
are therefore not the same as their contributions to the training objective.

The unchanged inverse-frequency weighting formula produces different domain
weights when sources are removed. `data-verification.json` records both actual
weights and effective loss shares. This tests **the source-mixture change under
the existing training method**, not teacher quality in isolation. Position
weight tables match exactly because their cell counts match.

Other limits:

- Compact training rows do not retain the original proxy glyph or document ID.
  Proxy-type matching is therefore **not** claimed. Length and position matching
  cannot eliminate differences in punctuation type, topic, style, or difficulty.
- Both arms inherit substantial previous mixed-source training. Improvement on
  B would support further continuation on that mixture; it would not establish
  that training without web data from scratch is best.
- Both subsets come exclusively from the existing protected v7 training split.
  Validation/test are not resplit or used to generate new training examples.
  These holdouts have already been consulted in previous experiments; the test
  is a fixed regression benchmark, not a fresh untouched final test.
- One seed and one short continuation do not establish a universal source ranking.

Selection uses the existing half-overall, half-macro-domain validation score,
including the inherited model as a candidate. Report both **epoch-1 endpoints**
even if they lose to initialization. Freeze that comparison and selection
before evaluating either arm's validation-selected best checkpoint on test.
If an arm retains epoch 0, reuse the identical initial model's measured test
result and label it as such; do not present it as the epoch-1 endpoint result.
Test results do not change selection. Before a source policy or backend change,
review per-domain regressions and real reading cuts using the existing user
standard: broken phrases and cross-clause misattachment are both unacceptable.

Run and recovery:

```sh
python3 training/run_source_comparison.py
python3 training/run_source_comparison.py --resume
```

Artifacts live in
`training/artifacts/source-mix-vs-wiki-synthetic-10m-v7-20260924/`:

- `run.json`: frozen input/source hashes, configuration and protocol.
- `status.json`, `coordinator.log`, `train-<arm>.log`: live stage and progress.
- `data-verification.json`: source counts, weighting, sample/token/step matching.
- `<arm>/training-state.pt`: resumable weights, optimizer, shard and next batch.
- `validation-progress.json`, `selection.json`, `comparison.json`: results.

SIGTERM to the coordinator requests a safe stop and is forwarded to training.
The trainer saves at safe batch boundaries and after every shard; a hard reboot
can lose work since the last checkpoint. Preparation caches completed shard
counts; an interrupted sampling arm restarts its own output files. Completed
data and checkpoints are hash-checked on resume. The frozen coordinator and
training source are used after restart, including automatic Metal worker
refresh when it saves a checkpoint under memory pressure. `caffeinate` keeps an
open, powered laptop awake while running; it does not promise training through
lid closure or shutdown.

Validation of the experiment implementation:

```sh
python3 -m unittest discover -s training -p 'test_source_comparison.py' -v
```

The tests exercise exact capped allocation, within-stratum sampling without
replacement, preservation of labels/domain identity, matching of shard lengths
and batch counts, saved-manifest reload, changed-input rejection, and identical
training commands except the dataset/output paths.
