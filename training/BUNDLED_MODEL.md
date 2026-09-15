# Bundled boundary model

`src/boundary-model-data.js` contains the completed **epoch 1 best_state** from
`unicode-context-192ch-16conv-20260914`, exported on 2026-09-15 while epoch 2
continued training. This full-corpus continuation starts from the completed
12-layer epoch-2 best weights and trains on the same data; the newly downloaded
Ultra-FineWeb corpus is not yet included.

| Property | Value |
| --- | ---: |
| Input representation | unicode-context-v1 |
| Channels | 192 |
| Residual blocks / convolution layers | 8 / 16 |
| Parameters | 3,496,329 |
| Structural gap receptive field | Up to 34 tokens, 17 on each side |
| Vocabulary entries | 8,192 |
| Training examples per epoch | 74,121,940 |
| Full validation examples | 727,578 |
| Validation accuracy | 89.1495% |
| Mean domain validation accuracy | 86.0492% |
| Validation selection score | 87.5993% |
| Test accuracy | Not evaluated for this interim export |

Accuracy measures recovery of hidden punctuation boundaries. Historical Han-only
scores used different inputs and holdouts and are not directly comparable.

Checkpoint SHA-256:
`6027142f074a5609495b81cbee68044e1ff6d908ce8d4ffe333165bb8c007871`

The frozen release artifacts are in
`training/artifacts/unicode-context-192ch-16conv-20260914/epoch-1-backend/`.
`source-training-state.pt` captures the live checkpoint by reading one open file;
`selected-state.pt` contains its completed best weights. `smoke-metrics.json`
records the selection and source SHA-256
`ec63d5f420b67bee1b533024a3d559edabfd9ea8e3d3cf2ba8a0187b6c7c1402`.
The release uses `best_state`, never the partially trained epoch-2 `model_state`.

`npm run model:export` reproduces the bundle from these local release artifacts.
The script is 18,740,840 bytes, with SHA-256
`590221a6dc0622f253f93480d69c26d2ffeb64d4db4d41bc933c9d97444ac0bd`.
A clean checkout includes the exported model and
test references; runtime and ordinary model tests require no training artifacts or PyTorch.

`test/model-backend-reference.json` contains 11 independent PyTorch float32 cases,
including times, numbers, Latin text, quotes, supplementary Han, emoji and a
two-character input. JavaScript selects the same best gap in all cases.
Maximum logit difference is 0.00004578; maximum softmax probability difference is
0.00000195. Tests check absolute-plus-relative float32 tolerances, probabilities
and selected gaps.

An isolated headless Chrome 152 check loads the production inference service and
Worker. Six inputs cover short-clause gating, long clauses, enumeration items,
times, normalized numbers, emoji and supplementary Han. Cold and warm Worker
results match Node, with valid UTF-16 offsets and original text preserved. The
first service request, including model loading, took 325 ms while training was
running, within the service's 5-second timeout. This is a smoke measurement, not
a general latency guarantee; details are in `chrome-worker-verification.json`
in the frozen release directory.

With confidence abstention disabled, three model-output snapshots changed with this promotion: the divider example
now starts `同一个无标点子句｜内的短语块`; the Euler example keeps
`用"无数个小矩形累加"｜来近似积分` together around the quoted phrase; the driving
example becomes `驾驶员会出于本能｜进行｜向左打方向盘等避险动作`. The last example
still illustrates that short fragments can occur. These snapshots record runtime
behavior, not human-labeled quality targets; validation accuracy alone does not
establish that every full-sentence segmentation improves.

## Backend preprocessing

1. Classify proxy punctuation after NFKC normalization, preserving original text
   and UTF-16 offsets.
2. Pre-split on the proxy punctuation in `text-policy.json`: commas, periods,
   exclamation/question marks, semicolons and ellipses. Also pre-split on enumeration
   commas (`、`, including NFKC-equivalent forms) as a backend rule. Periods and
   commas between digits remain inside numbers. Other punctuation and whitespace
   stay in the clause; enumeration commas remain excluded from training proxy labels.
3. Leave clauses of at most 12 visual units intact. A Han character, numeric
   expression or Latin word contributes one unit; this is not a 12-token cap.
   Enumeration items follow this same threshold, with no additional list protection.
4. Send the remaining longer clauses to the model with their context characters.
   Normalize input and whitespace, hide pre-split proxy punctuation, and preserve
   numeric separators. Word/quantity protections apply.
   Opening and closing marks stay attached to adjacent content at model cuts.
5. Cache the original window logits. For each fragment above 12 visual units,
   compute softmax over its internal gaps and allow probabilities strictly above 75%.
   Rank eligible gaps by raw model logit, breaking ties by the earlier gap.
   Recurse into both children using the same logits and a new child softmax;
   removing protected gaps never renormalizes probabilities. An uncertain
   fragment stays intact even above 12 visual units. The threshold measures relative model
   confidence, not a calibrated probability that a cut is appropriate.

Rendered text retains its original punctuation, spacing and character widths.
Only model-selected cuts inside a clause receive visual dividers.

By default the chunker scores the clause once and reuses its logits while
recursively choosing the highest-scoring eligible gap in each fragment.
The default [recursive softmax](RECURSIVE_SOFTMAX.md) requires no Worker options
and never runs CNN inference on a child fragment.

A separate [recursive >90% trial](RECURSIVE_CONFIDENCE_EXPERIMENT.md) is available
with `{ minConfidence: 0.9, scoringStrategy: "recursive-model" }`. It reruns the
model on longer child fragments while preserving tokenization and protection ranges.

The subsequent [confidence audit](CONFIDENCE_THRESHOLD.md) measures the precision,
recall and coverage tradeoff on full validation for original-input top-1 predictions;
its precision figures do not establish the accuracy of recursive child decisions.

The completed [architecture smoke](CNN_TRANSFORMER_EXPERIMENT.md) was a separate
experiment; its scripts and generated artifacts have been removed. The 16-layer
continuation uses its own frozen training/export sources, so this backend release
does not change the running experiment. Completing epoch 2 does not automatically
replace this bundle.
