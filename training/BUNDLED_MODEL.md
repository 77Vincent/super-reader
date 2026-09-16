# Bundled boundary model

`src/boundary-model-data.js` contains the completed **epoch 1 best_state** from
`web-mix-20m-192ch-16conv-20260915`, exported on 2026-09-16. This continuation
starts from the previous 16-layer best weights and adds 20 million Ultra-FineWeb
training pairs to the existing corpus. It is the latest completed, validated best
checkpoint; the separate 40-million-web-pair expansion is still training epoch 1.

| Property | Value |
| --- | ---: |
| Input representation | unicode-context-v1 |
| Channels | 192 |
| Residual blocks / convolution layers | 8 / 16 |
| Parameters | 3,496,329 |
| Structural gap receptive field | Up to 34 tokens, 17 on each side |
| Vocabulary entries | 8,192 |
| Training examples per epoch | 94,121,940 |
| Full validation examples | 1,158,011 |
| Validation accuracy | 87.4859% |
| Mean domain validation accuracy | 86.5490% |
| Validation selection score | 87.0174% |
| Test accuracy | Not evaluated for this interim export |

Accuracy measures recovery of hidden punctuation boundaries. On this same full
validation set, the previous bundled model scored 86.3990%; the gain is 1.0869
percentage points. Its previously reported 89.1495% used only the original
727,578 validation examples, so that number is not directly comparable with the
expanded set. The new set adds 430,433 held-out web examples, and its SHA-256 is
`46b6643fbe5511eacd67e4200b4fa55fbdf3b49c8bb71119270a582bbeedc43d`.

Checkpoint SHA-256:
`681666f1105f91e32e5d4fc2b9b7e5a2090cb794ef137d01c700b3e9f7d553c4`

The frozen release artifacts are in
`training/artifacts/web-mix-20m-192ch-16conv-20260915/epoch-1-backend/`.
`source-training-state.pt` preserves the stopped 20-million run's checkpoint;
`selected-state.pt` is the frozen selection also used to initialize the expanded
run, with SHA-256
`0dc993a820dc151ef8bc0bf2232722179a253fec443e8a5a11411cc172f191d2`.
The export checks every parameter against the source's `best_state`, rather than
using its partially trained epoch-2 `model_state`. `smoke-metrics.json` records
the source hashes, data identity and validation results; `export_selected.py`
records the selection and reference-generation procedure.

`npm run model:export` reproduces the bundle from these local release artifacts.
The script is 18,740,840 bytes, with SHA-256
`ad6f0f8daa93cba44cded73facbf645622599d39ff83d763d77281e62b5cdaeb`.
A clean checkout includes the exported model and
test references; runtime and ordinary model tests require no training artifacts or PyTorch.

`test/model-backend-reference.json` contains 11 independent PyTorch float32 cases,
including times, numbers, Latin text, quotes, supplementary Han, emoji and a
two-character input. JavaScript selects the same best gap in all cases.
Maximum logit difference is 0.00003815; maximum softmax probability difference is
0.00000202. Tests check absolute-plus-relative float32 tolerances, probabilities
and selected gaps.

An isolated headless Chrome 152 check loads the production inference service and
Worker. Six inputs cover short-clause gating, long clauses, enumeration items,
times, normalized numbers, emoji and supplementary Han. Cold and warm Worker
results match Node, with valid UTF-16 offsets and original text preserved. The
first service request, including model loading, took 332 ms while training was
running, within the service's 5-second timeout. This is a smoke measurement, not
a general latency guarantee; details are in `chrome-worker-verification.json`
in the frozen release directory.

Backend rules are unchanged by this promotion, but model-dependent snapshots
change. At the default 75% threshold, `我一直在思考明天早上的早餐吃什么`
stays intact: its best gap after `思考` has 53.21% confidence. The warning example
now begins `在｜没有人｜被人特别留意的情况下，` while still avoiding cuts next
to `（切勿模仿）`. With abstention disabled, the driving example becomes
`驾驶员会出于本能进行｜向左打方向盘等避险动作，`.
These are observed outputs, not human-labeled quality targets; short or awkward
fragments remain possible despite aggregate validation improvement. Tests for
bracket protection check bracket-adjacent cuts rather than requiring the entire
input to have no cuts. The old model's confidence precision/recall figures have
not been remeasured for this checkpoint.

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
   numeric separators. Model cuts require immediately adjacent Han characters
   in the original text, as well as browser word protection. This includes
   supplementary Han and treats spaces as non-Han. Dedicated number-interior,
   numeric-attachment, quantity-phrase and quote/bracket attachment rules have been removed.
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

The historical [confidence audit](CONFIDENCE_THRESHOLD.md) measures the previous
model's precision, recall and coverage on its original validation set. Those
figures do not describe this checkpoint or establish the accuracy of recursive
child decisions.

The completed [architecture smoke](CNN_TRANSFORMER_EXPERIMENT.md) was a separate
experiment; its scripts and generated artifacts have been removed. The current
40-million-web-pair continuation uses its own frozen training/export sources, so
this backend release does not change that run. Completing an epoch does not
automatically replace this bundle.
