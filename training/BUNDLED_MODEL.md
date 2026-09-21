# Bundled boundary model

`src/boundary-model-data.js` contains the completed **epoch 2 best_state** from
`chinese-line-web-100m-16conv-v7-20260917`, exported on 2026-09-21. It is the
best completed checkpoint by the run's validation selection score. Training,
validation and test data use the v7 cleaning policy. The separate expansion to
200 million web training pairs is still training and has not replaced this model.

| Property | Value |
| --- | ---: |
| Input representation | unicode-context-v1 |
| Channels | 192 |
| Residual blocks / convolution layers | 8 / 16 |
| Parameters | 3,496,329 |
| Structural gap receptive field | Up to 34 tokens, 17 on each side |
| Vocabulary entries | 8,192 |
| Training examples per epoch | 146,266,210 |
| Web / other training examples | 100,000,000 / 46,266,210 |
| Full validation examples | 2,522,347 |
| Validation accuracy | 88.2298% |
| Mean domain validation accuracy | 88.1696% |
| Validation selection score | 88.1997% |
| Full test examples | 2,569,996 |
| Test accuracy | 88.3654% |

Accuracy measures recovery of the single hidden punctuation boundary in each
sample. It does not measure precision at the backend's 75% confidence threshold
or the accuracy of recursive cuts. On these same full datasets, epoch 1 scored
88.0176% validation accuracy and 88.1614% test accuracy; epoch 2 improves these by
0.2122 and 0.2040 percentage points. Selection uses validation results, not test
results. These aggregate gains do not imply every domain or sentence improves.
Older bundled-model figures used different data and cleaning policies and are
not directly comparable.

The frozen release artifacts are in
`training/artifacts/chinese-line-web-100m-16conv-v7-20260917/epoch-2-backend/`.
`training-state.pt` preserves the completed source checkpoint. Its SHA-256 is
`eaa1fecea56261d85a55664eb2e2ceefa18d00cd270d2d3ed90ea741eb33df5c`.
`verify_release.py` checks every exported parameter against `best_state`, accounting
for the convolution export layout, and generates independent PyTorch Metal (MPS)
float32 references. `release-verification.json` records the hashes and data identity;
`smoke-metrics.json` contains the full validation and test results despite its
historical filename.

Exported safetensors SHA-256:
`264eb00ba804f80997272750036f10226dd1ffd5ca89ab7d6faacdc0745fd255`.

`npm run model:export` reproduces the bundle from these local release artifacts.
The script is 18,740,854 bytes, with SHA-256
`7d7417038f73f3c0689089e26d77db1082fbb6f0c5d22043dc01870ed8d8c803`.
A clean checkout includes the exported model and test references; runtime and
ordinary model tests require no training artifacts or PyTorch.

`test/model-backend-reference.json` contains 11 independent PyTorch Metal float32
cases, including times, numbers, Latin text, quotes, supplementary Han, emoji and
a two-character input. JavaScript selects the same best gap in all cases.
Maximum raw-logit difference is 0.00134278 (on a logit of magnitude about 1,172);
maximum softmax probability difference is 0.00000298. Tests check absolute-plus-relative
float32 tolerances, probabilities and selected gaps. The Worker tests run the
actual production Worker scripts in a JavaScript VM, including the demo's recursive
75% behavior, mixed Unicode, long strings and ordered results.

Backend rules are unchanged by this promotion, but model-dependent snapshots
change. At the default 75% threshold, `我一直在思考明天早上的早餐吃什么`
stays intact: its best gap after `思考` has 53.74% confidence. The warning example
now begins `在没有人｜被人特别留意的情况下，` while still avoiding cuts next
to `（切勿模仿）`. These are observed outputs, not human-labeled quality targets;
short or awkward fragments remain possible despite aggregate validation improvement.
The old model's confidence precision/recall figures have not been remeasured for
this checkpoint.

## Backend preprocessing

1. Classify proxy punctuation after NFKC normalization, preserving original text
   and UTF-16 offsets.
2. Pre-split on the backend punctuation rules: commas, periods,
   exclamation/question marks, semicolons and ellipses. Also pre-split on enumeration
   commas (`、`, including NFKC-equivalent forms) as a backend rule. Periods and
   commas between digits remain inside numbers. Other punctuation and whitespace
   stay in the clause. Training v7 identifies its Chinese proxy punctuation before
   NFKC normalization; the runtime uses its own broader punctuation rules and does
   not read `text-policy.json`. This promotion does not change either policy.
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
200-million-web-pair continuation uses its own frozen training/export sources, so
this backend release does not change that run. Completing an epoch does not
automatically replace this bundle.
