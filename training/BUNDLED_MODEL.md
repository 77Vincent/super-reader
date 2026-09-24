# Bundled boundary model

`src/boundary-model-data.js` contains the completed **epoch 1 best_state** from
`learning-rate-192ch-full-246m-v7-20260923/lr-3e-5`, promoted on 2026-09-24.
It won the full-data learning-rate comparison by validation selection score.
Both arms started from the same preceding 100-million-web epoch-2 weights with
fresh AdamW optimizers and used identical data, order, seed, batch limits and
training steps. Learning rate alone changes from 0.0003 to **0.00003**.
Training and both full holdouts use the existing v7 cleaning policy.

| Property | Value |
| --- | ---: |
| Input representation | unicode-context-v1 |
| Channels | 192 |
| Residual blocks / convolution layers | 8 / 16 |
| Parameters | 3,496,329 |
| Structural gap receptive field | Up to 34 tokens, 17 on each side |
| Vocabulary entries | 8,192 |
| Training examples per epoch | 246,266,210 |
| Web / other training examples | 200,000,000 / 46,266,210 |
| Full validation examples | 2,522,347 |
| Validation accuracy | 89.2357% |
| Mean domain validation accuracy | 89.2437% |
| Validation selection score | 89.2397% |
| Full test examples | 2,569,996 |
| Test accuracy | 89.3738% |

The previous bundle scored 88.7786% validation and 88.8945% test accuracy on
these same holdouts. Gains are **0.4571 and 0.4793 percentage points** respectively.
Accuracy measures recovery of the single hidden punctuation boundary; it is not
precision at the backend's 50% threshold or accuracy of recursive child cuts.
Selection uses validation, followed by full-test evaluation. These holdouts have
been reused across experiments; they are not a new untouched test set.
See [FULL_LEARNING_RATE_COMPARISON.md](FULL_LEARNING_RATE_COMPARISON.md).

The frozen release is
`training/artifacts/learning-rate-192ch-full-246m-v7-20260923/epoch-1-backend/`.
`verify_release.py` checks every exported parameter against `best_state`, accounting
for convolution export layout, and generates independent PyTorch Metal float32
references. `release-verification.json` records hashes and data identity;
`smoke-metrics.json` contains the full validation/test results despite its filename.

- Training checkpoint SHA-256: `5169707d95d51fe2515496f372f8ad52197f531f37d84f4b6775d32a5b9e7fca`.
- Exported safetensors SHA-256: `d7a19c35d73bb73d0bb346cd13e7c84154ba9545b311c8064649b21d83144aa5`.
- Browser bundle SHA-256: `bb90cb253f364e7b0e923455366224ed78c8612034efc6ae81c1ff8a34664081`.
- Browser bundle size: 18,740,854 bytes.

`npm run model:export` reproduces this bundle from the local frozen release.
A clean checkout includes the exported model and independent test references;
runtime and ordinary model tests require no training artifacts or PyTorch.

The 11 independent Metal reference cases cover times, numbers, Latin text,
quotes, supplementary Han, emoji and a two-character input. JavaScript chooses
the same best gap in every case. Maximum absolute logit difference is
0.000305176, maximum relative difference 0.000001143, and maximum softmax
probability difference 0.000002165, all within the existing tolerances.

Model-dependent snapshots were updated without changing runtime rules. At the
default 50% threshold the unpunctuated demo now omits the previous `句子｜模型`
cut, while retaining `模型｜依然`. With abstention explicitly disabled, the
phrase example changes from `短语｜块` to a cut before `内的短语块`; the reading
example still cuts at `信息｜结构`. These are observations, not human-reviewed
quality targets; aggregate improvement does not fix every sentence.
See [MODEL_ERROR_AUDIT.md](MODEL_ERROR_AUDIT.md) for the full-test error audit.

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
   compute softmax over its internal gaps and allow probabilities strictly above 50%.
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
experiment; its scripts and generated artifacts have been removed. The full-data learning-rate comparison completed epoch 1 using its own frozen
training/export sources; this promotion does not resume or modify that run.
Completing an epoch does not
automatically replace this bundle.
