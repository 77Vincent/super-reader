# Bundled boundary model

`src/boundary-model-data.js` contains the completed **epoch 1 best_state** from
`padding-fixed-web-350m-v7-20260925/candidate`, promoted on 2026-09-27.
The continuation starts from the previously bundled low-learning-rate best
weights, fixes intermediate padding leakage and adds 150 million web pairs.
It trains one epoch at learning rate **0.00003** with fresh AdamW. Training
and both complete holdouts use the existing v7 cleaning policy.

| Property | Value |
| --- | ---: |
| Input representation | unicode-context-v1 |
| Training execution contract | masked-residual-convolutions-v1 |
| Channels | 192 |
| Residual blocks / convolution layers | 8 / 16 |
| Parameters | 3,496,329 |
| Structural gap receptive field | Up to 34 tokens, 17 on each side |
| Vocabulary entries | 8,192 |
| Training examples per epoch | 396,266,210 |
| Web / other training examples | 350,000,000 / 46,266,210 |
| Full validation examples | 2,522,347 |
| Validation accuracy | 89.4642% |
| Mean domain validation accuracy | 89.3941% |
| Validation selection score (overall top-1) | 89.4642% |
| Full test examples | 2,569,996 |
| Test accuracy | 89.5785% |

The previous weights were re-evaluated on the same full validation split with
corrected padding: **89.2074%**. The new epoch gains **0.2569 percentage points**
on this comparable baseline. The previous historical test result (89.3738%)
used the old padding implementation; it was not remeasured with the correction,
so it is not an isolated before/after test comparison. This continuation also
adds training exposure; it does not isolate the effect of data size or padding.

Accuracy measures recovery of the single hidden punctuation boundary, not
precision at the backend's 50% threshold or accuracy of recursive child cuts.
Selection uses overall full-validation top-1, with starting weights eligible;
the selected model is then evaluated on the full test split. These holdouts have
been reused across experiments and are not a fresh untouched test set.
See [PADDING_FIXED_EXPANSION.md](PADDING_FIXED_EXPANSION.md).

The frozen release is
`training/artifacts/padding-fixed-web-350m-v7-20260925/epoch-1-backend/`.
Its `verify_release.py` checks every exported tensor against `best_state`,
accounting for convolution export layout, and generates independent PyTorch
Metal float32 references using the run's frozen, corrected training source.
`release-verification.json` records hashes and data identity;
`smoke-metrics.json` contains full validation/test results despite its filename.

- Training checkpoint SHA-256: `874f451a904a3358c3c45d570fa52af0fb22be604a3f4724df8d8549e68d89aa`.
- Exported safetensors SHA-256: `20ddd336b174cc467d3ff0f3c0b15b0310256e1f212d2e5b4202958870b00144`.
- Browser bundle SHA-256: `ecffb9e81f7001c752b357cc87f41da6bea61d98744c0de0c74912399d72b9d3`.
- Browser bundle size: 18,740,854 bytes.

`npm run model:export` reproduces this bundle from the local frozen release.
A clean checkout includes the exported model and independent test references;
runtime and ordinary model tests require no training artifacts or PyTorch.

The 11 independent Metal reference cases cover times, numbers, Latin text,
quotes, supplementary Han, emoji and a two-character input. JavaScript chooses
the same best gap in every case. Maximum absolute logit difference is
0.000274658, maximum relative difference 0.000001109, and maximum softmax
probability difference 0.000001778, all within the existing tolerances.
All 245 `npm test` cases pass, including the actual worker integration;
`promotion.json` and `promotion-tests.log` in the frozen release record this check.
The corrected trainer masks intermediate padded activations; browser inference
already processes individual unpadded sequences, so no inference-rule changes
are needed. The historical padding diagnosis is in
[GOOD_TEACHER_DIAGNOSIS.md](GOOD_TEACHER_DIAGNOSIS.md).

The checkpoint-specific output snapshots are observations, not human quality
labels. With default 50% confidence, the long unpunctuated demo now has two
cuts (`语法的｜通顺的｜但没有…`) instead of four; the rest stays intact.
`短语块` remains whole in the default divider example. With abstention explicitly
disabled, it can split as `短语｜块`; that is a remaining model error, not a quality
expectation. The forced Euler example changes `来近似｜积分` to `来｜近似积分`,
and the driving example moves a cut from `进行｜向左打方向盘` to `进行向左打｜方向盘`.
The confidence for `在没有人｜被人特别留意的情况下` is now about 53.25%, so the
explicit 75% threshold abstains. Aggregate gains do not establish improvement
on every sentence. That promotion left the runtime threshold, preprocessing and
recursion unchanged. The later preprocessing corrections are described below.

## Backend preprocessing

Updated 2026-09-29 to align with the existing training input contract; weights
and training/evaluation data are unchanged. See [INPUT_ALIGNMENT.md](INPUT_ALIGNMENT.md).

1. Classify the original Chinese proxy glyphs `，。；！？…` before NFKC,
   preserving original text and UTF-16 offsets. ASCII punctuation, enumeration
   commas and compatibility forms are retained context. The former additional
   enumeration pre-splitting rule has been removed.
2. Pre-split on these proxies and all training physical line boundaries. Chinese
   commas between numbers remain numeric context, including signed numbers,
   decimals, exponents and surrounding whitespace. Closing quotes/brackets stay
   on their original side of the proxy, matching training fragment extraction.
3. Leave clauses of at most 12 visual units intact. A Han character, numeric
   expression or Latin word contributes one unit; this is not a 12-token cap.
4. Send the remaining longer clauses to the model with their context characters.
   Normalize input and within-line whitespace after excluding the raw delimiters,
   and preserve numeric separators. A compatibility glyph that normalizes into a
   proxy-looking character remains a token. Model cuts require immediately adjacent Han characters
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

Inputs over 256 tokens are scored in bounded windows with a full convolution
halo around each owned gap. For this kernel-3, dilation-1, 16-convolution model,
the halo is 16 tokens on each side. Each gap retains one score, avoiding missing
context at internal seams. At most 256 tokens still enter any model call;
shorter inputs still use one call. The true CNN's 512 gap logits on a 513-token
fixture exactly match whole-sequence inference in the regression test.

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
