# Bundled boundary model

`src/boundary-model-data.js` contains the completed **epoch 1 best_state** from
`fresh-lr3e-4-max256-20261005/candidate`, promoted on 2026-10-07.
This run starts from random weights and a fresh AdamW optimizer at learning rate
**0.0003**, with no position or corpus-source weighting. It completed one epoch
on 2026-10-06 and stopped for validation review. The selected model's full test
evaluation and browser export followed on 2026-10-07, without further optimizer steps.

| Property | Value |
| --- | ---: |
| Input representation | unicode-context-v1 |
| Training execution contract | masked-residual-convolutions-v1 |
| Channels | 192 |
| Residual blocks / convolution layers | 8 / 16 |
| Parameters | 3,496,329 |
| Structural gap receptive field | Up to 34 tokens, 17 on each side |
| Vocabulary entries | 8,192 |
| Training examples | 395,216,214 |
| Full retained validation examples | 2,515,459 |
| Validation accuracy | 90.6079% |
| Validation MRR | 0.946725 |
| Full retained test examples | 2,563,530 |
| Test accuracy | 90.6847% |
| Test MRR | 0.947210 |

The prepared corpus remains **unicode-context-v7**. All three splits retain only
samples with at least two characters on each side of the target and at most
256 characters in total. This excludes 1,049,996 training, 6,888 validation and
6,466 test examples. The later v8 cleaning rules were not retroactively applied.
Corpus-source labels describe provenance and do not affect loss or selection.
Loss reporting sums weighted per-example losses over total example weight;
MRR breaks score ties by ascending gap index, consistent with top-1 argmax.

Accuracy measures recovery of the single hidden punctuation boundary. It does
not measure precision at the backend's 45% threshold or correctness of recursive
child cuts. The previous bundled model's 89.4642% validation and 89.5785% test
scores used unfiltered holdouts, so they are not a matched before/after comparison.
The holdouts have also been reused across experiments; they are not newly
collected unseen evaluation data.

## Reproduction and verification

The frozen release is
`training/artifacts/fresh-lr3e-4-max256-20261005/epoch-1-backend/`.
`evaluation-launch.json` records the frozen trainer command: resume the completed
checkpoint with `--epochs 1`, remove `--defer-test`, and use the separate release
directory. The checkpoint is already positioned at epoch 2, so the training loop
is skipped. `smoke-metrics.json` contains the complete validation/test results
despite its filename. Re-evaluated validation exactly matches the reviewed result.

`verify_release.py` checks all 61 exported tensors against `best_state`, including
convolution layout conversion, and confirms that the original training checkpoint
is unchanged. It generates independent PyTorch CPU float64 and Metal float32
references from identical stored float32 weights. `verify_browser.mjs` checks the
JavaScript forward pass against both and records example output changes.

- Training checkpoint SHA-256: `3948c2bc87140806082e3be1a2b2eee57ac8dc7eb76492e5ef56eac2be106ebf`.
- Exported safetensors SHA-256: `dffbde883daf16759b52d4834cd73625eedde291fb114ad17c21c747223c26b3`.
- Browser bundle SHA-256: `d441ad21fcb8a869aac2b43121303149bd073fead4b5b85c9d927939a730cae6`.
- Browser bundle size: 18,740,854 bytes.

`npm run model:export` reproduces the bundle from this local release.
A clean checkout includes the exported weights and independent reference fixture;
ordinary backend tests need no training artifacts or PyTorch installation.

All **264** `npm test` cases pass, including the real Worker script loaded by
the integration harness and the 513-token window/whole-sequence parity check.
The extension release build succeeds at version **0.1.0**; its ZIP includes the
same verified bundle. `promotion.json` and `promotion-tests-final.log` in the
frozen release record these checks.

The 12 reference cases preserve all previous inputs and add a 256-character
sequence. They cover Chinese, supplementary Han, emoji, numbers, times, Latin
text and a two-character input. Every best gap matches both PyTorch references.
Maximum absolute JavaScript/CPU float64 logit difference is
0.000077092; maximum probability difference is
0.000000560. The original logit tolerance
`2e-5 + 4e-6 * abs(reference)` is unchanged. Metal float32 reduction order produces
larger logit differences on mixed-number inputs (up to
0.000610352), while its maximum probability difference remains
0.000002706, below the unchanged `1e-5` probability tolerance.

## Runtime behavior

The model promotion preserved the architecture, input processing and then-current
50% confidence threshold. On 2026-10-09 the runtime threshold was changed to **45%**:
it had the highest first-round F1 among 30%, 35%, 40%, 45% and 50% in a fixed
100,000-row test sample. On the 88,676 runtime-eligible, input-matched rows,
precision was 92.06%, recall 82.78% and F1 87.18% (50% F1: 87.09%).
This is exact recovery of one punctuation-proxy label, not full recursive or
human-rated reading quality. The sample has now been used for threshold selection.
Reproducible results are in `training/artifacts/confidence-30to50-step5-20261009/`.
Training excluded one-character target sides;
this promotion adds no inference-time prohibition on one-character cuts.
Longer inputs still use the existing 256-token windows with convolution halos.

Checkpoint-specific snapshots describe observed output, not human quality labels.
For example, forced splitting now keeps `短语块` whole in the divider example;
at the default confidence threshold that sentence stays intact. The driving
example also stays intact at the default threshold. The long unpunctuated demo
retains its two default cuts. The confidence for
`在没有人｜被人特别留意的情况下` is now about 82.52%, so a 75% threshold accepts
it while a 90% threshold abstains. Aggregate accuracy does not establish that
every sentence or confidence-based decision improved.

## Backend preprocessing

The 2026-09-29 training-input alignment is supplemented by runtime enumeration
pre-splitting on 2026-10-09. See [INPUT_ALIGNMENT.md](INPUT_ALIGNMENT.md).

1. Classify the original Chinese proxy glyphs `，。；！？…` before NFKC,
   preserving original text and UTF-16 offsets. ASCII punctuation, Chinese colons
   and compatibility forms are retained context.
2. Pre-split on these proxies and all training physical line boundaries. Chinese
   commas between numbers remain numeric context, including signed numbers,
   decimals, exponents and surrounding whitespace. Closing quotes/brackets stay
   on their original side of the proxy, matching training fragment extraction.
   Also pre-split at original enumeration commas (`、`) so existing list items
   are processed independently. Keep each comma in the preceding fragment and
   its normalized model tokens. These additional runtime boundaries add no
   visual dividers and do not change training proxies or prepared data.
3. Leave clauses of fewer than 13 visual units intact. A Han character, numeric
   expression or Latin word contributes one unit; this is not a 13-token cap.
4. Send the remaining longer clauses to the model with their context characters.
   Normalize input and within-line whitespace after excluding the raw delimiters,
   and preserve numeric separators. A compatibility glyph that normalizes into a
   proxy-looking character remains a token. Model cuts require immediately adjacent Han characters
   in the original text, as well as browser word protection. This includes
   supplementary Han and treats spaces as non-Han. Dedicated number-interior,
   numeric-attachment, quantity-phrase and quote/bracket attachment rules have been removed.
5. Run fresh inference for each fragment of at least 13 visual units,
   using only that fragment's tokens. Compute softmax over its newly scored internal
   gaps. Require probabilities strictly above **45%** for every eligible fragment;
   an explicit `minConfidence` overrides this default without bypassing the length gate.
   Rank eligible gaps by raw model logit, breaking ties by the earlier gap.
   Recurse into both children, rerunning the CNN only if they have at least 13 visual units;
   removing protected gaps never renormalizes probabilities. An uncertain
   fragment stays intact regardless of length. The threshold measures relative model
   confidence, not a calibrated probability that a cut is appropriate.

The current default is at least 13 visual units and >45%, selected for the highest
first-round F1 in the tested 30–50% sweep. This does not establish an optimum for
complete recursive segmentation. That sweep predates the runtime enumeration
pre-splitting; its metrics have not been remeasured for this preprocessing change.

Rendered text retains its original punctuation, spacing and character widths.
Only model-selected cuts inside a clause receive visual dividers.

Inputs over 256 tokens are scored in bounded windows with a full convolution
halo around each owned gap. For this kernel-3, dilation-1, 16-convolution model,
the halo is 16 tokens on each side. Each gap retains one score, avoiding missing
context at internal seams. At most 256 tokens still enter any model call;
each shorter fragment uses one call when it needs scoring. The true CNN's 512 gap logits on a 513-token
fixture exactly match whole-sequence inference in the regression test.

The default `recursive-model` strategy requires no Worker options. New child
boundaries change the model's context and padding; parent logits are not reused
to decide child cuts. Tokenization, source offsets and word protection are kept
from the original clause. An explicit stack avoids JavaScript recursion limits.
This change adds inference calls without changing model weights or training.
The former [cached-logit strategy](RECURSIVE_SOFTMAX.md) and fixed original
probabilities remain explicit offline comparison options.

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
