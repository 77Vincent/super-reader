# Bundled boundary model

`src/boundary-model-data.js` contains the completed **epoch 1 best_state** from
`unicode-context-192ch-12conv-20260913`, exported on 2026-09-14 while epoch 2
continued training. This is the full-corpus model, separate from the architecture smoke.

| Property | Value |
| --- | ---: |
| Input representation | unicode-context-v1 |
| Channels | 192 |
| Residual blocks / convolution layers | 6 / 12 |
| Parameters | 3,052,423 |
| Vocabulary entries | 8,192 |
| Training examples per epoch | 74,121,940 |
| Full validation examples | 727,578 |
| Validation accuracy | 88.1812% |
| Mean domain validation accuracy | 85.0983% |
| Validation selection score | 86.6397% |
| Test accuracy | Not evaluated for this interim export |

Accuracy measures recovery of hidden punctuation boundaries. Historical Han-only
scores used different inputs and holdouts and are not directly comparable.

Checkpoint SHA-256:
`3ac72e4bd965e833273e916f9ee97dd7520340661a7c73e8d95eee3be46a69d4`

The frozen release artifacts are in
`training/artifacts/unicode-context-192ch-12conv-20260913/epoch-1-backend/`.
`source-training-state.pt` captures the live checkpoint by reading one open file;
`selected-state.pt` contains its completed best weights. `smoke-metrics.json`
records the selection and source SHA-256
`f31a02f2258e2e3bdfde3c0c8e3d05d5b7204c6caf26b186b26f29a5b8ecab88`.
The release uses `best_state`, never the partially trained epoch-2 `model_state`.

`npm run model:export` reproduces the bundle from these local release artifacts.
The script is 16,372,300 bytes. A clean checkout includes the exported model and
test references; runtime and ordinary model tests require no training artifacts or PyTorch.

`test/model-backend-reference.json` contains 11 independent PyTorch float32 cases,
including times, numbers, Latin text, quotes, supplementary Han, emoji and a
two-character input. JavaScript selects the same best gap in all cases.
Maximum logit difference is 0.00007630; maximum softmax probability difference is
0.00000243. Tests check absolute-plus-relative float32 tolerances, probabilities
and selected gaps.

## Backend preprocessing

1. Classify proxy punctuation after NFKC normalization, preserving original text
   and UTF-16 offsets.
2. Pre-split on the proxy punctuation in `text-policy.json`: commas, periods,
   exclamation/question marks, semicolons and ellipses. Also pre-split on enumeration
   commas (`、`, including NFKC-equivalent forms) as a backend rule. Periods and
   commas between digits remain inside numbers. Other punctuation and whitespace
   stay in the clause; enumeration commas remain excluded from training proxy labels.
3. Preserve every enumeration item internally, including the first and last,
   regardless of length. Identify lists within the original proxy-delimited clause
   before splitting on enumeration commas, then skip model inference for all their
   items. Thus `A、B、C` becomes `A、`, `B、`, `C`, with no internal model cuts.
4. Leave other clauses of at most 12 visual units intact. A Han character, numeric
   expression or Latin word contributes one unit; this is not a 12-token cap.
5. Send the remaining longer clauses to the model with their context characters.
   Normalize input and whitespace, hide pre-split proxy punctuation, and preserve
   numeric separators. The existing balance and word/quantity protections apply.
   Opening and closing marks stay attached to adjacent content at model cuts.

Rendered text retains its original punctuation, spacing and character widths.
Only model-selected cuts inside a clause receive visual dividers.

The chunker scores the clause once and reuses its logits while
recursively choosing cuts with the current fragment's length balance.
Recursive model rescoring was rejected and removed; see the
[experiment conclusion](SCORING_COMPARISON.md).

The completed [architecture smoke](CNN_TRANSFORMER_EXPERIMENT.md) was a separate
experiment; its scripts and generated artifacts have been removed. The full run's
source guard includes backend files, so a finalizer uses the original
`source-snapshot-before-backend-release-20260914/` for export and evaluation after
the old coordinator reaches that guard. `finalization-handoff.json` tracks this
continuation. Completing epoch 2 does not automatically replace this bundle.
