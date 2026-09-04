# Tokenization candidate comparison

Run date: 2026-09-04

This medium smoke test compares two candidate definitions while holding the
source documents, selected pair IDs, gold character boundaries, model, seed,
batching policy, and three training epochs constant.

- `word`: tokenize each punctuation-derived side with `Intl.Segmenter`, then
  score the gaps between adjacent word tokens.
- `character`: discard non-Han content, then score every adjacent Han-character
  gap.
- train / validation / test pairs: 16,000 / 4,000 / 4,000 in each mode.
- each split is balanced equally across news, academic, encyclopedia, and
  dialogue sources.
- exact sample, reconstructed text, and gold character-boundary equality check:
  passed.
- non-Han content is excluded in both modes.

## Result

| Metric | Word gaps | Character gaps | Character minus word |
| --- | ---: | ---: | ---: |
| Best validation epoch | 3 | 2 | — |
| Average test candidate gaps | 12.77 | 20.17 | +7.40 |
| Test top-1 accuracy | **48.55%** | 46.28% | -2.27 pp |
| Test MRR | **64.70%** | 61.92% | -2.78 pp |
| Mean absolute boundary error | **3.16 chars** | 3.27 chars | +0.11 chars |
| Within ±1 character | **52.60%** | 51.53% | -1.07 pp |
| Within ±2 characters | **62.08%** | 61.18% | -0.90 pp |
| Random-choice baseline | 11.23% | 6.83% | -4.40 pp |
| Center baseline | 20.30% | 14.25% | -6.05 pp |

Test top-1 accuracy by domain:

| Domain | Word gaps | Character gaps | Word advantage |
| --- | ---: | ---: | ---: |
| Academic | 50.50% | 48.10% | +2.40 pp |
| Dialogue | 59.70% | 57.60% | +2.10 pp |
| Encyclopedia | 43.60% | 39.50% | +4.10 pp |
| News | 40.40% | 39.90% | +0.50 pp |

## Interpretation

At this scale and three-epoch budget, word-gap candidates are modestly better
on every recorded quality metric. They remove 36.7% of the character model's
candidate gaps, raise exact accuracy by 2.27 percentage points, and reduce the
mean boundary error by 0.11 characters. The difference is not large enough to
establish a general winner from one seed.

The fixed six-layer, kernel-3 CNN also has different effective textual context:
approximately 13 word tokens in word mode but only 13 Han characters in
character mode. The character result therefore measures both the larger search
space and the shorter character-level context of this unchanged architecture.

There are two additional caveats:

1. The word vocabulary reached its 4,096-token cap, while the character
   vocabulary contained 3,969 tokens, producing 252,068 versus 245,845 stored
   parameters. The convolutional trunk is identical; the difference is in the
   embedding table.
2. Word mode currently tokenizes A and B independently before concatenation.
   When the same test strings are concatenated first and then segmented, 26 of
   4,000 gold boundaries (0.65%) fall inside a resulting word token. This small
   train/runtime mismatch can favor word mode and should be removed in a
   definitive comparison.

Detailed generated metrics are stored under the ignored directory
`training/artifacts/tokenization-medium/`. Reproduce the run with:

```bash
npm run smoke:compare
```
