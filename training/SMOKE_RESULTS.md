# Boundary model smoke result

Run date: 2026-09-04

The smoke test completed the full path from public source archives to a saved
model and recursive boundary tree. It is an integration check, not a production
quality claim.

## Reproducibility

- seed: `2026090405`
- pair-length policy: no minimum or maximum side length and no cropping
- training-position policy: ten round-robin buckets of
  `A_length / (A_length + B_length)` within each source domain
- validation and test retain their natural position distributions
- architecture: 3 residual blocks, 2 convolutions per block, kernel 3,
  dilation 1, 48 channels
- candidate positions: every adjacent Han-character gap
- non-Han policy: excluded before model input
- model parameters: 252,068
- vocabulary size: 4,096
- maximum sequence length: 365 Han characters
- train / validation / test pairs: 40,000 / 8,000 / 8,000
- each split contains equal samples from news, academic, encyclopedia, and
  dialogue sources
- split unit: source document, before adjacent-pair generation
- global exact-document deduplication: 18,315 duplicates removed
- document leakage check: passed
- generated-data audit: 2,479 training pairs contain a side longer than 32
  characters; the longest side is 459 characters
- dynamic batching: power-of-two length buckets, per-batch padding, at most 64
  examples and a target budget of 2,048 padded tokens
- training padding efficiency: 73.09%
- training epochs: 3

## Result

| Metric | Value |
| --- | ---: |
| Best validation epoch | 2 |
| Validation top-1 accuracy | 52.36% |
| Validation MRR | 66.98% |
| Test top-1 accuracy | 53.01% |
| Test MRR | 67.60% |
| Test mean absolute boundary error | 2.71 chars |
| Test within ±1 character | 57.78% |
| Test within ±2 characters | 67.50% |
| Random-choice test baseline | 6.83% |
| Always choose center | 14.84% |
| Train-set length lookup | 8.06% |

Test top-1 accuracy by domain:

| Domain | Accuracy |
| --- | ---: |
| Academic | 55.25% |
| Dialogue | 63.75% |
| Encyclopedia | 47.85% |
| News | 45.20% |

This run replaces word-gap candidates with every adjacent Han-character gap.
It used an average of 20.23 test candidates and reached 53.01% exact accuracy,
well above the 6.83% random, 14.84% center, and 8.06% length-lookup baselines.
Validation peaked at epoch 2 and declined slightly at epoch 3, so the exported
browser model uses the restored epoch-2 checkpoint. Dialogue remained the
strongest domain, while news and encyclopedia remained below 50%.

Generated checkpoints and detailed metrics live in the ignored
`training/artifacts/` directory and can be recreated with:

```bash
npm run smoke:model
```
