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
- model parameters: 252,068
- vocabulary size: 4,096
- maximum sequence length: 301 words
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
- training padding efficiency: 73.29% instead of 4.67% under global
  301-token padding
- training epochs: 3

## Result

| Metric | Value |
| --- | ---: |
| Best validation epoch | 3 |
| Validation top-1 accuracy | 54.98% |
| Validation MRR | 70.11% |
| Test top-1 accuracy | 55.88% |
| Test MRR | 70.62% |
| Random-choice test baseline | 11.37% |
| Always choose center | 19.99% |
| Train-set length lookup | 14.55% |

Test top-1 accuracy by domain:

| Domain | Accuracy |
| --- | ---: |
| Academic | 58.70% |
| Dialogue | 66.60% |
| Encyclopedia | 48.80% |
| News | 49.40% |

Increasing the training set from 6,144 to 40,000 pairs raised test accuracy
from 45.57% to 55.88%, but did not reach the 70% target within three epochs.
The CNN exceeded both the center and train-length lookup baselines, so its
result is not explained by those two length shortcuts alone. Dialogue reached
66.60%, while news and encyclopedia remained below 50%, making those domains
the main limit on aggregate accuracy.

Generated checkpoints and detailed metrics live in the ignored
`training/artifacts/` directory and can be recreated with:

```bash
npm run smoke:model
```
