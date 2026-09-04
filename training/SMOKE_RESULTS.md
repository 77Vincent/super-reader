# Boundary model smoke result

> These metrics describe the currently exported checkpoint, which was trained
> before the data pipeline switched from equal-domain downsampling to
> all-sample length-position weighting. Run `npm run smoke:model` to produce a
> directly comparable checkpoint under the new policy.

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
- train / validation / test pairs: 133,716 / 16,576 / 16,924
- maximum balanced samples per domain: 33,429 / 4,144 / 4,231
- each split contains equal samples from news, academic, encyclopedia, and
  dialogue sources
- split unit: source document, before adjacent-pair generation
- global exact-document deduplication: 18,315 duplicates removed
- document leakage check: passed
- generated-data audit: 4,008 training pairs contain a side longer than 32
  characters; the longest side is 353 characters
- dynamic batching: power-of-two length buckets, per-batch padding, at most 64
  examples and a target budget of 2,048 padded tokens
- training padding efficiency: 72.92%
- training epochs: 2

## Result

| Metric | Value |
| --- | ---: |
| Best validation epoch | 2 |
| Validation top-1 accuracy | 59.60% |
| Validation MRR | 73.09% |
| Test top-1 accuracy | 60.41% |
| Test MRR | 73.74% |
| Test mean absolute boundary error | 2.16 chars |
| Test within ±1 character | 64.81% |
| Test within ±2 characters | 73.67% |
| Random-choice test baseline | 6.85% |
| Always choose center | 14.31% |
| Train-set length lookup | 13.89% |

Test top-1 accuracy by domain:

| Domain | Accuracy |
| --- | ---: |
| Academic | 63.93% |
| Dialogue | 68.94% |
| Encyclopedia | 53.84% |
| News | 54.93% |

This run uses the maximum amount of data compatible with equal representation
of all four domains in each split. The training set grew from 40,000 to 133,716
pairs, and test exact accuracy rose from 53.01% to 60.41% in two epochs. The
model used an average of 20.18 test candidates and exceeded the 6.85% random,
14.31% center, and 13.89% length-lookup baselines. All four domains improved;
dialogue remained strongest at 68.94%.

Generated checkpoints and detailed metrics live in the ignored
`training/artifacts/` directory and can be recreated with:

```bash
npm run smoke:model
```
