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
- model parameters: 219,434
- vocabulary size: 3,430
- maximum sequence length: 120 words
- train / validation / test pairs: 512 / 128 / 128
- each split contains equal samples from news, academic, encyclopedia, and
  dialogue sources
- split unit: source document, before adjacent-pair generation
- global exact-document deduplication: 18,315 duplicates removed
- document leakage check: passed
- generated-data audit: 43 training pairs contain a side longer than 32
  characters; the longest side is 187 characters
- dynamic batching: power-of-two length buckets, per-batch padding, at most 64
  examples and a target budget of 2,048 padded tokens
- training padding efficiency: 74.51% instead of 12.18% under global
  120-token padding

## Result

| Metric | Value |
| --- | ---: |
| Best validation epoch | 5 |
| Validation top-1 accuracy | 23.44% |
| Validation MRR | 42.09% |
| Test top-1 accuracy | 26.56% |
| Test MRR | 45.40% |
| Random-choice test baseline | 12.34% |
| Always choose center | 13.28% |
| Train-set length lookup | 14.84% |

Test top-1 accuracy by domain:

| Domain | Accuracy |
| --- | ---: |
| Academic | 37.50% |
| Dialogue | 25.00% |
| Encyclopedia | 21.88% |
| News | 21.88% |

Training loss reached approximately zero after only a few epochs while
validation loss increased. The 512-pair smoke set is therefore far too small
for a deployable model. In this run the CNN exceeded both the center and
train-length lookup baselines, so its result is not explained by those two
length shortcuts alone; more data and repeated seeds are still required.

Generated checkpoints and detailed metrics live in the ignored
`training/artifacts/` directory and can be recreated with:

```bash
npm run smoke:model
```
