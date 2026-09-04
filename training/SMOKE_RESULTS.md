# Boundary model smoke result

Run date: 2026-09-04

The smoke test completed the full path from public source archives to a saved
model and recursive boundary tree. It is an integration check, not a production
quality claim.

## Reproducibility

- seed: `2026090432`
- pair-length policy: discard the pair if either side exceeds 32 content
  characters; never crop; no minimum side length
- architecture: 3 residual blocks, 2 convolutions per block, kernel 3,
  dilation 1, 48 channels
- model parameters: 206,988
- vocabulary size: 3,176
- maximum sequence length: 34 words
- train / validation / test pairs: 512 / 128 / 128
- each split contains equal samples from news, academic, encyclopedia, and
  dialogue sources
- split unit: source document, before adjacent-pair generation
- global exact-document deduplication: 18,315 duplicates removed
- document leakage check: passed
- generated-data audit: no side exceeded 32 characters and all target indices
  were valid; the training and validation data both retained one-character sides

## Result

| Metric | Value |
| --- | ---: |
| Best validation epoch | 7 |
| Validation top-1 accuracy | 32.03% |
| Validation MRR | 48.20% |
| Test top-1 accuracy | 18.75% |
| Test MRR | 39.78% |
| Random-choice test baseline | 11.27% |

Test top-1 accuracy by domain:

| Domain | Accuracy |
| --- | ---: |
| Academic | 21.88% |
| Dialogue | 18.75% |
| Encyclopedia | 15.62% |
| News | 18.75% |

Training loss reached approximately zero after only a few epochs while
validation loss increased. The 512-pair smoke set is therefore far too small
for a deployable model, but the test accuracy above the random-choice baseline
confirms that the complete variable-choice training path learns a signal.

Generated checkpoints and detailed metrics live in the ignored
`training/artifacts/` directory and can be recreated with:

```bash
npm run smoke:model
```
