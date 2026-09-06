# Boundary model result

These metrics describe the checkpoint currently exported to
`src/boundary-model-data.js`.

Run date: 2026-09-06

## Reproducibility

- seed: `2026090405`
- tokenization: individual Han characters
- candidate positions: every adjacent Han-character gap
- non-Han policy: excluded before model input
- architecture: 4 residual blocks, 2 kernel-3 convolutions per block,
  dilation 1, 128 channels
- initialization: all 128 channels and 4 residual blocks copied from the
  retained 80.23% full-Wikipedia checkpoint
- model parameters: 985,349
- vocabulary size: 4,096
- maximum sequence length: 1,986 Han characters
- train / validation / test pairs: 39,341,665 / 829,297 / 872,558
- added training data: 5,000,000 globally deduplicated boundaries from the
  Chinese multi-style synthetic subset of `openbmb/Ultra-FineWeb-L3`
- original training data: 34,341,665 Wikipedia and retained CLUE boundaries
- weighting: existing length-position inverse-cell weights multiplied by
  smoothed inverse-domain-frequency weights with exponent `0.65`
- checkpoint selection: equal parts overall validation accuracy and macro
  validation accuracy across the five real holdout domains
- holdout policy: validation and test remain the original real-data holdouts;
  their exact boundaries and all old training boundaries are excluded from the
  synthetic expansion
- dynamic batching: power-of-two length buckets, at most 512 examples and
  8,192 padded tokens per batch
- memory policy: 160 compact disk shards, one training shard resident at a
  time, and atomic preparation/training checkpoints for safe resume
- evaluation padding efficiency: 71.72% validation, 71.80% test
- CPU threads: 5 intra-op, 1 inter-op
- training epochs: 2
- pure batch-training time: 3 hours 31 minutes 39 seconds

## Result

| Metric | Previous | Current | Change |
| --- | ---: | ---: | ---: |
| Best validation epoch | 3 | 2 | — |
| Validation top-1 accuracy | 80.30% | 81.99% | +1.70 pp |
| Validation macro-domain accuracy | 73.40% | 78.12% | +4.72 pp |
| Test top-1 accuracy | 80.23% | 81.96% | +1.72 pp |
| Test macro-domain accuracy | 73.93% | 78.62% | +4.69 pp |
| Test MRR | 87.87% | 89.03% | +1.15 pp |
| Test mean absolute boundary error | 1.29 chars | 1.20 chars | -0.09 chars |
| Test within ±1 character | 82.13% | 83.64% | +1.51 pp |
| Test within ±2 characters | 86.07% | 87.23% | +1.16 pp |

Test top-1 accuracy by real holdout domain:

| Domain | Previous | Current | Change |
| --- | ---: | ---: | ---: |
| Academic | 69.74% | 77.72% | +7.98 pp |
| Dialogue | 82.38% | 87.34% | +4.96 pp |
| Encyclopedia | 78.96% | 80.79% | +1.83 pp |
| News | 58.16% | 65.22% | +7.06 pp |
| Wikipedia | 80.42% | 82.03% | +1.61 pp |

Validation accuracy and macro-domain accuracy by epoch:

| Epoch | Accuracy | Macro-domain accuracy | Selection score |
| --- | ---: | ---: | ---: |
| 1 | 81.90% | 77.43% | 79.66% |
| 2 | 81.99% | 78.12% | 80.06% |

The generated artifacts live in the ignored directory
`training/artifacts/wiki-ultra-domain-weighted-128ch-2ep/`. Reproduce data
preparation, training, and browser export with:

```bash
npm run synthetic:data
npm run synthetic:model
npm run model:export
```
