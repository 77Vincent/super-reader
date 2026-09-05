# Boundary model result

These metrics describe the checkpoint currently exported to
`src/boundary-model-data.js`.

Run date: 2026-09-05

## Reproducibility

- seed: `2026090405`
- tokenization: individual Han characters
- candidate positions: every adjacent Han-character gap
- non-Han policy: excluded before model input
- architecture: 4 residual blocks, 2 kernel-3 convolutions per block,
  dilation 1, 128 channels
- initialization: all 64 overlapping channels and 4 residual blocks copied
  from the retained 75.50% checkpoint
- model parameters: 985,349
- vocabulary size: 4,096
- maximum sequence length: 1,986 Han characters
- train / validation / test pairs: 34,341,665 / 829,297 / 872,558
- training data: every block of the downloaded Chinese Wikipedia dump plus
  the retained non-Wikipedia CLUE data; each eligible Wikipedia article
  contributes at most 128 evenly distributed boundaries
- training data scale: 5.17 times the previous 6,638,889-pair run
- training data policy: balance relative boundary-position buckets with loss
  weights
- holdout policy: validation and test documents are excluded before sample
  generation; their boundaries seed a Bloom filter with no false negatives
- dynamic batching: power-of-two length buckets, at most 512 examples and
  8,192 padded tokens per batch
- memory policy: 128 compact disk shards, one training shard resident at a
  time, and preparation/training checkpoints for safe resume
- evaluation padding efficiency: 71.72% validation, 71.80% test
- CPU threads: 5 intra-op, 1 inter-op
- training epochs: 3
- pure batch-training time: 4 hours 31 minutes 11 seconds

## Result

| Metric | Value |
| --- | ---: |
| Best validation epoch | 3 |
| Validation top-1 accuracy | 80.30% |
| Validation MRR | 87.92% |
| Test top-1 accuracy | 80.23% |
| Test MRR | 87.87% |
| Test mean absolute boundary error | 1.29 chars |
| Test within ±1 character | 82.13% |
| Test within ±2 characters | 86.07% |
| Random-choice test baseline | 6.69% |
| Always choose center | 12.17% |

Test top-1 accuracy by domain:

| Domain | Accuracy |
| --- | ---: |
| Academic | 69.74% |
| Dialogue | 82.38% |
| Encyclopedia | 78.96% |
| News | 58.16% |
| Wikipedia | 80.42% |

Validation accuracy by epoch:

| Epoch | Accuracy |
| --- | ---: |
| 1 | 79.14% |
| 2 | 79.83% |
| 3 | 80.30% |

The generated training artifacts live in the ignored directory
`training/artifacts/wiki-full-128ch-8conv-3ep/`. Reproduce preparation,
training, and export with:

```bash
npm run full:data
npm run full:model
python3 training/export_browser_model.py \
  --artifact-dir training/artifacts/wiki-full-128ch-8conv-3ep
```
