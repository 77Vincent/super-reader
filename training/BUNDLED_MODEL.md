# Bundled boundary model

`src/boundary-model-data.js` contains the completed epoch 1 checkpoint from
`all-local-192ch-12conv-20260912`. It was selected from the training checkpoint's
`best_state` while epoch 2 was still running, and exported on 2026-09-13.

| Property | Value |
|---|---:|
| Channels | 192 |
| Residual blocks / convolution layers | 6 / 12 |
| Parameters | 2,265,991 |
| Vocabulary entries | 4,096 |
| Training examples per epoch | 74,218,072 |
| Validation accuracy | 83.60% |
| Mean domain validation accuracy | 80.78% |
| Validation selection score | 82.19% |
| Test accuracy | Not evaluated for this interim export |

The validation set contains 731,290 examples with enumeration commas excluded
as boundary proxies. Its records are identical to the corrected holdout used
for the previous model. The training overlap audit found no overlapping pairs.

Checkpoint SHA-256:
`626c7aa7599c9a556bfefd04b8ddd054cf019282c056caad197bebac56fc784c`

The frozen export artifacts are under
`training/artifacts/all-local-192ch-12conv-20260912/epoch-1-backend/`.
`smoke-metrics.json` records the source checkpoint hash, selection metric and
extracted state. `selected-state.pt` contains the completed epoch 1 weights;
it is not a resumable optimizer checkpoint.

Run `npm run model:export` to reproduce the bundled file from those local
artifacts. The generated script is approximately 12.13 MB and is loaded by the
existing inference worker and backend. Model architecture is read from its
metadata. `test/model-backend-reference.json` contains independent PyTorch
reference logits, including a two-character input and a supplementary Han
character, for checking the JavaScript implementation.

JavaScript and PyTorch agree on all five reference boundaries, with a maximum
logit difference of 0.00001526. On the fixed 2,500-example validation sample
(500 per domain), JavaScript accuracy is 80.88%, compared with 79.48% for the
previous 128-channel no-enumeration model. This sample has a different domain
mix from the full validation set above.

The full test suite passes all 174 tests, including a run from a fresh source
snapshot without local dependencies, datasets or training artifacts. The corpus
test installs its pinned PyArrow dependency when missing and generates tiny
Parquet fixtures; no tests are skipped. Chunker tests cover the finalized
12-visual-unit threshold and the bundled model's expected outputs. The backend
model-loading and PyTorch parity tests also pass.

Epoch 2 continues in the separate `candidate/` directory. Its completion does
not automatically replace this bundled epoch 1 model.
