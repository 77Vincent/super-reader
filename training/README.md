# Boundary model smoke test

This experiment checks the complete weak-supervision pipeline without claiming
production model quality. Its default dataset uses every eligible pair in each
split without equalizing or downsampling source domains. All eligible source
documents participate in the document-level split; `0` for
`--docs-per-domain` means that no document-count cap is applied. Training runs
for two epochs by default and retains the checkpoint with the best validation
accuracy.

## Architecture

- Han-character embedding with 48 channels;
- three residual blocks;
- two `Conv1d(kernel_size=3, dilation=1)` layers per block;
- one score for every adjacent Han-character gap;
- masked softmax cross-entropy with exactly one target gap.

## Data

The preparation script downloads four small public CLUE task archives:

- TNEWS for news text;
- CSL for academic abstracts;
- CMRC2018 for encyclopedia passages;
- C3 for dialogue.

Documents are normalized and globally deduplicated before being split. Each
document belongs to exactly one of train, validation, or test. Adjacent `A+B`
pairs are generated only after that split. Every eligible pair is retained, so
the four source domains keep their natural sample counts.

There is no minimum or maximum length for either side: even a one-character
side remains valid, and long sides are never cropped. Punctuation and
whitespace are excluded from the recorded side lengths.

Training samples are grouped by total Han-character length (`2–8`, `9–16`,
`17–32`, and `33+`) and then by ten equal-width buckets of
`A_length / (A_length + B_length)`. No samples are discarded: inverse-cell loss
weights make the occupied position buckets contribute equally within each
length bucket. Domains are not part of the weighting calculation. Validation
and test retain their natural distributions and use ordinary unit weights.

Training batches use power-of-two token-length buckets. Every batch is padded
only to its own longest sequence, with at most 512 examples and a target budget
of 8,192 padded tokens. That produces 512-example batches for the common
16-character bucket, 256 for the 32-character bucket, and 128 for the
64-character bucket. Longer buckets automatically use fewer examples; a single
sequence longer than the budget is still retained in a one-example batch.
Side-length and relative-position metadata are kept for auditing but are not
copied into the model input.

Training uses PyTorch's multithreaded CPU backend. The default is five intra-op
threads and one inter-op thread: a local backward-pass benchmark on the M5 Pro
showed that this is faster for these short convolutions than scheduling work
across all 18 logical CPUs. Each kernel-3 convolution is evaluated as three
mathematically equivalent left/center/right matrix products, avoiding the high
overhead of the generic macOS `Conv1d` kernel on short sequences. Thread counts
remain explicitly configurable with `--threads` and `--interop-threads`. Each
epoch reports wall-clock training throughput, and the selected CPU settings
are saved in the metrics artifact.

Downloaded and generated files are ignored by Git. Source URLs and SHA-256
checksums are saved in `training/data/processed/summary.json`.

## Run

```bash
npm run smoke:model
```

The default production run uses character tokenization, every available pair,
and two epochs. Passing `0` for any `--train-per-domain`,
`--validation-per-domain`, or `--test-per-domain` keeps every eligible sample
from each domain. Positive values remain available for explicitly capped
comparison experiments. Every adjacent Han-character gap is a candidate;
non-Han content is excluded from model input.

The command installs pinned PyTorch and tinygrad wheels under the ignored
`training/.deps/` directory. PyTorch performs multithreaded CPU training;
tinygrad is retained only for writing the existing browser-compatible
safetensors checkpoint. To prepare and train separately:

```bash
node training/prepare_smoke_data.mjs
python3 training/run_smoke.py
```

For example, an explicit CPU configuration can be tested with:

```bash
python3 training/run_smoke.py --threads 5 --interop-threads 1 \
  --batch-size 512 --max-tokens-per-batch 8192
```

The checkpoint and metrics are written under `training/artifacts/`.
The checked-in summary of the latest completed run is in
[`SMOKE_RESULTS.md`](SMOKE_RESULTS.md).

## Candidate-tokenization comparison

To compare `Intl.Segmenter` word gaps with every adjacent Han-character gap on
the same medium-size sample set, run:

```bash
npm run smoke:compare
```

Both modes exclude non-Han content. The controlled comparison uses 16,000
training pairs, 4,000 validation pairs, 4,000 test pairs, and three epochs per
mode. Its latest result and methodological caveats are documented in
[`TOKENIZATION_COMPARISON.md`](TOKENIZATION_COMPARISON.md).
