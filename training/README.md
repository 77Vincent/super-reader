# Boundary model smoke test

This experiment checks the complete weak-supervision pipeline without claiming
production model quality. Its default dataset contains 40,000 training pairs
and 8,000 pairs in each evaluation split, balanced equally across four domains.
All eligible source documents participate in the document-level split; `0` for
`--docs-per-domain` means that no document-count cap is applied. Training runs
for three epochs by default and retains the checkpoint with the best validation
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
pairs are generated only after that split, and every split contains an equal
number of samples from all four domains.

There is no minimum or maximum length for either side: even a one-character
side remains valid, and long sides are never cropped. Punctuation and
whitespace are excluded from the recorded side lengths.

Within each source domain, training samples are selected round-robin from ten
equal-width buckets of `A_length / (A_length + B_length)`. This suppresses the
shortcut of always choosing a boundary near the middle. Validation and test
retain their natural position distributions.

Training batches use power-of-two token-length buckets. Every batch is padded
only to its own longest sequence, with at most 64 examples and a target budget
of 2,048 padded tokens. Longer buckets automatically use fewer examples; a
single sequence longer than the budget is still retained in a one-example
batch. Side-length and relative-position metadata are kept for auditing but
are not copied into the model input.

Downloaded and generated files are ignored by Git. Source URLs and SHA-256
checksums are saved in `training/data/processed/summary.json`.

## Run

```bash
npm run smoke:model
```

The default production run uses character tokenization: 40,000 training pairs,
8,000 validation pairs, 8,000 test pairs, and three epochs. Every adjacent Han
character gap is a candidate; non-Han content is excluded from model input.

The command installs the pinned 751 KB `tinygrad` wheel under the ignored
`training/.deps/` directory. To prepare and train separately:

```bash
node training/prepare_smoke_data.mjs
python3 training/run_smoke.py
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
