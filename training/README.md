# Boundary model smoke test

This experiment checks the complete weak-supervision pipeline without claiming
production model quality.

## Architecture

- word embedding with 48 channels;
- three residual blocks;
- two `Conv1d(kernel_size=3, dilation=1)` layers per block;
- one score for every adjacent word gap;
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

For each adjacent pair, punctuation and whitespace are excluded from the
length count. If either `A` or `B` contains more than 32 content characters,
the whole pair is discarded rather than cropped. A side of exactly 32
characters is allowed, and there is no minimum length: even a one-character
side remains a valid training example.

Downloaded and generated files are ignored by Git. Source URLs and SHA-256
checksums are saved in `training/data/processed/summary.json`.

## Run

```bash
npm run smoke:model
```

The command installs the pinned 751 KB `tinygrad` wheel under the ignored
`training/.deps/` directory. To prepare and train separately:

```bash
node training/prepare_smoke_data.mjs
python3 training/run_smoke.py
```

The checkpoint and metrics are written under `training/artifacts/`.
The checked-in summary of the latest completed run is in
[`SMOKE_RESULTS.md`](SMOKE_RESULTS.md).
