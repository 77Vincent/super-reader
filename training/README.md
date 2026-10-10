# Boundary model training

The backend now bundles **epoch 1** of the fresh run
`fresh-lr3e-4-max256-20261005`, promoted on 2026-10-07. It trained from random
weights on **395,216,214** examples at learning rate **0.0003**, with no position
or corpus-source weighting. Validation accuracy is **90.6079%** and test accuracy
is **90.6847%**, over the complete retained splits of 2,515,459 and 2,563,530 examples.
All splits exclude one-character target sides and inputs over 256 characters.
The prepared v7 corpus is unchanged. See [BUNDLED_MODEL.md](BUNDLED_MODEL.md) for
export hashes, numerical parity checks and the exact evaluation scope.
`npm run model:export` uses this run's frozen `epoch-1-backend/` directory.

The previous [padding-corrected continuation](PADDING_FIXED_EXPANSION.md) is a
completed historical experiment. Its selected epoch 1 scored 89.4642% validation
and 89.5785% test; epoch 2 scored 89.4241% validation. Those results used unfiltered
holdouts and are not directly comparable to the new retained-split scores.
The old continuation commands below describe that historical run.

The completed [source-mixture pilot](SOURCE_MIX_COMPARISON.md) compared 10 million
matched training pairs per arm at learning rate 0.00003. Complete validation
accuracy was 89.1853% for the current mixture and 89.1624% for Wikipedia plus
synthetic, versus the inherited model's 89.2357%. Both retained epoch 0 under the
selection rule; their recorded test results reuse that initial checkpoint and
do not measure the new endpoints. That pilot did not replace the backend. The recorded
run can be inspected with `python3 training/run_source_comparison.py --resume`.

The current data contract is **unicode-context-v8**, defined in
[text-policy.json](text-policy.json). Its model input representation remains
**unicode-context-v1**. V8 adds literal whitespace/control escape rejection to
v7's existing filters. In addition to requiring Han in each fragment, it rejects
structural symbols, single invisible controls, every physical line's first
fragment, and an unterminated last fragment. Bounded symbol windows, explicit
source-deletion barriers, signed-number protection and neighbor rules remain. It does not change the CNN architecture
or token IDs. Training, validation and test preparation share
this contract. Historical corpora cannot be silently mixed into a new run.
The existing large prepared corpus is still **v7**; reusing it does not apply
v8's new raw-document filters retroactively. The [cleanup record](DISK_CLEANUP.md)
covers the 2026-09-30 data cleanup and the 2026-10-04 removal of stale process
records and a rebuildable audit environment. Current data, historical
deduplication dependencies, model checkpoints and experiment reports are retained.

## Input and labels

Adjacent fragments A and B become the input AB. Only their separating proxy
punctuation is hidden; the target is the code-point gap between A and B.
`proxy_punctuation` is the single whitelist: `， 。 ； ！ ？ …`.
These exact characters are recognized in the original source, before NFKC.
Other punctuation remains context and does not produce a label; no inverse
exclusion list is maintained. Thus ASCII punctuation and compatibility forms
such as `﹐` or `｡` do not become proxies merely because normalization changes
their appearance.

The browser now follows this same raw-proxy, numeric-comma and normalization
contract, including physical line barriers. Inference receives an already
delimited clause; training constructs AB by hiding the proxy between two valid
fragments. That intentional difference remains. Corpus eligibility filters
select teachers and are not used to delete webpage content. See
[the input-alignment fixes and verification](INPUT_ALIGNMENT.md).

Other punctuation, digits, letters and quotes remain in eligible input;
fragments with `{ } < > & =` or their NFKC equivalents are discarded whole.
Actual line breaks are split before whitespace normalization: no pair crosses a
source line, and a newline itself never supplies a target. Every line's first
nonempty fragment is ineligible, including the first line of a document.
The last fragment is also ineligible unless terminated by a recognized proxy
(ASCII punctuation does not qualify). Consecutive proxies form one delimiter.
These are independent defaults under `sample_filter.line_edges`: a line with only
`A，B。` yields no samples, while `A，B，C。` can yield `B｜C`. Leading empty
punctuation does not rescue the first real fragment. After proxy detection,
retained fragments are normalized with NFKC. Whitespace within a line
becomes a single ASCII space and fragment edges are trimmed. A comma proxy
between decimal numbers stays intact as numeric context, including signed values,
decimal fractions, exponents and surrounding whitespace. This protection retains
the separator inside the fragment: `(-8545，-27679)` becomes `(-8545,-27679)`
after NFKC and creates no internal target. Both complete fragments must each contain at least one Han character
(`sample_filter.require_han`, enabled by default). A fragment without Han
is rejected together with either touching pair, without reconnecting its neighbors.
For example, `正文。3。后文` yields neither `正文｜3`, `3｜后文`, nor `正文｜后文`. At least one of the two characters immediately adjoining the target
must be Han; if both are non-Han, discard that pair. Check normalized fragment
edges after trimming whitespace, without skipping quotes or other symbols.
Thus `使用API｜进行调用` remains eligible, whereas `中文｜API` and
`API｜中文` fail the whole-fragment rule. `中文API｜HTTP中文` has Han in
both fragments but still fails the immediate-neighbor rule.
The immediate-neighbor rejection is target-specific: it does not discard either fragment from
other valid adjacent pairs, and does not reconnect across the rejected boundary.
Source filtering runs through the shared policy in every preparation route.
Wikipedia extraction also preserves explicit visible link labels; discarded
templates, references, markup content and external links leave barriers. Every
original newline inside a removed span survives. A fragment containing a barrier
is ineligible on either side of any target; no surviving pieces are rejoined.
Unclosed templates or code fences are discarded through their remaining extent.

For each proxy, find the nearest non-proxy punctuation/symbol on **both** sides
within the same physical line, without crossing a deletion barrier. When the sum
of code-point distances is **at most 18**, reject that target. Spaces count before
normalization; the symbols need not match. Unicode punctuation/symbol categories
are frozen in [unicode-symbols.json](unicode-symbols.json), shared and hash-checked
by Python and JavaScript. A one-sided symbol does not trigger this rule.
A rejected proxy still delimits fragments: in `引句。前句，甲(，乙)，后句。`,
`前句｜甲(` and `乙)｜后句` remain eligible; `甲(｜乙)` is rejected, and
no larger fragment is formed by joining across it. If any mark in a consecutive
proxy run is rejected, the whole target is rejected. Numeric separators differ:
they are retained *inside* fragments as described above.

The [v6 window review](HAN_FRAGMENT_WINDOW_REVIEW.md) evaluates the window after
the Han-fragment rule; the [v5 study](SYMBOL_WINDOW_STUDY.md) is historical. Normal quoted/list text is also lost;
this is a conservative rule, not a guarantee of clean semantics.

Discard pairs touching explicit placeholders (`$P$`, `(I_M_G)`), fill-in blanks,
empty templates, Markdown heading markers, even single invisible controls, or a
whole fragment that is a known web control such as `更多`. HTML/Wiki/Markdown
residue, literal whitespace/control escapes, replacement characters and URL fragments are also
rejected; matched code/format spans are masked with barriers instead of repaired. Rejected fragments
remain barriers: A / rejected / B does not become A+B. Ordinary extra spaces,
code identifiers such as `__init__`, and semantic topic changes are retained
unless another rule rejects the pair. Parentheses and quotes remain in eligible
inputs, but the symbol-window rule may reject targets between them. There is no new score threshold or semantic prose filter;
these rules reduce visible formatting noise but do not certify meaning or quality.
The structural-symbol gate intentionally also loses legitimate formulas and
names containing `&`. Invisible controls are checked before whitespace folding,
including U+FEFF; zero-width joiner emoji are also rejected by this conservative
policy. Ordinary spaces and single non-joiner emoji remain eligible.

V8 (`surface-noise-v5`) rejects these literal escapes **after normal source
parsing**, when backslashes and letters/digits still remain in the text:

| Kind | Literal spellings |
| --- | --- |
| Spaces | `\u0020`, `\u00A0`, `\u1680`, `\u2000`–`\u200A`, `\u202F`, `\u205F`, `\u3000` |
| Zero-width characters and direction marks | `\u200B`–`\u200F`, `\u2060`, `\uFEFF` |
| Line/paragraph separators | `\u0085`, `\u2028`, `\u2029` |
| Basic controls | `\u0000`–`\u001F`, `\u007F` |
| Short escapes | `\r`, `\n`, `\t`, `\b`, `\f`, `\v`, `\0` |

The Unicode list contains exactly 60 code points; hexadecimal letters are
case-insensitive. All spellings accept one or more consecutive backslashes.
This does not decode or remove an escape, and does not turn a literal newline
into a physical line break. The complete affected fragment remains a barrier;
for `引句。甲，\u3000乙，丙，丁。`, only `丙｜丁` survives. Retained labels,
fragment ordinals and normalized code-point indices keep their existing meaning.
Ordinary actual spaces, including U+3000, still normalize as before. Other
Unicode escapes such as `\u4E00` are not covered by this rule. This is a
conservative training-input filter, so deliberate textual explanations containing
these listed escapes can also be excluded; a match is not proof of a bad target.

Rejection counters keep `literal_line_escape` for `\r`/`\n` and add
`literal_control_escape` and `literal_unicode_residue`. Counts describe rejected
candidate pairs (including overlaps with other reasons), not unique documents
or wrong-label counts. Both Python and JavaScript preparation use this shared
policy. Fresh v8 datasets must be regenerated from source into a new directory;
do not relabel an old v7 manifest or edit characters inside old training rows.
Policy validators reject v7 and missing/altered escape rules. Existing corpora,
frozen historical experiments and shipped weights are not migrated automatically.

No Japanese-script blacklist is added: Japanese/mixed-language fragments may
remain if they satisfy the existing Han and boundary rules. Kana presence in
an audit is a script indicator, not a language classifier.

The [v6 expanded audit](V6_EXPANDED_QUALITY_AUDIT.md) supplied the new counterexamples;
the [v7 paired loss study](CONSERVATIVE_CLEANING_V7.md) records actual per-source
losses, overlap between gates, and the disproportionate cost to short headlines.

```text
Source: 引句。女：那可挺麻烦的，吃点儿治疗过敏的药吧。
Input:  女:那可挺麻烦的吃点儿治疗过敏的药吧
Target: 女:那可挺麻烦的 | 吃点儿治疗过敏的药吧

Source: 引句。上午8:30至下午4:30；假日关门。
Input:  上午8:30至下午4:30假日关门
Target: 上午8:30至下午4:30 | 假日关门

Source: 引句。F. Billinghurst负责设计，团队实施。
Input:  F. Billinghurst负责设计团队实施
Target: F. Billinghurst负责设计 | 团队实施

Source: 引句。坐标为(a,b)，继续计算。
Input:  坐标为(a,b)继续计算
Target: 坐标为(a,b) | 继续计算
```

The labels are punctuation-derived weak supervision. Accuracy measures recovery
of the hidden proxy; it does not establish human-rated reading chunk quality.

## Training metrics, position weights, and source tracing

The 2026-10-04 full fresh run remains paused with a saved checkpoint after the
initial-learning-rate comparison. The comparison does not automatically resume it.

```sh
npm run model:compare-initial-learning-rate
```

This compares fixed rates 0.0001, 0.0003, 0.001 and 0.002 from a common random
initialization, with fresh AdamW in each arm. It samples 1,000,000 training rows
uniformly across the full existing corpus and 100,000 validation rows, preserving
the v7 preparation policy. Both samples exclude single-character sides and have
no additional length cap. No source quotas or position/source weights are used.
All arms share data, batch order, initialization and the update budget. After one
subset epoch, the two highest endpoint validation accuracies continue to three
subset epochs with their own optimizer states; ties favor the smaller rate.
The test set is neither read nor scored. The result describes one initialization
seed and repeated exposure to the sampled subset, not a guarantee of ranking on
the full corpus or under learning-rate decay. The command reuses its frozen plan
and resumes incomplete arms. Results and row provenance are retained under
`artifacts/initial-learning-rate-1m-20261004/`.

The completed comparison favored **0.0003** under this short-run budget:

| Learning rate | Validation accuracy, epoch 1 | Validation accuracy, epoch 3 |
| --- | ---: | ---: |
| 0.0001 | 73.127% | Not extended |
| 0.0003 | 74.874% | 78.191% |
| 0.001 | 73.799% | 77.359% |
| 0.002 | 72.005% | Not extended |

Independent prediction replay matched both extended endpoints exactly. Their
0.832 percentage-point difference had a paired document-bootstrap 95% interval
of [0.618, 1.053] percentage points (34,625 validation documents; 2,000 draws).
This interval describes validation-document uncertainty for these fixed models,
not variability across training seeds. The slower first-round candidates were
not extended, so a later reversal is not ruled out. See the local
[experiment report](artifacts/initial-learning-rate-1m-20261004/report.md).

```sh
PYTHONPATH=training/.deps:training DEBUG=0 \
  python3 training/verify_initial_learning_rate_results.py \
  --run-dir training/artifacts/initial-learning-rate-1m-20261004
```

The new full-corpus run, started on 2026-10-05, uses the learning-rate and
length-cap smoke results above and below:

```sh
python3 training/run_fresh_training.py \
  --run-dir training/artifacts/fresh-lr3e-4-max256-20261005 \
  --epochs 1 --learning-rate 0.0003 --max-sequence-length 256 \
  --seed 2026100405 --defer-test
```

It starts from random weights and a new optimizer, excludes targets with a
one-character side, and applies no position or corpus-source weighting. The
existing v7 corpus stays unchanged; filters apply at read time. After one full
epoch it evaluates validation and stops for review, without reading test records
or exporting a model. The report is `candidate/validation-only.json`.
That review completed on 2026-10-06. After epoch 1 was selected for backend use,
a separate `epoch-1-backend/` directory was finalized on 2026-10-07 for full
validation/test evaluation and export; no additional training steps were taken.
`plan.json` pins the configuration and trainer source snapshot; `training.log`
records progress. To continue this run after interruption, use the same
`--run-dir` with `--resume`; all settings come from its frozen plan.

The separate, paused full 192-channel, 16-convolution run was originally started with:

```sh
python3 training/run_fresh_training.py \
  --run-dir training/artifacts/fresh-no-single-no-position-20261004
```

This starts from random weights and a new AdamW optimizer, with three epochs,
learning rate 0.002, no position or corpus-source weighting, and gradient clipping
at 1. Checkpoints are selected solely by overall validation accuracy. No old
checkpoint is used and no model is automatically
promoted. `--resume` reuses the frozen plan and the new run's own checkpoint.
The coordinator keeps macOS awake on AC power and resumes workers after a saved
MPS memory-pressure checkpoint. SIGTERM to the coordinator requests a safe stop.

### New-corpus continuation and intermediate validation

`prepare_training_snapshot.py` can freeze a randomized selection of completed
`prepare_remaining_web_data.py` chunks while preparation continues. It references
the sealed gzip shards in place, verifies their SHA-256 hashes, and reuses the
byte-identical evaluation files and vocabulary. It never includes `.chunk-writing`
or changes the preparation directory. `--target-samples` is an **approximate
retained-sample budget**: complete random chunks estimate the retention rate, then
whole shards are selected. `data/selection.json` records both raw counts and the
estimate; the trainer records exact retained counts as it reads the selected shards.

```sh
python3 training/prepare_training_snapshot.py \
  --source-dir training/data/processed/ultra-fineweb-remaining-v7-20261009 \
  --output-dir training/artifacts/NEW_RUN/data --target-samples 400000000
python3 training/run_fresh_training.py \
  --manifest training/artifacts/NEW_RUN/data/manifest.json \
  --run-dir training/artifacts/NEW_RUN \
  --continue-from training/artifacts/fresh-lr3e-4-max256-20261005-epoch2/candidate/training-state.pt \
  --epochs 3 --learning-rate 0.0003 --max-sequence-length 256 \
  --validation-every-samples 50000000 --defer-test
```

`--continue-from` requires a completed epoch, matching vocabulary, optimizer
configuration and evaluation identity. It carries the last model, AdamW state,
and epoch number into an explicitly new dataset. Ordinary `--resume` continues
the resulting run and still rejects changes to its manifest or configuration.
The parent checkpoint and its best validation model are preserved.

`--validation-every-samples` defaults to zero (epoch-end evaluation only).
When enabled, full validation runs at the first shard boundary after each
multiple of that many **retained training samples**. Validation restores training
mode and does not update the optimizer. Its time is excluded from training-speed
statistics. An interrupted validation retries at the saved boundary without
replaying optimizer steps. `candidate/validation-history.json` records results;
`candidate/validation-checkpoints/` retains resumable intermediate checkpoints.
`best_position` identifies the best epoch and sample position, including a
mid-epoch winner. No automatic early stopping, learning-rate change, test-set
evaluation (with `--defer-test`), or backend promotion occurs.

`train_sharded.py --min-side-characters 2` filters targets with a one-character
side from **training, validation and test** at read time. The default is 1 (no
exclusion). Lengths count Unicode code points, matching model tokenization.
It does not mask prediction candidates or modify original data. The default
unweighted path does not prescan training shards. Counts, exclusions and baselines
are accumulated when each shard is loaded and saved in the checkpoint and
`candidate/source-statistics.json`. Each file has one inventory entry: resuming
mid-shard or starting another epoch replaces that entry rather than counting it
again. `loaded_shards` and `complete` describe coverage of loaded file contents,
not how many batches have been optimized. Until every shard has been read, the
counts are partial. Validation/test counts describe retained rows before any
evaluation limit; `evaluated_samples` and the metric breakdowns describe what was
actually evaluated. Checkpoints record the filter and reject an incompatible resume.

`train_sharded.py --max-sequence-length 256` discards entire samples whose
normalized A+B input exceeds 256 Unicode code points, in training, validation
and test. It never truncates text or moves targets. The default is 0 (no cap),
and `run_fresh_training.py` exposes the same option. To preserve reproducible
sample order, training plans the original batches first, then drops overlength
records and empty batches before allocating batch tensors or updating weights.
Excluded records may remain in host memory for batch planning only. Filtering
reduces optimizer updates; it does not duplicate retained rows to replace them.
The cap is recorded in metrics and checkpoints; changing it on resume is rejected.
Old checkpoints without a recorded cap are interpreted as uncapped.

```sh
npm run model:compare-length
```

The length ablation reuses the verified three-epoch 0.0003 control above and
trains a capped arm from the identical random initialization, with the same
retained batches in the same order. Its main comparison evaluates both models
on the same validation rows within the cap. Full validation and overlength rows
are reported separately, so removing difficult validation rows is not counted
as a training gain. It never reads or scores the test set. Frozen inputs, code,
counts, batch schedule and paired predictions are retained under
`artifacts/length-cap-256-1m-20261004/`. It leaves the full run paused and does not
change the default cap or promote a model.

The completed 256-character smoke removed 441 of 1,000,000 training rows
(0.0441%). On the same 99,964 validation rows within the limit, accuracy rose
from 78.2032% to 78.5013%: 298 additional correct predictions, or +0.2981
percentage points. The paired document-bootstrap 95% interval was
[+0.1295, +0.4659] percentage points. On all original 100,000 validation rows,
accuracy rose from 78.191% to 78.488%; the 36 overlength rows changed from 16
to 15 correct, too few to establish a reliable long-input trend. Both models
saw three passes, with 16,113 versus 15,909 optimizer updates because the long
batches were omitted. This supports trying the cap but does not establish its
optimality or full-corpus benefit. See the local
[length ablation report](artifacts/length-cap-256-1m-20261004/report.md).

Legacy `domain` / `domains` fields in prepared data are read only as corpus IDs.
New trainer reports use `corpus_source`, `per_source` and `per_source_accuracy`:

| Legacy ID | Corpus source |
| --- | --- |
| `news` | CLUE TNEWS |
| `academic` | CLUE CSL |
| `encyclopedia` | CLUE CMRC2018 |
| `dialogue` | CLUE C3 |
| `wikipedia` | Chinese Wikipedia |
| `synthetic_multistyle` | openbmb/Ultra-FineWeb-L3, zh Multi-Style-Synthetic |
| `web` | openbmb/Ultra-FineWeb, zh |

These names describe dataset provenance, not topics inferred from the text.
Unknown legacy IDs are explicitly reported as `Unknown corpus source`.
Source counts and accuracies are diagnostic only: they never change loss,
sampling or checkpoint selection. Source-specific metrics include sample counts.
No raw data or historical metrics are rewritten. Existing preparation manifests
retain their legacy field names for serialization compatibility.

The fresh-run coordinator freezes current trainer code alongside the prepared
corpus's **recorded** text policy. It only permits surface-policy differences;
input representation and label semantics must match. Existing v7 shards remain
v7: newer raw-document cleaning rules are not retroactively claimed or applied.
Changing that preparation policy requires a separate corpus rebuild.

The sharded trainer reports training loss as `sum(weight * example_loss) / sum(weight)`.
This reporting reduction is separate from the fixed-global-normalization loss used
for optimization. MRR breaks equal scores by ascending gap index, matching the
first maximum selected by `argmax`. Historical reports retain their old values.
New checkpoints record the loss-reporting version, position-weight settings,
`source_weighting: none`, and `selection_metric: overall_validation_accuracy`;
old checkpoints with incompatible reporting state must use their original frozen
trainer to resume, or initialize a new run from their weights.

`--position-weighting none` is now the sharded trainer default. Explicit
`--position-weighting inverse-cell` remains available for controlled experiments;
`--weighting-manifest` can provide reference position frequencies. Only that
nonuniform position-weight path needs a scan to normalize its mean on retained
training rows. Length-based batching is unchanged. The old
`--domain-weight-power` and `--selection-macro-weight` options are removed from
the current trainer, rather than silently ignored.

Historical source/width/depth/learning-rate/rebuild coordinators encode their
original weighted experiment contracts. Use their original frozen sources to
reproduce those runs; use `run_fresh_training.py` for the current unweighted
large-corpus run. Do not resume an old weighted checkpoint with the new trainer
or compare old macro selection scores with new overall accuracy scores.
The obsolete `synthetic:model` npm entry was removed on 2026-10-04 because it
passed the deleted weighting options directly to the current trainer. Its
historical experiment settings and results remain archived.

```sh
npm run model:compare-position
```

New paired position smokes use the current sharded trainer, the same 1,000,000 training
rows and a seeded 100,000-row validation sample, starting from the current
padding-corrected epoch-1 weights. Both arms use fresh AdamW, learning rate
0.00003, one epoch, identical batch order, and no source weights. Only
position weighting changes. Samples are capped at 256 characters for the smoke;
training allocation follows the five existing source blocks, with random whole
shards and reservoir sampling within each block. This is a short continuation on
clustered samples and reused validation, not a from-scratch or untouched-test result.
The comparison reports both actual epoch endpoints even if validation selects
the initial model. Paired document-bootstrap intervals and length/position/source
breakdowns accompany the aggregate scores. Training loss across the two weighting
schemes is not a like-for-like comparison; use their unweighted validation metrics.

The run is isolated under `artifacts/position-ablation-1m-20261004/`, including
`plan.json`, frozen sources, sampled rows with source-shard/line references,
checkpoints, logs, and `comparison.json`. Rerun the same command to resume; use a
fresh `--run-dir` to create a new experiment with the current unweighted-source
settings. Existing runs keep their recorded source weights and frozen trainer.
No test scoring or model promotion
is performed. The source and data hashes are checked before resuming.

The historical [2026-10-04 result](position-weight-comparison-20261004.json),
which kept source weights fixed in both arms, starts at
89.512% validation accuracy. After one identical 1M-row pass, inverse-cell weights
give 89.468% and no position weights give 90.084% (+0.616 percentage points;
paired document-bootstrap 95% interval +0.524 to +0.708). These are actual epoch
endpoints: the weighted arm's selected best remains its initial checkpoint.
Unweighted validation NLL is 0.315516 versus 0.297311. The improvement has a
tradeoff: gold boundaries in the first 10% of the input fall from 89.280% to
80.189% (1,166 samples), while the 50%-60% bin improves from 89.972% to 91.690%
(23,225 samples). This supports further weighting experiments, not a claim that
all positions benefit or that full test has passed 90%. Small-source results
have low sample counts. The shipped model remains unchanged; this result predates
the current trainer's removal of source weighting.

```sh
npm run model:position-baselines
```

This evaluation reuses the same 100,000 validation rows and saved endpoint
predictions. It fits a text-blind exact-length/gap frequency prior on the same
1,000,000 training rows, using raw counts without domain or position weights.
Validation labels never fit the prior. The center baseline uses the existing
`(length - 1) // 2` zero-based gap convention (right of the midpoint for odd
character counts); the learned prior falls back to it for unseen lengths.
Frozen data/shard hashes, prediction target order, endpoint accuracy and position
breakdowns are checked. The vectorized baseline counts also match the existing
scalar evaluators. No retraining, test scoring, or model promotion is performed.

The [2026-10-04 baseline result](position-baselines-20261004.json) is:

| Predictor | Exact-gap validation accuracy |
| --- | ---: |
| Fixed center | 11.476% |
| Training-fitted length prior | 11.514% |
| Model with inverse-cell weights | 89.468% |
| Model without position weights | 90.084% |

Both models substantially outperform these text-blind rules. Of the unweighted
model's 616 net additional correct predictions, 239 occur where fixed-center is
correct and 377 where fixed-center is wrong. This partition is descriptive; it
does not identify a causal percentage attributable to position versus text.
Of the prediction changes, 2,597 move closer to the geometric midpoint (1,374
newly correct, 756 newly wrong), 104 move farther away (42 newly correct, 46
newly wrong), and the remaining rows have equal distance (30 newly correct,
28 newly wrong). The gain is associated with moving inward, including predictions
that still differ from the fixed-center guess.

Giving each of the ten occupied **gold** position bins equal weight reverses the
comparison: 89.661% with position weights versus 88.498% without them. This is a
diagnostic macro average, not the original validation distribution or a human
reading-quality measure. The default still remains unchanged: these results show
an aggregate/edge tradeoff, not universal superiority of either training objective.
The prior uses only the smoke training subset while the models also inherit
earlier training; this is not an upper bound on all possible text-blind predictors.

```sh
npm run model:compare-single-character
```

This follow-up starts both position-weight settings from the original smoke's
initial checkpoint, using its frozen trainer and hyperparameters. It removes a
training row if either side of its labeled gap is exactly one normalized code
point. The original 1,000,000-row sample loses 2,226 rows (0.2226%); the remaining
997,774 rows and their source pointers keep their bytes and shard order. No rows
are relabeled or replaced. The vocabulary and complete 100,000-row validation
file stay identical, including 252 validation rows with a single-character side.
All candidate gaps remain available to the model, including the first and last.

The original full-population position/domain tables remain fixed; their combined
mean is recomputed on the retained training rows for loss normalization. The
same batching algorithm and seed are used, but removing rows changes shuffled
batch composition and can change the update count. This is a one-seed short
continuation from weights that previously saw single-character boundaries;
it does not establish the effect of training from scratch without them.

Artifacts live under `artifacts/single-character-ablation-1m-20261004/` with
frozen sources, a filter plan, aligned source pointers, checkpoints, and logs.
`filter-comparison.json` compares each filtered epoch endpoint with its original
unfiltered counterpart, using full validation, the single-character subgroup,
and its complement. It includes paired document-bootstrap intervals and
length/position breakdowns. The two disjoint groups must reconcile to the full
validation counts. No test scoring, permanent corpus filtering, or shipped-model
replacement is performed. Rerun the same command to resume the experiment.

The [2026-10-04 single-character filter result](single-character-comparison-20261004.json)
uses the unchanged validation population:

| Position weights | Full validation before / after | Single-character sides before / after | Other samples before / after |
| --- | ---: | ---: | ---: |
| Inverse-cell | 89.468% / 89.542% | 80.556% / 65.079% | 89.491% / 89.604% |
| None | 90.084% / 90.054% | 71.032% / 60.714% | 90.132% / 90.128% |

The weighted arm gains 113 net correct answers on the other 99,748 rows but loses
39 on the 252 single-character rows, for a net +74 (+0.074 percentage points;
paired document-bootstrap 95% interval +0.018 to +0.127). The unweighted arm loses
4 and 26 respectively, for a net -30 (-0.030 points; interval -0.074 to +0.013).
Its non-single-character subgroup shows no improvement in this run. Validation
NLL changes from 0.315516 to 0.312744 with weights, and 0.297311 to 0.298488 without.
Both filtered arms lose single-character accuracy, with no newly correct answers
in that subgroup. This does not support adopting a blanket single-character
filter for the currently higher-accuracy unweighted setup.

The intervals describe validation document sampling, not variability across
training seeds. Each original arm made 5,329 updates versus 5,302 after filtering;
batch composition also changes as described above. The existing initialization
had seen single-character labels previously. The tiny aggregate differences
should not be treated as definitive from-scratch training or data-quality results.

The [2026-10-04 label review](reviews/boundary-quality-20261004/findings.json)
audits all 252 single-character validation rows plus 200 randomly selected other
rows (seed 2026100417). It is a **single AI review, not human-adjudicated gold**.
Predictions were hidden during initial annotation, although earlier discussion
had exposed some examples. Sample and annotation hashes were frozen before
joining predictions. [The protocol](reviews/boundary-quality-20261004/protocol.json),
[all labels](reviews/boundary-quality-20261004/annotations.jsonl), and an
[executed notebook](reviews/boundary-quality-20261004/audit.ipynb) preserve the audit.

| Reviewed cohort | Acceptable A | Acceptable with listed alternatives M | Clear issue E | Uncertain U |
| --- | ---: | ---: | ---: | ---: |
| All 252 single-character rows in smoke validation | 133 | 4 | 4 | 111 |
| 200 random rows from the remaining 99,748 | 133 | 50 | 0 | 17 |

A does not mean a unique answer; listed alternatives are not exhaustive. U is
not an error. These differently sampled groups must not be pooled into a corpus
noise rate. Original documents were not recovered, so source IDs identify
prepared provenance rather than independently verified original text.

A separate, explicitly unblinded diagnostic reviews the 29 distinct A/M rows
that newly miss their original label after filtering. Of the weighted arm's 22
such misses, 8 alternative cuts remain acceptable, 7 are clearly defective and 7
uncertain; the unweighted arm's 14 split into 6, 5 and 3 respectively. The groups
overlap. This distinguishes reasonable alternatives from actual word splits;
it is not an independent semantic accuracy measurement. Confirmed defects support
cause-specific cleanup, not blanket removal of single-character examples.
No permanent corpus filtering, test scoring, training or model promotion occurred.

```sh
python3 training/audit_boundary_labels.py --analyze
# Optional report/notebook regeneration needs nbformat; execution needs nbclient/ipykernel.
python3 training/build_boundary_audit_report.py
```

Generated outputs are in `artifacts/label-quality-20261004/`: `artifact.json`,
the packaged [portable report](artifacts/label-quality-20261004/report.html),
`reviewed.csv`, and `human-review.csv` with blank human judgments and no AI labels
or model predictions. The canonical payload is packaged with the data-analytics
plugin's `npm run report:deliver -- --input <artifact.json> --output <report.html>`.

New web-data preparations retain a `web-train-*.provenance.jsonl` sidecar for
each compact training shard. Its rows contain the document byte offset, eligible
pair ordinal before deduplication, and original punctuation. A shared
`web-documents.provenance.jsonl` records the raw Parquet filename, physical row
group and zero-based row, source, and raw-content hash. Byte offsets allow direct
document lookup without scanning the full document index. Checkpoints commit all
data and provenance byte offsets together; recovery truncates all uncommitted
tails. Manifest entries publish sidecar hashes and coverage, and inherited shards
keep any provenance they already have. Existing corpora are not backfilled.

```sh
python3 training/inspect_training_sample.py \
  --manifest /path/to/new-web-data/manifest.json \
  --shard-index 0 --row-index 0
```

Both indices are zero-based; choose a shard with provenance. The command reads
the original document, checks its content hash, regenerates the recorded pair,
and verifies its text, target and punctuation. `--raw-dir` can relocate the raw
files. This supplies document-level traceability, not a pre-cleaning character
offset map. The current web dataset has no original-URL column. An inherited
shard without provenance produces an explicit error rather than an inferred
attribution. New preparation code requires a fresh output directory when the
existing preparation configuration predates this format.

The [2026-09-16 corpus quality audit](CORPUS_QUALITY_AUDIT.md) records a stratified
1,000-row AI semantic review, 20,000-row automatic checks, proxy-label failure
examples, and limits of the resulting quality estimates.

The [2026-09-24 training-teacher audit](TRAINING_TEACHER_AUDIT.md) samples the
current 246,266,210-row v7 training pool: 24,000 automatic checks, 500 probability
reviews, and 64 separate targeted reviews. It distinguishes acceptable target
pauses from damaged inputs, compares current model predictions, accounts for
training loss weights, and documents the limits of attributing errors to noise.
All 564 judgments are preserved in
[the review record](training-teacher-review-20260924.json).
The [follow-up close reading](GOOD_TEACHER_ERRORS.md) covers all 45 model misses
among the 492 acceptable targets. It distinguishes poor local cuts, locally
plausible cuts that miss the original structural boundary, acceptable alternatives,
and unresolved cases; these judgments do not define a new backend accuracy.
Under the user's overall-semantic acceptance standard, the first two categories
both fail: 32 unacceptable results, 9 acceptable alternatives, and 4 unresolved.

The [model-side diagnosis](GOOD_TEACHER_DIAGNOSIS.md) replays the frozen reviews:
28 of the 32 unacceptable errors had the original target in the top three;
22 had full input coverage at both compared gaps. It also identifies a padding
invariance bug: activations at padded positions inside convolution blocks affect
real-token logits. A 500-row Metal control reproduces the original batched
predictions and isolates this from weight export or device differences. See the
report for actual backend recursion results and the limits of these counts.

The [preparation-contract audit](DATA_PREPARATION_AUDIT.md) checks all 5,092,343
holdout rows plus 3,781,496 rows from 32 seeded training shards. It finds no
invalid target/encoding metadata or exact input conflicts in that checked union,
and documents strong position weighting, implicit length weighting from variable
batch sizes, and measured differences between training and backend inputs. These
are confirmed mechanisms or design tradeoffs, not measured causes of a 90%
accuracy ceiling. The audit does not change the data or start training.

The [source-by-source teacher review](SOURCE_TEACHER_AUDIT.md) separately reads
350 current-policy candidates from 13 source groups, including seven web
sub-sources: 343 usable, 2 unsuitable inputs, and 5 unresolved/task-fit concerns.
It reuses cached document pools, excludes known prior review documents, and
keeps original context. These are not verified members of the current training
set; the small document-balanced review is not a corpus-wide bad-label estimate.
All cases are in [the source review record](source-teacher-review-20260924.json).
The [expanded source review](SOURCE_TEACHER_AUDIT_R2.md) adds 650 disjoint
documents, reaching 1,000 reviewed documents (999 distinct input/target pairs):
973 usable, 15 unsuitable, and 12 unresolved. It checks 17 flagged web documents
against original Parquet and confirms exact current training membership for one
unsuitable academic example and one quote-position concern. Candidate quality
and actual training membership remain separate; no noise-causality estimate is
made. [All 650 new judgments](source-teacher-review-r2-20260924.json) retain source
context, provenance, and the targeted lookup results.

[Local web prose screening](WEB_SCREENING.md) documents an earlier screening
experiment. The current rebuild does not use its paragraph selection rules.

The existing dated corpora, evaluation splits and paused web-continuation job
were prepared under v1. Changing this default does not rewrite their samples or
alter their frozen source snapshots. Before a v7 run, regenerate train,
validation and test from the original documents into fresh directories while
preserving document ownership and overlap checks; dropping disallowed labels
alone cannot restore punctuation previously removed from their inputs. Current
training/evaluation loaders compare the proxy set, normalization, numeric
context, boundary-neighbor, line-boundary, source-filter, symbol-window and surface-filter rules, and reject incompatible metadata. Historical jobs can continue
using their recorded v1 source snapshots. Accuracy across these label standards
must not be compared as if it were measured on the same holdout.

## Current rebuild: local sources plus 100 million web pairs

**v7 preparation restarted (2026-09-17, 22:47 CST).** The latest
[independent label audit](V7_LABEL_QUALITY_AUDIT_R2.md) reviewed 500 fresh pairs
and found no confirmed incorrect targets in that sample. This is sampling
evidence, not a guarantee that the full corpus is free of label noise.
The new run rebuilds data under the audited v7 rules. Earlier v3/v4 runs stay
stopped; their frozen snapshots do not contain all the current fixes.

```bash
npm run model:retrain-surface
# After an interruption:
npm run model:retrain-surface -- --resume
# After the first epoch and its evaluation finish, continue through epoch 2:
npm run model:retrain-surface -- --resume --epochs 2
```

`--epochs` is the total epoch target. Extending a completed run archives its
checkpoint, exports and metrics under `completed-epochs/epoch-N/`, then resumes
the same weights and optimizer on the same data. Subsequent `--resume` commands
reuse the extended target without needing `--epochs` again.

[run_surface_retraining.py](run_surface_retraining.py) rebuilds every cached CLUE
entry, the full Wikipedia dump and the cached Chinese multi-style L3 source
under v7. It then adds **100,000,000 distinct web training pairs**, sampled across
all 256 downloaded Chinese web files. This is the web target, not the total
training count; the rebuilt older sources are additional. Existing source-level
Chinese-text checks remain. No score gate, per-document cap or sequence-length
cap is added. Historical training inputs are excluded from the new web sample.

The run is `training/artifacts/chinese-line-web-100m-16conv-v7-20260917/`.
The earlier v3 preparation was stopped before model training and is preserved in
its original directory. The new run regenerates original sources because source
barriers and symbol windows require text that old compact samples have lost;
filtering those old samples alone cannot reconstruct the correct context.
Initialization explicitly selects the validated epoch 1 **best_state** from
`web-mix-20m-192ch-16conv-20260915/candidate/training-state.pt`. The checkpoint
also contains later partial-epoch weights; those are not selected. The source
checkpoint hash, selected epoch's validation metrics, exact tensor-copy
verification and discarded numerical probe are recorded in
`initialization-verification.json`.

The architecture remains 16 convolutions / 192 channels / 3,496,329 parameters,
with the same 8192 vocabulary IDs. A fresh AdamW optimizer trains one new epoch
at learning rate 0.0003, using the previous domain weighting, batching and
checkpoint-selection settings. Before any updates, the complete rebuilt
validation split establishes the baseline. Each epoch uses the same complete
validation split; the selected checkpoint receives a final full test evaluation.
Whether to train further should be decided from the rebuilt validation results.

The [2026-09-18 training speed study](TRAINING_SPEED_STUDY.md) compares CPU and
MPS implementations on identical batches and checks numerical agreement.
All maintained training entry points now require **Apple Metal (MPS)** by
default, using FP32 native Conv1d and one host CPU thread. CPU training mode and
its optimized three-tap convolution branch have been removed. `--device cpu`
is rejected; a machine without MPS reports an error before training. This also
applies to smoke training and the boundary-encoder comparison. Small workloads
can still be slower on GPU once startup/dispatch overhead is included; requiring
MPS simplifies maintenance rather than promising speedups at every batch size.
The [small-smoke comparison](SMALL_SMOKE_SPEED_STUDY.md) measures that tradeoff:
CPU wins the 32-example case and the smaller model; MPS wins the production
model's 4,096-example case when including process startup and evaluation.
The measured CPU savings were only 0.6–0.8 seconds for the small cases, so we
keep a single Metal training implementation. Temporary CPU/MPS benchmark
drivers and their copied trainer have been removed; measurements and reports
remain as historical evidence. Numerical reference tests use the same model
implementation and do not introduce a second CPU training path.

New MPS runtimes use a 0.4 memory fraction with a 0.24 low watermark. After
completed batches, persistent driver usage above 60% of the hard limit triggers
an atomic checkpoint and exit 75. The surface coordinator then starts a fresh
worker with `--resume`, preserving weights, AdamW, batch order and progress.
Completed shards also release free allocator cache. This addresses the first
long MPS run's memory-limit failure without reducing training/evaluation data.

Parameter names/shapes, batching, loss and full evaluation are unchanged.
Checkpoints store CPU tensors, including AdamW moments, and historical CPU
checkpoints can still resume on MPS. CPU data processing, checkpoint I/O,
reference inference and browser inference remain supported.

To migrate an existing frozen CPU run, first send SIGTERM to its coordinator
and wait for `status.json` to report `stopped`, then run once:

```bash
npm run model:retrain-surface -- --resume --refresh-training-code --device mps
```

This preserves the original preparation snapshot and creates a separately
hashed training runtime under `training-runtimes/`, with a copy of the last CPU
checkpoint. `training-runtime.json` records the chosen runtime and device.
Subsequent interruptions use the ordinary `--resume` command above; it reuses
that frozen MPS runtime automatically. Migration preserves the optimizer,
sample order and epoch position. Original historical snapshots remain immutable
for provenance, and an active run keeps its frozen MPS code. They are not updated
in place during this cleanup. Older CPU snapshots require an explicit MPS
migration instead of being silently launched by the current surface coordinator.
Historical depth/web runners also explicitly request MPS; pre-MPS snapshots
cannot run through those commands without a deliberate runtime migration.
CPU and GPU floating point results need not be bit-identical.

Hardware checks (MPS tests skip on other machines):

```bash
PYTHONPATH=training/.deps:training DEBUG=0 python3 -m unittest training/test_training_device.py -v
```

These cover gradients, evaluation, portable optimizer save/resume, MPS-only
entry points, complete fixture validation/test, smoke and comparison training,
browser export and frozen-runtime integrity. The obsolete Node assertion for
CPU-specific convolution code is replaced by these behavioral Python checks.

Old document holdouts stay reserved. Revised evaluation pairs are screened
against inherited and newly rebuilt training inputs using Han-projected input
identity independent of the target gap. New web documents use the existing
96%/2%/2% train/validation/test hash partition. Web holdouts supplement the
rebuilt local holdouts, and all splits use the same v7 preparation rules.
The final evaluation directory is
`training/data/processed/chinese-line-web-100m-16conv-v7-20260917-eval/`.

Wikipedia holdouts are retrieved by their original page IDs through the dump
index, including documents with no surviving samples. Resampling a fixed
number of cleaned articles can select different pages when extraction changes.
The initial v7 attempt stopped at that recovery check before any training; its
logs and incomplete base output are archived with a `.failed-reference-lookup-`
suffix. The restarted run freezes the corrected lookup code and the same v7
cleaning policy.

The coordinator freezes source code and weights, records stage logs and
`status.json`, and automatically starts training after preparation. Resuming
truncates uncommitted shard tails before rebuilding deduplication state;
unfinished base/audit stages restart in fresh directories. `corpus-coverage.json`
records the final training and evaluation counts. Completion exports a separate
candidate; promoting it to the backend is a separate action.

## Historical full local-corpus candidate

```bash
npm run model:retrain-context
# After an interruption:
npm run model:retrain-context -- --resume
```

The pipeline in [run_context_pipeline.py](run_context_pipeline.py) uses all cached
CLUE text entries, all extracted Chinese Wikipedia articles, and every row of the
cached Ultra-FineWeb Chinese multi-style Parquet. It does not cap documents,
pairs per document, sequence length, or the total synthetic sample count.
Deduplication, document holdouts, article extraction and the existing synthetic
Chinese-text quality checks still apply. It does not fetch additional corpora.

The default run is `training/artifacts/unicode-context-192ch-12conv-20260913/`.
Its stages are preflight, base regeneration, inherited-checkpoint overlap audit,
Wikipedia preparation, synthetic preparation, new-training overlap audit,
vocabulary construction, training, export and browser evaluation.

The initial checkpoint is the completed best epoch 2 of
`all-local-192ch-12conv-20260912`. A separate copy is recorded with its SHA-256.
The new candidate retains 192 channels and six residual blocks (12 kernel-3
convolutions). It starts two new epochs with learning rate 0.0003, domain-weight
power 0.65, equal overall/macro validation selection weights, gradient clipping
at 1.0, batches of at most 512 examples/8192 padded tokens, and five CPU threads.

The vocabulary preserves all existing IDs and extends to at most 8192 tokens
using training data only, with dedicated ASCII and common punctuation entries.
Embedding weights are copied by token identity; convolution and scoring weights
are inherited. New tokens are initialized separately and AdamW starts fresh.
The initialized model is evaluated on the new validation set before updates, so
checkpoint selection can also retain initialization (new-run epoch 0).

Original document-to-split assignments are preserved. Before establishing the
new holdout version, a conservative Han-projection comparison removes pairs seen
by inherited checkpoints. The new training shards must have zero exact input/gap
overlap with that evaluation set. Historical holdouts and reports are preserved;
old and new input-policy scores are not directly comparable.

`status.json`, `completed-stages.json`, stage logs and `corpus-coverage.json`
record progress. Preparation and training support restart; incomplete base/audit
outputs are archived before retry. A source hash change stops the next stage.
The trainer saves atomic checkpoints every four shards and at safe interrupt
boundaries. Completing the pipeline exports a separate candidate and does not
replace the extension's bundled weights.

## 16-layer continuation and Chinese web download

```bash
npm run model:download-web
npm run model:continue-depth
# After an interruption, repeat the download command and resume training:
npm run model:continue-depth -- --resume
```

These are independent jobs. The download fetches only the Chinese split of
`openbmb/Ultra-FineWeb`, pinned to revision
`02c85641e3d19a854be2e09139c25adaa9518063`: 256 Parquet files totaling
324,321,485,291 bytes (302.05 GiB). It uses three concurrent transfers, resumes
partial files, verifies every file against the upstream SHA-256, and reserves
32 GiB of free disk space. Connection outages are retried with up to 60 seconds
between attempts until connectivity returns or the job is stopped. Other errors
retain bounded retries; size and checksum mismatches stop the download.
Its manifest, verified-file list, progress and upstream
documentation are stored in `training/data/raw/ultra-fineweb-zh/`.
The upstream license also requires checking the original component datasets'
terms. Downloaded text is not automatically added to a running epoch; the separate
web continuation below prepares samples and protects the holdouts first.

The continuation run is
`training/artifacts/unicode-context-192ch-16conv-20260914/`. It inherits the
completed best epoch 2 from `unicode-context-192ch-12conv-20260913/candidate/`
and trains two new epochs on the same 74,121,940 examples, with all 727,578
validation and 766,489 test examples. Channels (192), vocabulary (8192), learning
rate (0.0003), domain weighting, batch limits and selection criteria remain the
same. AdamW starts fresh for the expanded architecture; new-run epoch 1 follows
the inherited epoch 2.

Eight residual blocks give 16 convolutions and 3,496,329 parameters. All inherited
tensors are copied exactly. The two added blocks start with zero residual scales,
so initialization preserves the old logits; the scales and convolution branches
then learn during optimization. The structural gap context can grow to 34
characters, up to 17 on each side. The preflight verifies weight/output equality,
nonzero learning gradients and exported JavaScript inference. Its probe updates
are discarded before formal training, which evaluates the complete validation
split before making updates and includes initialization in checkpoint selection.

The run snapshots its training/export code and initialization checkpoint so later
workspace edits do not change a running experiment. It saves atomic checkpoints
every four shards and at safe interrupt boundaries. `status.json`, `training.log`,
`preflight/verification.json` and the candidate checkpoints record progress.
Completion exports a separate candidate; the bundled backend stays on its
currently released model until explicitly promoted.

## Prepare all remaining web records (2026-10-09)

`prepare_remaining_web_data.py` exhausts the unvisited suffix of each of the
256 local Parquet files. The reconstructed cursors in
`artifacts/web-source-usage-20261009/usage-and-cursors.json` identify 23,277,777
previously visited records and **107,728,685 remaining records**. This is a
document count, not a training-pair count. Cursor evidence is checked against
historical log hashes, shuffled row-group order and Parquet row counts.

```sh
python3 training/prepare_remaining_web_data.py \
  --output-dir training/data/processed/ultra-fineweb-remaining-v7-20261009
# After interruption: keep the same output directory and frozen plan.
python3 training/prepare_remaining_web_data.py \
  --output-dir training/data/processed/ultra-fineweb-remaining-v7-20261009 --resume
```

This preparation deliberately freezes the original **unicode-context-v7 /
surface-noise-v4** extraction code from the 350m preparation. It does not apply
the newer v8 filters or a score threshold. All previously visited documents,
including the entire last partially consumed document, are skipped. Old training
shards are not included, and pairs are not deduplicated against old training data;
previously unvisited documents can still repeat text seen before. New pairs are
deduplicated internally and against the byte-identical inherited validation/test
sets. The original 96/2/2 document partition and holdout document registries are
also retained. Bloom false positives can discard additional pairs. No claim of
near-duplicate removal is made.

Each transaction scans up to 5,000 documents and atomically publishes gzip
training, pair-provenance and document-provenance files together. Training readers
support these shards directly. The full deduplication snapshot is saved at file
boundaries and graceful stops; recovery replays committed shards after that
snapshot and discards unfinished transactions. Defaults use a 4 GiB deduplication
bitmap and a 64 MiB holdout bitmap, reserving 32 GiB of free disk plus room for
the next bitmap snapshot. A low-space stop is resumable. `--session-documents N`
allows a finite pilot followed by `--resume` without that option to exhaust the rest.

`plan.json` freezes inputs, rules and code; `status.json` and `preparation.log`
show progress. The final `manifest.json` is published only after all remaining
records are processed. Preparation keeps original eligible pairs; later training
still applies minimum two characters per side and maximum 256 total characters.
No training or model promotion starts automatically. Trace v7 samples with the
frozen `source/inspect_training_sample.py` and `PYTHONPATH=training/.deps` so that
pair regeneration uses the same policy. Compressed document offsets are measured
in uncompressed bytes and require decompression within one chunk.

## Web data continuation

The current v7 expansion (2026-09-20) retains all **146,266,210** existing
training pairs and adds **100,000,000** distinct web pairs: **200 million web /
246,266,210 total**. The audited cleaning policy, 16-layer / 192-channel model,
8,192-character vocabulary and complete validation/test files stay fixed.
It warm-starts from the completed v7 **best epoch 2**, initializes a new AdamW
optimizer for the changed corpus, and trains one new epoch before evaluation.
Initialization is evaluated on the full validation split and remains eligible
for best-checkpoint selection. Artifacts are separate from the 100m web run.

```bash
npm run model:expand-web-v7
# Resume preparation or training after an interruption:
npm run model:expand-web-v7 -- --resume
```

The added pairs use 1,024 shards. The coordinator automatically resumes saved
batches when Metal requests a worker refresh, and honors explicit stop signals.
The following smaller runs are historical configurations.

`npm run model:continue-web` prepares 20 million new training pairs across all
256 downloaded Chinese web files, combines them with the existing 74,121,940
pairs, and continues the best 16-layer epoch-1 weights for two epochs. It keeps
the vocabulary and architecture fixed. Whole-document web holdouts supplement
the unchanged original validation/test sets; no evaluation subset is used.
Historical input deduplication runs before sampling. Preparation and training
resume with `npm run model:continue-web -- --resume`.
See [the run design and overlap controls](WEB_CONTINUATION.md).

After the first mixed epoch improved full validation accuracy to 87.4859%,
`npm run model:expand-web` starts a separate continuation from that best epoch.
It retains the existing data, adds another 20 million distinct web training
pairs (40 million web / 114,121,940 total), and keeps both complete evaluation
splits byte-identical. `npm run model:expand-web -- --resume` resumes this run.

## Evaluation

```bash
npm run model:baseline
```

The real defaults are the bundled `src/boundary-model-data.js` model and
`training/data/processed/chinese-line-web-100m-16conv-v7-20260917-eval/validation.jsonl`.
The rebuilt holdout must exist before this command can run. `--model`,
`--input` and `--output` are optional explicit experiment paths; they cannot bypass
input-policy checks. The evaluator rejects mismatched model/data representations,
invalid proxy labels anywhere in the input, and recorded checksum mismatches.
This browser smoke check uses a deterministic sample of 500 examples per domain
(seed 20260911); formal training always evaluates the complete validation/test splits.
Keep input and sample hashes fixed for within-version comparisons.

The bundled model is the completed epoch-1 best checkpoint from the fresh run
(`fresh-lr3e-4-max256-20261005/candidate`), promoted on 2026-10-07. It retains 192
channels and 16 convolution layers. See [BUNDLED_MODEL.md](BUNDLED_MODEL.md) for
its training configuration, retained-split metrics and export provenance.
The backend first splits clauses on original Chinese proxy punctuation,
physical line boundaries and original enumeration commas (`、`). Enumeration
commas stay in the preceding fragment and its model tokens; they are runtime
reading boundaries, not additional training proxies. Chinese colons and
compatibility punctuation remain context. All clauses
follow the same rule:
leave fewer than 13 visual units intact and send eligible clauses with their
remaining Unicode context to the model. Numeric separators remain within numbers;
original text and UTF-16 offsets are preserved. Model cuts require immediately adjacent
Han characters on both sides in the original text, plus browser word protection.
The default [recursive model inference](BUNDLED_MODEL.md#backend-preprocessing) scores
each eligible child afresh and computes softmax over its internal gaps.
Every fragment of at least 13 visual units requires probability strictly above 45%.
Rank eligible positions by raw logit. An uncertain clause stays intact.
The [confidence audit](CONFIDENCE_THRESHOLD.md) measures original-input predictions
for the previous, pre-web model; its precision figures do not describe this new
checkpoint or establish the accuracy of recursive child decisions.

## Verification and small experiments

The completed [192/256-channel continuation pilot](WIDTH_COMPARISON.md)
(2026-09-22) used the same 10,015,214 training samples and complete fixed
validation/test sets. The 256-channel endpoint gained only 0.0436 percentage
points in test accuracy over the equally trained 192-channel endpoint, with
57.7% more parameters and 52.1% longer JS inference time. Neither arm improved
the validation selection score over its inherited weights. **Keep the existing
192-channel production model.** Setup, results, limitations and local artifact
paths are preserved in the experiment report.

The [lower-learning-rate pilot](LEARNING_RATE_COMPARISON.md) reuses that frozen
192-channel control and tests 0.0001 and 0.00003 with identical training samples,
initial weights and optimizer settings. It selects on full validation before
testing the selected candidate. Run or resume with
`python3 training/run_learning_rate_comparison.py [--resume]`.

The [full-data learning-rate comparison](FULL_LEARNING_RATE_COMPARISON.md)
replayed the preceding model's 246,266,210-pair epoch from its original
pre-expansion weights, changing only 0.0003 to 0.00003. It preserves the original
seed, optimizer setup, batches and fixed complete holdouts, and reuses the existing
backend model as control. Run or resume with
`python3 training/run_full_learning_rate_comparison.py [--resume]`.

The completed [16/20-layer comparison](DEPTH_COMPARISON.md) reused the 16-layer,
0.00003 control and trained only a 20-layer candidate on the same 10,015,214 pairs.
The 20-layer endpoint gained 0.01035 percentage points on full validation and
0.00148 on full test (38 additional correct labels), with 22.8% longer JS inference.
Longer inputs improved slightly more, but the measured benefit remains too small
to justify adopting 20 layers. Keep 16 layers. The report preserves the protocol,
context-coverage breakdown and local reproduction/resume commands.

The [receptive-field analysis](RECEPTIVE_FIELD_ANALYSIS.md) scans all 74,121,940
training examples and stratifies the complete epoch-2 validation results by
context coverage. It records the coverage expected at larger CNN depths and the
limits of using those statistics to choose a model size.

The completed [CNN / Transformer experiment](CNN_TRANSFORMER_EXPERIMENT.md)
used identical training data and full validation/test splits. CNN reached 68.60%
test accuracy versus 60.85% for Transformer and led in all five test domains.
The experiment is closed; its dedicated scripts and generated artifacts were
removed after preserving the setup, results, limitations and audit fingerprints.

An earlier experiment with recursive fragment rescoring without a confidence gate
showed higher inference cost without a clear quality benefit. Only its historical
[conclusion](SCORING_COMPARISON.md) is retained; the original scripts and generated
reports have been removed. That result does not describe the current default.

A separate [recursive >90% experiment](RECURSIVE_CONFIDENCE_EXPERIMENT.md)
compares fresh child inference with fixed original probabilities. Its
`scoringStrategy: "recursive-model"` became the default on 2026-10-08, with the
then-existing 50% gate: children above 12 visual units receive fresh CNN inference.
On 2026-10-09 a 30–50% sweep found the highest first-round F1 at 45%; this does not
establish full recursive quality. The current default is at least 13 visual units
and >45%, following that first-round F1 selection.
Cached logits and fixed original probabilities remain explicit offline controls.
See [current backend behavior](BUNDLED_MODEL.md#backend-preprocessing).
Reproduce the historical strategy comparison with
`node training/compare_recursive_confidence.mjs`. The article examples are fixed
text-node snapshots in the script, independent of the product site's copy.

```bash
npm test
npm run smoke:data
python3 training/run_smoke.py --epochs 2
# Extend an otherwise unchanged small run:
python3 training/run_smoke.py --epochs 6 --resume
```

`npm test` uses small temporary fixtures, not downloaded corpora or ignored model
artifacts. The real Parquet fixture installs pinned PyArrow on first use if it is
missing; Python 3.9+, pip and first-install network access are required. Full
training installs pinned PyTorch/tinygrad through the existing local bootstrap.

[preflight_context.py](preflight_context.py) verifies all inherited embeddings
with deliberately permuted IDs, checks unchanged non-embedding weights, runs an
actual optimization step involving a colon, exports it and compares browser and
PyTorch logits. Its tiny metrics are implementation checks, not performance claims.

See [BUNDLED_MODEL.md](BUNDLED_MODEL.md) for the currently shipped model and
[HISTORICAL_TRAINING.md](HISTORICAL_TRAINING.md) for previous experiment records.
