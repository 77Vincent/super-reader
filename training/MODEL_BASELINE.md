# Model quality baseline — 2026-09-11

The deployed model scores **78.56% top-1 and 94.44% top-3** on a fixed,
domain-balanced validation sample. News is the weakest domain at **66.60%**.
These numbers measure recovery of a removed punctuation boundary. They do not
measure the quality of complete reading chunks.

No model weights or runtime segmentation rules were changed for this baseline.

## Reproduce

```bash
npm run model:baseline -- --output training/artifacts/baseline-2026-09-11.json
```

Requires the existing local validation corpus. The evaluator uses the actual
`src/backend/inference.js` and exported weights, with no training dependency.
The JSON includes all 2,500 predictions, their IDs, targets, confidence, ranks,
and the segmentation examples below. Generated reports are ignored by Git;
this Markdown file preserves the initial results.

- Checkpoint SHA-256: `5c44196f1660ec64e5dff492aa4a3b4b27ceb85187f15eae98d9bde47c332677`
- Model source SHA-256: `a80a6b549c8597b89717f9d76c11545e2198b0c9c50299e585378c1762855253`
- Model: 985,349 parameters; 4,096-token vocabulary; 128 channels; four residual blocks, eight kernel-3 convolutions.
- Input: `training/data/processed/validation.jsonl`, 829,297 available pairs.
- Input SHA-256: `0ed114ad6032dcf5989a173768cd87356cc34ad21e3cc3a02eb3df5c1100a464`
- Sampling: 500 pairs per domain, separate seeded reservoirs, seed `20260911`.
- Selected-ID SHA-256: `ce73a6226e0e421c9fbf23f73d59e1db1a52d0789db06fc4f39299378562354d`.
- Environment: Node v25.9.0, ICU 78.3, macOS arm64, Apple M5 Pro.

Keep the input, seed and sample size fixed when comparing candidates. Use
validation to tune; do not choose parameters by repeatedly inspecting test
scores. Multiple sampled pairs can come from one document, so small differences
should not be treated as conclusive independent-sample improvements.

## Current model measurements

| Domain | Available validation pairs | Evaluated | Raw top-1 | Raw top-3 | Balance-only top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Academic | 5,468 | 500 | 78.20% | 93.20% | 79.00% |
| Dialogue | 10,053 | 500 | 85.20% | 97.40% | 85.40% |
| Encyclopedia | 8,341 | 500 | 78.80% | 95.80% | 78.60% |
| News | 4,832 | 500 | 66.60% | 89.60% | 68.00% |
| Wikipedia | 800,603 | 500 | 84.00% | 96.20% | 84.80% |
| Equal-domain aggregate | 829,297 | 2,500 | **78.56%** | **94.44%** | **79.16%** |

Raw top-1 uses the highest unmodified model logit. Balance-only chooses one gap
using `logit + log(p * (1 - p))` on the same complete input. It does not apply
the eight-unit stopping threshold, word protection, punctuation pre-splitting,
or recursive chunking. Its score is therefore not full-pipeline accuracy.
Balance changes 98 of 2,500 choices: 43 become correct, 28 become incorrect,
and 27 change between two incorrect choices. The net gain is 0.60 percentage
points on this sample.

Other raw-model measurements:

- Mean reciprocal rank: **0.8677**; mean absolute boundary error: **1.33 Han characters**.
- Cross-entropy: **0.6798**; unknown-token rate: **0.833%**.
- Warm sequential Node inference: approximately **5 ms median / 12 ms p95**
  per sampled input. These are `scoreTokens()` timings, excluding DOM, messaging,
  initialization and chunking; they are not browser end-to-end timings.

| Input length, Han characters | Count | Raw top-1 |
| --- | ---: | ---: |
| 2–8 | 251 | 88.84% |
| 9–16 | 805 | 80.12% |
| 17–32 | 1,110 | 76.22% |
| 33–256 | 334 | 74.85% |

No input over 256 characters was selected, so this sample does not evaluate
production window-boundary behavior. All sampled inputs were scored intact.

| Highest-gap softmax share | Count | Observed raw top-1 |
| --- | ---: | ---: |
| Below 50% | 394 | 41.62% |
| 50% to below 90% | 922 | 70.28% |
| 90% to below 99% | 670 | 95.52% |
| At least 99% | 514 | 99.61% |

The highest-confidence group covers only 20.56% of this sample. These labels
always contain one target boundary; there are no negative examples meaning
“this text should remain unsplit.” This confidence audit cannot establish a
stopping rule for arbitrary phrases or validate every later recursive cut.

## Historical full-test result

The matching checkpoint's archived training metrics report **81.956% top-1**
on 872,558 test pairs and **78.619% macro-domain accuracy**. This full-test
number was recorded by the PyTorch evaluation, not rerun here. Its natural
domain mix differs from the equal-domain validation sample above; the two
aggregate scores should not be interpreted as a regression.

Source: `training/artifacts/wiki-ultra-domain-weighted-128ch-2ep/smoke-metrics.json`.
The safetensors checkpoint hash matches the exported model metadata. The
artifact records three epochs, with epoch 2 selected. Epoch 3 reduced overall
validation accuracy from 81.99% to 81.28% and macro accuracy from 78.12% to
77.75%. Additional epochs already failed to improve this particular run.

## Current segmentation outputs for review

These are observations, **not gold answers**. The five demo cases were copied
from `demo.html`; the other seven are authored diagnostic examples. `｜` marks
an inserted divider. Inline DOM nodes are passed separately, as in production.
Cases are stored in `evaluation-cases.json`; no DOM traversal runs in this check.

| Case | Current output |
| --- | --- |
| Demo intro | 加载扩展后，点击浏览器｜工具栏中的 Super Reader 按钮｜开启阅读辅助，再次点击即可关闭。 |
| Demo reading | 真正高效的阅读｜并不是追求速度，而是帮助大脑｜更快地识别信息｜结构。 |
| Demo breakfast | 我一直在思考｜明天早上的早餐｜吃什么，也希望阅读的时候｜能够更加｜轻松地找到｜句子中的重点。 |
| Demo inline nodes | `这里有` / `需要保留的｜加粗文字` / `，以及 ` / `需要保留的链接｜文字` / `。` |
| Demo mixed text | English words 和｜中文可以同时出现，数字 1/4 与标点｜也应当保留。 |
| Steering wheel | 司机一直紧握方向｜盘｜控制车辆前进。 |
| News | 有关部门｜将进一步完善｜公共服务体系，推动优质资源｜向基层延伸。 |
| Dialogue | 我以为你｜已经把昨天说的｜那件事情｜处理好了，结果｜你居然还没有开始。 |
| Academic | 实验结果表明｜该方法｜能够有效降低复杂｜环境下的预测误差。 |
| Traditional | 我們希望｜透過清楚的資訊｜結構｜幫助讀者｜理解文章內容。 |
| Quantity | 请把3个苹果｜和1/4杯牛奶｜放在桌子上。 |
| Short | 今天下午开会。 |

In the steering-wheel example, both raw and balance-weighted first choices are
`司机一直紧握方向盘｜控制车辆前进`. The left part has nine Han characters, so
the current eight-unit threshold requests another cut using the original
scores. ICU word segmentation produces `方向` and `盘` separately, allowing
the cut inside `方向盘`. This is a downstream chunking failure despite a useful
first model prediction; it should not be counted as a first-boundary error.

## What to improve next

1. Establish human-reviewed acceptable and forbidden boundaries for reading
   chunks, including phrases that should stay whole. Keep those evaluation
   cases separate from any examples used for fine-tuning. Current labels are
   created by joining adjacent punctuation-separated fragments after removing
   non-Han content; that supervision differs from splitting within a clause.
2. Inspect news errors first. Some disagreements are valid alternative cuts:
   the validation target `北京｜城六区陆续启动最后一批小升初特长生报名和初试`
   is predicted as `北京城六区陆续启动｜最后一批小升初特长生报名和初试`.
   Others can break a term: `与普通液压机比较｜多向模锻液压机…` is predicted as
   `与普通液压机比较多｜向模锻液压机…` with 97.70% softmax share.
3. Compare one change at a time against this fixed baseline. Track model
   boundary accuracy separately from complete segmentation behavior and
   browser performance. Avoid treating more data, more epochs, or a confidence
   cutoff as an established solution before checking those examples.
