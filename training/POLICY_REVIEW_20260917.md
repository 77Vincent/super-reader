# Data-policy audit, 2026-09-17

The `unicode-context-v4` rebuild is **paused before model training**. Do not resume
its frozen source snapshot as though the issues below were fixed. This audit did
not change production generation rules, prepare full datasets or start training.

## Evidence and scope

Reused saved probability samples: 32,768 web documents covering all 256 source
files, 5,000 L3 documents, 3,674 extracted Wikipedia documents, and 127,786 CLUE
records. The frozen v4 generators produced 1,786,572 candidate occurrences before
global deduplication or training quotas. Independently read 600 new candidates,
excluding documents used in the earlier 420-example audit and exact targets from
both earlier audits. A separate review covered 120 rejected candidates.

| Source | Reviewed | No obvious local issue | Findings |
| --- | ---: | ---: | --- |
| Web | 400 | 382 | 7 surface, 7 language, 3 metadata, 1 wrong target |
| L3 | 80 | 79 | 1 structured-list boundary question |
| Wikipedia | 60 | 57 | 3 inputs damaged by template removal |
| CLUE | 60 | 60 | None detected in this draw |

These are one assistant's local language judgments, not human gold labels or
corpus-wide error estimates. Domain facts were not verified. Ordinary whitespace,
terminology, lists and minor spelling differences were not automatically failures.

## Confirmed issues

- **Structured separators can still become targets.** Original web text
  `a* 由0或多个a构成的字符串集合 {空，a，aa，aaa，a…a(n个a)}` produces
  `…{空｜a`. “空” passes the Han-neighbor condition. L3 also contains name-list
  fields such as `王海云（庆云县，小学语文）`; both neighbors are Han, but this
  is not a natural clause boundary.
- **Upstream extraction can destroy meaningful text.** Original Wikipedia XML
  contains `{{Coord|53|S|48|W||display=inline}}`, Wikidata postal/INSEE codes, and
  team-name templates. Removing these produces incomplete prose such as
  `的邮政编码为｜INSEE市镇编码为`. All three flagged examples were traced back
  to the original compressed XML, not inferred from the extracted fragments.
- **Cleanup can bypass the physical-line barrier.** Removing a multiline fenced
  block, Wiki template, reference, comment or math block can join its outer text.
  Of 89 adversarial checks, 83 pass and 6 fail this requirement. This is a test
  matrix, not a frequency estimate. Signed numbers, existing Han-neighbor behavior,
  actual newlines without intervening removal, and non-proxy glyphs pass.
- **Surface residue remains.** Examples include literal `\r`, HTML breaks,
  Markdown stars, and a book title written as `《论语。泰伯》`, whose internal
  period breaks the surrounding input. Other residual problems require semantic
  interpretation and remain outside the agreed surface-cleaning scope.

## Cost and false-positive checks

Among 80 deliberately sampled neighbor-rule rejections, 42 have an interpretable
ordinary clause boundary, including technical prose and quotes/dates. Among 40
surface-rule rejections, 4 are ordinary explanations of `print()`, `assert()`,
`show()` or `(?)` rejected by the empty-template pattern. These are separate
stratified draws; their proportions do not estimate total data loss.

A **study-only candidate**, rejecting pairs touching explicit residue or guarded
formal/title spans, affected the following existing candidate occurrences:

| Source | Candidate pool | Local rejection | Whole-line rejection |
| --- | ---: | ---: | ---: |
| Web | 1,115,986 | 8,486 (0.760%) | 36,304 |
| L3 | 204,196 | 122 (0.060%) | 510 |
| Wikipedia | 81,487 | 294 (0.361%) | 1,382 |
| CLUE | 384,903 | 898 (0.233%) | 8,984 |

The candidate flags replacement characters, literal newline escapes, a narrow
HTML-tag list, Wiki conversion markup, Markdown stars, formal braces, and book
title spans containing proxy punctuation. It does not reconnect across rejected
fragments. In the 600 reviewed examples it removes six flagged examples and one
previously usable example. These figures measure cost, not quality improvement;
the proposal was designed using these findings and is not independently validated.
Upstream extraction fixes and their cost are not included in this estimate.

## Required sequence before full preparation

1. Preserve physical-line barriers throughout upstream cleanup. Mark removed
   substantive content as a barrier and reject affected pairs; do not silently
   concatenate its surroundings. Treat removable reference metadata separately
   from templates representing missing prose or values.
2. Implement and test conservative local surface/span rules, retaining normal
   Chinese/English mixtures, numbers, ordinary brackets and whitespace. Include
   legitimate technical examples as counterexamples. Keep the existing signed
   numeric and boundary-neighbor decisions.
3. Run a bounded end-to-end preparation on new documents not used to design the
   rules. Check actual output, all source extractors, train/validation/test policy
   parity, source ownership, and rejected examples before freezing a new version.
4. Only then begin full preparation. Semantic flaws remain accepted scope;
   structured name lists remain a documented boundary ambiguity rather than a
   reason to reject all parenthesized prose.

## Local audit artifacts

- [Readable report](artifacts/v4-policy-validation-20260917/report.html)
- [All 600 judgments and original lines](artifacts/v4-policy-validation-20260917/reviews.json)
- [Rejected-candidate judgments](artifacts/v4-policy-validation-20260917/rejection-reviews.json)
- [Raw Wikipedia traceback](artifacts/v4-policy-validation-20260917/wiki-raw-traceback.json)
- [89 adversarial checks](artifacts/v4-policy-validation-20260917/contract-checks.json)
- [Sampling, counts and hashes](artifacts/v4-policy-validation-20260917/profile.json)
- [Executable accounting notebook](artifacts/v4-policy-validation-20260917/analysis.ipynb)

Artifacts and frozen source copies are local and ignored by Git. Reproduction
scripts are saved beside them (`audit.py`, `review.py`, `probes.py`,
`scoped_impact.py`, `build_report.py`). The audit only scanned about 195 MB of saved
JSONL source samples, plus three compressed XML blocks for tracing; it did not
rescan the full downloaded web corpus.
