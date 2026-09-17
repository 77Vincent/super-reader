const test = require("node:test");
const assert = require("node:assert/strict");

test("training pairs use adjacent punctuation fragments with one target gap", async () => {
  const { buildAdjacentSamples } = await import("../training/prepare_smoke_data.mjs");
  const samples = buildAdjacentSamples({
    id: "fixture:1",
    domain: "fixture",
    text: "今天下雨，我们留在家里；明天再出去。",
  });

  assert.equal(samples.length, 2);
  assert.equal(samples[0].tokens.join(""), "今天下雨我们留在家里");
  assert.equal(samples[0].target_index, 3);
  assert.equal(samples[0].tokens[samples[0].target_index], "雨");
  assert.equal(samples[0].tokens[samples[0].target_index + 1], "我");
  assert.equal(samples[1].document_id, samples[0].document_id);
});

test("colons and ASCII dots remain context while Chinese ellipses create boundaries", async () => {
  const { splitIntoFragments, buildAdjacentSamples } = await import(
    "../training/prepare_smoke_data.mjs"
  );
  const text = "先说明：这里需要停顿……然后继续...最后结束。";

  assert.deepEqual(splitIntoFragments(text), [
    { text: "先说明:这里需要停顿", punctuation: "……" },
    { text: "然后继续...最后结束", punctuation: "。" },
  ]);

  const samples = buildAdjacentSamples({
    id: "fixture:colon-ellipsis",
    domain: "fixture",
    text,
  });
  assert.deepEqual(
    samples.map((sample) => sample.punctuation),
    ["……"],
  );
});

test("enumeration commas keep list items together without creating training targets", async () => {
  const { buildAdjacentSamples } = await import("../training/prepare_smoke_data.mjs");
  const document = { id: "fixture:enumeration", domain: "fixture" };

  for (const tokenization of ["character", "word"]) {
    assert.deepEqual(buildAdjacentSamples({
      ...document,
      text: "我买了苹果、香蕉、橙子。",
    }, { tokenization }), []);

    const samples = buildAdjacentSamples({
      ...document,
      text: "我买了苹果、香蕉，准备制作果汁、沙拉。",
    }, { tokenization });
    assert.equal(samples.length, 1);
    const [sample] = samples;
    assert.equal(sample.punctuation, "，");
    assert.equal(sample.tokens.slice(0, sample.target_index + 1).join(""), "我买了苹果、香蕉");
    assert.equal(sample.tokens.slice(sample.target_index + 1).join(""), "准备制作果汁、沙拉");
  }
});

test("regenerated validation and test use the revised proxies and retain document ownership", async () => {
  const { samplesForReferenceDocument } = await import("../training/prepare_retraining_base.mjs");
  const owners = new Map([["train-doc", "train"], ["validation-doc", "validation"], ["test-doc", "test"]]);
  for (const [id, split] of owners) {
    const result = samplesForReferenceDocument({ id, domain: "fixture", text: "我买了苹果、香蕉，准备做果汁。" }, owners);
    assert.equal(result.split, split);
    assert.equal(result.samples.length, 1);
    const [sample] = result.samples;
    assert.equal(sample.punctuation, "，");
    assert.equal(sample.tokens.slice(0, sample.target_index + 1).join(""), "我买了苹果、香蕉");
    assert.equal(sample.tokens.slice(sample.target_index + 1).join(""), "准备做果汁");
  }
  const empty = samplesForReferenceDocument({ id: "test-doc", domain: "fixture", text: "苹果、香蕉。" }, owners);
  assert.equal(empty.split, "test");
  assert.deepEqual(empty.samples, []);
  assert.equal(owners.get("test-doc"), "test");
  assert.equal(samplesForReferenceDocument({ id: "unknown", text: "甲，乙。" }, owners), null);
});

test("corrected holdouts exclude known training pairs and reject enumeration labels", () => {
  const { mkdtempSync, mkdirSync, writeFileSync, readFileSync, statSync, rmSync } = require("node:fs");
  const { tmpdir } = require("node:os");
  const { join } = require("node:path");
  const { execFileSync, spawnSync } = require("node:child_process");
  const directory = mkdtempSync(join(tmpdir(), "reader-holdout-test-"));
  const record = (id, text, punctuation = "，") => ({ id, document_id: id, domain: "fixture", tokens: Array.from(text), target_index: 1, punctuation });
  try {
    const input = join(directory, "input");
    const output = join(directory, "output");
    mkdirSync(input);
    const seen = record("seen", "已有样本");
    const clean = record("clean", "全新边界");
    const heldout = record("test", "独立测试");
    writeFileSync(join(input, "validation.jsonl"), [seen, clean].map(JSON.stringify).join("\n") + "\n");
    writeFileSync(join(input, "test.jsonl"), JSON.stringify(heldout) + "\n");
    const shard = join(directory, "train.jsonl");
    writeFileSync(shard, JSON.stringify(["已有样本", 1, 0, 0, 5]) + "\n");
    const manifest = join(directory, "manifest.json");
    writeFileSync(manifest, JSON.stringify({ shards: [{ path: shard, bytes: statSync(shard).size }] }));
    const args = ["training/filter_retraining_holdouts.py", "--data-dir", input, "--output-dir", output, "--training-manifest", manifest];
    execFileSync("python3", args);
    assert.deepEqual(JSON.parse(readFileSync(join(output, "validation.jsonl"), "utf8")), clean);
    assert.deepEqual(JSON.parse(readFileSync(join(output, "test.jsonl"), "utf8")), heldout);
    const summary = JSON.parse(readFileSync(join(output, "summary.json"), "utf8"));
    assert.equal(summary.splits.validation.removed_training_overlap.fixture, 1);
    writeFileSync(join(input, "validation.jsonl"), JSON.stringify(record("bad", "苹果香蕉", "、")) + "\n");
    args[4] = join(directory, "invalid-output");
    const rejected = spawnSync("python3", args, { encoding: "utf8" });
    assert.notEqual(rejected.status, 0);
    assert.match(rejected.stderr, /Invalid proxy label remains/u);
  } finally { rmSync(directory, { recursive: true, force: true }); }
});

test("Wikipedia XML keeps article text and removes wiki markup", async () => {
  const { wikipediaDocumentsFromXml } = await import(
    "../training/prepare_smoke_data.mjs"
  );
  const xml = `<mediawiki>
    <page>
      <title>阅读</title><ns>0</ns><id>42</id>
      <revision><id>99</id><text xml:space="preserve">'''阅读'''是[[语言|文字]]活动。{{来源请求}}它帮助理解。&lt;ref&gt;引用&lt;/ref&gt;</text></revision>
    </page>
    <page>
      <title>模板</title><ns>10</ns><id>43</id>
      <revision><text>不应保留。</text></revision>
    </page>
    <page>
      <title>跳转</title><ns>0</ns><id>44</id><redirect title="阅读" />
      <revision><text>#REDIRECT [[阅读]]</text></revision>
    </page>
  </mediawiki>`;

  assert.deepEqual(wikipediaDocumentsFromXml(xml, 123), [{
    id: "wikipedia:42",
    domain: "wikipedia",
    title: "阅读",
    text: "阅读是文字活动。它帮助理解。",
    stream_offset: 123,
  }]);
});

test("training uses the latest Chinese Wikipedia current-article dump", () => {
  const { readFileSync } = require("node:fs");
  const preparation = readFileSync("training/prepare_smoke_data.mjs", "utf8");

  assert.match(
    preparation,
    /zhwiki\/latest\/zhwiki-latest-pages-articles-multistream\.xml\.bz2/u,
  );
  assert.match(preparation, /wikipediaDocs:\s*250000/u);
  assert.doesNotMatch(preparation, /zhwiki-latest-pages-meta-history/u);
});

test("training input hides the target proxy and retains quotation context", async () => {
  const { buildAdjacentSamples } = await import("../training/prepare_smoke_data.mjs");
  const [sample] = buildAdjacentSamples({
    id: "fixture:2",
    domain: "fixture",
    text: "模型只看文字，“标点”不会泄露。",
  });

  assert.equal(sample.punctuation, "，");
  assert.equal(sample.tokens.join(""), "模型只看文字“标点”不会泄露");
});

test("training sides have no character-length ceiling", async () => {
  const { buildAdjacentSamples } = await import("../training/prepare_smoke_data.mjs");
  const [sample] = buildAdjacentSamples({
    id: "fixture:no-limit",
    domain: "fixture",
    text: `${"甲".repeat(64)}，好。`,
  });

  assert.equal(sample.left_character_length, 64);
  assert.equal(sample.right_character_length, 1);
  assert.equal(sample.tokens.join(""), `${"甲".repeat(64)}好`);
});

test("one-character fragments remain valid training sides", async () => {
  const { buildAdjacentSamples } = await import("../training/prepare_smoke_data.mjs");
  const [sample] = buildAdjacentSamples({
    id: "fixture:minimum",
    domain: "fixture",
    text: "好，走。",
  });

  assert.deepEqual(sample.tokens, ["好", "走"]);
  assert.equal(sample.target_index, 0);
  assert.equal(sample.left_character_length, 1);
  assert.equal(sample.right_character_length, 1);
});

test("sample boundaries are unique within and across data splits", async () => {
  const { deduplicateSampleBoundaries } = await import(
    "../training/prepare_smoke_data.mjs"
  );
  const sample = (id, text, targetIndex, domain = "fixture") => ({
    id,
    domain,
    tokens: Array.from(text),
    target_index: targetIndex,
  });
  const result = deduplicateSampleBoundaries({
    train: [
      sample("train:shared-test", "共同边界", 1),
      sample("train:shared-validation", "验证边界", 1),
      sample("train:unique", "训练独有", 1),
      sample("train:duplicate", "训练独有", 1),
    ],
    validation: [sample("validation:shared", "验证边界", 1)],
    test: [sample("test:shared", "共同边界", 1)],
  }, 42);

  assert.deepEqual(
    result.samplesBySplit.test.map(({ id }) => id),
    ["test:shared"],
  );
  assert.deepEqual(
    result.samplesBySplit.validation.map(({ id }) => id),
    ["validation:shared"],
  );
  assert.equal(result.samplesBySplit.train.length, 1);
  assert.match(result.samplesBySplit.train[0].id, /^train:(?:unique|duplicate)$/u);
  assert.deepEqual(result.removed, {
    test: 0,
    validation: 0,
    train: 3,
    total: 3,
  });
});

test("training selection balances relative boundary-position buckets", async () => {
  const { selectPositionBalancedSamples } = await import(
    "../training/prepare_smoke_data.mjs"
  );
  const candidates = Array.from({ length: 10 }, (_, bin) => (
    Array.from({ length: 4 }, (_, index) => ({
      id: `fixture:bin:${bin}:${index}`,
      relative_boundary_position: (bin + 0.5) / 10,
    }))
  )).flat();
  const selected = selectPositionBalancedSamples(candidates, 30, "fixture", 10);
  const histogram = Array(10).fill(0);
  for (const sample of selected) {
    const bin = Math.min(9, Math.floor(sample.relative_boundary_position * 10));
    histogram[bin] += 1;
  }

  assert.deepEqual(histogram, Array(10).fill(3));
});

test("training weights keep every sample and balance positions within each length bucket", async () => {
  const { assignLengthPositionWeights } = await import(
    "../training/prepare_smoke_data.mjs"
  );
  const makeSamples = (prefix, length, leftLength, count, domain) => (
    Array.from({ length: count }, (_, index) => ({
      id: `${prefix}:${index}`,
      domain,
      tokens: Array(length).fill("字"),
      left_character_length: leftLength,
      right_character_length: length - leftLength,
      relative_boundary_position: leftLength / length,
    }))
  );
  const candidates = [
    ...makeSamples("short-left", 8, 1, 6, "news"),
    ...makeSamples("short-right", 8, 6, 2, "dialogue"),
    ...makeSamples("medium-left", 16, 2, 1, "news"),
    ...makeSamples("medium-right", 16, 12, 3, "dialogue"),
  ];

  const weighted = assignLengthPositionWeights(candidates);
  assert.deepEqual(
    weighted.map((sample) => sample.id),
    candidates.map((sample) => sample.id),
  );
  assert.ok(weighted.every((sample) => sample.training_weight > 0));

  const totals = new Map();
  for (const sample of weighted) {
    const key = `${sample.length_bucket}:${sample.position_bin}`;
    totals.set(key, (totals.get(key) || 0) + sample.training_weight);
  }
  assert.ok(Math.abs(totals.get("<=8:1") - totals.get("<=8:7")) < 1e-9);
  assert.ok(Math.abs(totals.get("<=16:1") - totals.get("<=16:7")) < 1e-9);
  assert.ok(
    Math.abs(
      weighted.reduce((total, sample) => total + sample.training_weight, 0)
        / weighted.length
        - 1,
    ) < 1e-9,
  );
});

test("production data defaults to all samples and weighted training loss", () => {
  const { readFileSync } = require("node:fs");
  const preparation = readFileSync("training/prepare_smoke_data.mjs", "utf8");
  const training = readFileSync("training/train_smoke.py", "utf8");

  assert.match(preparation, /requested > 0/u);
  assert.match(preparation, /: candidates;/u);
  assert.match(preparation, /domain_balancing:\s*false/u);
  assert.match(training, /per_sample_loss \* batch\["sample_weights"\]/u);
  assert.match(training, /batch\["sample_weight_sum"\]/u);
});

test("CPU training uses benchmarked threads and equivalent three-tap matrix products", () => {
  const { readFileSync } = require("node:fs");
  const training = readFileSync("training/train_smoke.py", "utf8");

  assert.match(training, /torch\.set_num_threads\(threads\)/u);
  assert.match(training, /default=min\(5, os\.cpu_count\(\) or 1\)/u);
  assert.match(training, /class ThreeTapConv1d/u);
  assert.match(training, /self\.weight\[:, :, 0\]/u);
  assert.match(training, /self\.weight\[:, :, 1\]/u);
  assert.match(training, /self\.weight\[:, :, 2\]/u);
  assert.match(training, /foreach=True/u);
  assert.match(training, /--channels", type=int, default=64/u);
  assert.match(training, /--residual-blocks/u);
  assert.match(training, /default=4/u);
});

test("training saves atomic resumable state at safe interrupt boundaries", () => {
  const { readFileSync } = require("node:fs");
  const training = readFileSync("training/train_smoke.py", "utf8");
  const documentation = readFileSync("training/README.md", "utf8");

  assert.match(training, /"--resume"/u);
  assert.match(training, /training-state\.pt/u);
  assert.match(training, /"optimizer_state": optimizer\.state_dict\(\)/u);
  assert.match(training, /"next_batch": next_batch/u);
  assert.match(training, /os\.replace\(temporary_path, path\)/u);
  assert.match(training, /signal\.SIGINT/u);
  assert.match(documentation, /--epochs 6 --resume/u);
});

test("full Wikipedia preparation and training stay memory bounded", () => {
  const { readFileSync } = require("node:fs");
  const preparation = readFileSync(
    "training/prepare_full_wikipedia_data.mjs",
    "utf8",
  );
  const training = readFileSync("training/train_sharded.py", "utf8");

  assert.match(preparation, /wiki-full-sharded-128/u);
  assert.match(preparation, /maxSamplesPerDocument:\s*128/u);
  assert.match(preparation, /BLOOM_BYTES = 256 \* 1024 \* 1024/u);
  assert.match(preparation, /preparation-state\.json/u);
  assert.match(preparation, /maxSequenceLength:\s*2048/u);
  assert.match(training, /read_training_shard/u);
  assert.match(training, /--channels", type=int, default=128/u);
  assert.match(training, /"sharded_streaming": True/u);
  assert.match(training, /Training state already exists/u);
});

test("synthetic expansion is bounded, deduplicated, domain weighted, and resumable", () => {
  const { readFileSync } = require("node:fs");
  const preparation = readFileSync("training/prepare_synthetic_data.py", "utf8");
  const training = readFileSync("training/train_sharded.py", "utf8");

  assert.match(preparation, /Ultra-FineWeb-L3-zh-Multi-Style-Synthetic/u);
  assert.match(preparation, /target-samples.*5_000_000/u);
  assert.match(preparation, /BLOOM_BYTES = 256 \* 1024 \* 1024/u);
  assert.match(preparation, /preparation-state\.json/u);
  assert.match(preparation, /iter_batches/u);
  assert.match(training, /--domain-weight-power/u);
  assert.match(training, /counts\[domain\] \*\* -power/u);
  assert.match(training, /loss = weighted\.mean\(\) \/ training_weight_mean/u);
  assert.doesNotMatch(training, /loss = weighted\.sum\(\) \/ batch\["sample_weight_sum"\]/u);
  assert.match(training, /clip_grad_norm_/u);
  assert.match(training, /validation_macro_accuracy/u);
});

test("comparison modes keep the same text and gold boundary but change candidates", async () => {
  const { buildAdjacentSamples } = await import("../training/prepare_smoke_data.mjs");
  const document = {
    id: "fixture:comparison",
    domain: "fixture",
    text: "阅读不是更快地扫过文字，模型开始工作。",
  };
  const [word] = buildAdjacentSamples(document, { tokenization: "word" });
  const [character] = buildAdjacentSamples(document, { tokenization: "character" });

  assert.equal(word.id, character.id);
  assert.equal(word.tokens.join(""), character.tokens.join(""));
  assert.equal(
    word.tokens.slice(0, word.target_index + 1).join(""),
    character.tokens.slice(0, character.target_index + 1).join(""),
  );
  assert.ok(word.tokens.some((token) => token.length > 1));
  assert.ok(character.tokens.every((token) => Array.from(token).length === 1));
  assert.ok(character.tokens.length > word.tokens.length);
});

test("both comparison modes retain non-Chinese context", async () => {
  const { buildAdjacentSamples } = await import("../training/prepare_smoke_data.mjs");
  const document = {
    id: "fixture:han-only",
    domain: "fixture",
    text: "Synthetic 中文 20，AI 模型。",
  };

  for (const tokenization of ["word", "character"]) {
    const [sample] = buildAdjacentSamples(document, { tokenization });
    assert.equal(sample.tokens.join(""), "Synthetic 中文 20AI 模型");
  }
});
