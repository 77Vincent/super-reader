const test = require("node:test");
const assert = require("node:assert/strict");
const { mkdtempSync, writeFileSync, rmSync } = require("node:fs");
const { tmpdir } = require("node:os");
const { join } = require("node:path");
const { execFileSync } = require("node:child_process");

test("Wikipedia rebuild retrieves exact holdout IDs across streams, including empty text", async () => {
  const { referenceWikipediaDocuments, samplesForReferenceDocument } = await import("../training/prepare_retraining_base.mjs");
  const folder = mkdtempSync(join(tmpdir(), "reader-wiki-reference-"));
  const page = (id, text) => `<page><title>页面${id}</title><ns>0</ns><id>${id}</id><revision><text>${text}</text></revision></page>`;
  const compress = (text) => execFileSync("bzip2", ["-c"], { input: text });
  try {
    const first = compress(page(1, "{{模板}}") + page(2, "引句。普通正文，接着说明。"));
    const last = compress(page(3, "引句。独立验证，继续说明。") + page(4, ""));
    const dump = join(folder, "dump.bz2");
    const index = join(folder, "index.bz2");
    writeFileSync(dump, Buffer.concat([first, last]));
    writeFileSync(index, compress(`0:1:页面1\n0:2:页面2\n${first.length}:3:页面3\n${first.length}:4:页面4\n`));
    const downloaded = { primary: { path: dump }, index: { path: index } };
    const owners = new Map([["wikipedia:3", "validation"], ["wikipedia:4", "test"], ["news:1", "train"]]);
    const docs = await referenceWikipediaDocuments(downloaded, owners.keys());
    assert.deepEqual(docs.map((doc) => doc.id), ["wikipedia:3", "wikipedia:4"]);
    assert.equal(docs[0].stream_offset, first.length);
    assert.equal(samplesForReferenceDocument(docs[0], owners).samples.length, 1);
    assert.deepEqual(samplesForReferenceDocument(docs[1], owners), { split: "test", samples: [] });
    await assert.rejects(referenceWikipediaDocuments(downloaded, ["wikipedia:999"]), /missing from index/u);
    writeFileSync(index, compress(`0:1:页面1\n${first.length}:999:缺页\n`));
    await assert.rejects(referenceWikipediaDocuments(downloaded, ["wikipedia:999"]), /were not recovered/u);
  } finally { rmSync(folder, { recursive: true, force: true }); }
});
