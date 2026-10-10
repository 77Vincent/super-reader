const test = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const chunker = require('../src/backend/chunker.js');
const generator = import('../scripts/sync-site-reading.mjs');

function readMarkers(markedText) {
  const parts = markedText.split('｜');
  let offset = 0;
  const cuts = parts.slice(0, -1).map((part) => { offset += part.length; return offset; });
  return { text: parts.join(''), cuts };
}

test('site markers match extension decisions and preserve original text and UTF-16 offsets', async () => {
  const { markReaderText } = await generator;
  for (const input of [
    '让中文｜更好读',
    '的确有许多工作｜是极端累人的。',
    '𠮷🌈的确有许多工作｜是极端累人的。',
    '所有那些｜畏惧并赞美软弱的人。',
    '使用离线运行的｜微型神经网络，极致智能、安全、高效｜且无额外网络开销。',
    '示例：字体、颜色、布局。',
  ]) {
    const text = input.replaceAll('｜', '');
    const output = markReaderText(input);
    assert.deepEqual(readMarkers(output), { text, cuts: chunker.process([text])[0] });
    assert.equal(markReaderText(output), output);
  }
  assert.equal(markReaderText('让中文｜更好读'), '让中文更好读');
});

test('homepage sync preserves the fixed headline and updates every lead and demo paragraph', async () => {
  const { syncReaderCopy, markReaderText } = await generator;
  const text = '的确有许多工作｜是极端累人的。';
  const source = [
    '+++', "title = '保留｜标题元数据'", "headline = '让中文｜更好读'  ",
    'lead = [', `  '${text}',`, "  '短｜句',", ']', '+++', '',
    '{{< reader-demo source="原始｜署名" >}}', text, '', '短｜句', '{{< /reader-demo >}}', '',
    '{{< reader-demo >}}', text, '{{< /reader-demo >}}', '',
  ].join('\n');
  for (const input of [source, source.replaceAll('\n', '\r\n')]) {
    const output = syncReaderCopy(input);
    assert.equal(output.replaceAll('｜', ''), input.replaceAll('｜', ''));
    assert.equal(output.split(markReaderText(text)).length - 1, 3);
    assert.ok(output.includes("title = '保留｜标题元数据'"));
    assert.ok(output.includes("headline = '让中文｜更好读'  "));
    assert.ok(output.includes('source="原始｜署名"'));
    assert.ok(!output.includes('短｜句'));
    assert.equal(syncReaderCopy(output), output);
  }
  assert.throws(() => syncReaderCopy(source.replace(text, '**粗体**')), /plain text/);
  assert.throws(() => syncReaderCopy(source.replaceAll('{{< /reader-demo >}}', '')), /demo format/);
});

test('checked-in homepage cuts match the current backend', async () => {
  const { syncReaderCopy } = await generator;
  const source = readFileSync(join(__dirname, '../site/content/_index.md'), 'utf8');
  assert.equal(syncReaderCopy(source), source, 'Run npm run site:sync-reading after backend or copy changes.');
});
