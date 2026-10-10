import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import chunker from '../src/backend/chunker.js';

const homepage = fileURLToPath(new URL('../site/content/_index.md', import.meta.url));

export function markReaderText(markedText) {
  const text = markedText.replaceAll('｜', '');
  // Use the extension entry point, including its default thresholds, word
  // protection and recursive inference. Offsets refer to the original UTF-16.
  const [offsets] = chunker.process([text]);
  let start = 0;
  return [...offsets, text.length].map((end) => {
    const part = text.slice(start, end);
    start = end;
    return part;
  }).join('｜');
}

export function syncReaderCopy(source) {
  // The homepage uses single-quoted TOML copy and plain-text demo paragraphs.
  // Fail on rich text rather than infer on markup that visitors won't see.
  const mark = (text) => {
    if (/[<>\[\]*_`\\\r\n]|&(?:#\w+|\w+);/u.test(text)) {
      throw new Error('Reading preview requires plain text, one line per paragraph.');
    }
    return markReaderText(text);
  };
  const frontmatter = source.match(/^\+\+\+\r?\n([\s\S]*?)\r?\n\+\+\+/u);
  if (!frontmatter) throw new Error('Missing homepage TOML front matter.');
  let header = frontmatter[0];
  // The headline's divider is a fixed brand treatment, not a model prediction.
  let leads = 0;
  header = header.replace(/^(lead[ \t]*=[ \t]*\[\r?\n)([\s\S]*?)(^\])/mu,
    (_, before, lines, after) => before + lines.split('\n').map((line) => {
      if (!line.trim()) return line;
      const match = line.match(/^([ \t]*')([^']*)('[ \t]*,?[ \t]*\r?)$/u);
      if (!match) throw new Error('Expected single-quoted homepage lead lines.');
      leads++;
      return match[1] + mark(match[2]) + match[3];
    }).join('\n') + after);
  if (!leads) throw new Error('Missing homepage lead copy.');

  let demos = 0;
  const body = source.slice(frontmatter[0].length);
  const rendered = body.replace(/(\{\{< reader-demo(?:[ \t]+[^\r\n]*?)? >\}\}\r?\n)([\s\S]*?)(\r?\n\{\{< \/reader-demo >\}\})/gu,
    (_, before, text, after) => {
      demos++;
      // Blank lines separate paragraphs in Hugo; infer on each independently.
      return before + text.split(/(\r?\n[ \t]*\r?\n)/u)
        .map((part, index) => index % 2 ? part : mark(part)).join('') + after;
    });
  if (!demos || demos !== [...body.matchAll(/\{\{< reader-demo\b/gu)].length) {
    throw new Error('Unrecognized homepage reading demo format.');
  }
  return header + rendered;
}

async function main() {
  const { values } = parseArgs({ options: { check: { type: 'boolean', default: false } } });
  const source = await readFile(homepage, 'utf8');
  const updated = syncReaderCopy(source);
  if (source === updated) {
    console.log('Site reading cuts match the current extension backend.');
  } else if (values.check) {
    throw new Error('Site reading cuts are stale. Run npm run site:sync-reading.');
  } else {
    await writeFile(homepage, updated);
    console.log('Updated site reading cuts from the current extension backend.');
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
