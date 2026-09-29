import { execFileSync } from 'node:child_process';
import { readFile, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const { values } = parseArgs({ options: {
  'base-url': { type: 'string' }, preview: { type: 'boolean', default: false },
} });
const input = values['base-url'] || process.env.SITE_BASE_URL;
if (!values.preview && !input) throw new Error('Supply --base-url https://your-domain/ or SITE_BASE_URL for a release build.');
if (values.preview && values['base-url']) throw new Error('Choose --preview or --base-url, not both.');
const url = values.preview ? null : new URL(input);
if (url && (url.protocol !== 'https:' || url.username || url.password || url.search || url.hash
  || /^(localhost|127\..*|\[::1\]|example\.(org|com|net))$/.test(url.hostname)
  || /\.(invalid|test|localhost)$/.test(url.hostname))) {
  throw new Error('Release base URL must be a real HTTPS address without credentials, query or fragment.');
}
if (url && !url.pathname.endsWith('/')) url.pathname += '/';
const root = fileURLToPath(new URL('../', import.meta.url));
const output = values.preview ? 'dist/site-preview' : 'dist/site';
const destination = join(root, output);
await rm(destination, { recursive: true, force: true });
execFileSync('npm', ['run', 'build', '--', '--baseURL', url?.href || '/',
  '--environment', values.preview ? 'preview' : 'production', '--destination', destination], {
  cwd: join(root, 'site'), stdio: 'inherit',
});
for (const page of ['index.html', 'privacy/index.html', '404.html']) {
  const html = await readFile(join(destination, page), 'utf8');
  if (!/<meta\b[^>]*\bname=["']?super-reader["']?\s+content=["']?off["']?[\s/>]/.test(html)) {
    throw new Error(`${page} is missing the extension opt-out flag.`);
  }
  if (/https?:\/\/(?:localhost|example\.org)(?:[:/"\s<])/.test(html)) {
    throw new Error(`${page} contains a development or placeholder URL.`);
  }
}
for (const filename of url ? ['sitemap.xml', 'robots.txt'] : []) {
  const body = await readFile(join(destination, filename), 'utf8');
  if (!body.includes(url.href)) throw new Error(`${filename} does not reference the release URL.`);
}
console.log(values.preview
  ? `Static preview: ${output}/ — noindex; rebuild with a real --base-url before publishing.`
  : `Static release: ${output}/ — ${url.href}`);
