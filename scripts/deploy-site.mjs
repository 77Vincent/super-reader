import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { lstat, readdir, readFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { extname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { createHash } from 'node:crypto';

const { values } = parseArgs({ options: {
  'dry-run': { type: 'boolean', default: false },
  profile: { type: 'string' },
  ossutil: { type: 'string' },
} });
const root = fileURLToPath(new URL('../', import.meta.url));
const destination = join(root, 'dist/site');
const bucket = 'haodu-site';
const localOssutil = join(homedir(), '.local/bin/ossutil');
const ossutil = values.ossutil || (existsSync(localOssutil) ? localOssutil : 'ossutil');
const common = ['--region', 'cn-shanghai', '--endpoint', 'https://oss-cn-shanghai.aliyuncs.com',
  '--retry-times', '2', '--loglevel', 'off'];
if (values.profile) common.push('--profile', values.profile);

function run(command, args) {
  const result = spawnSync(command, args, { cwd: root, stdio: 'inherit' });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${command} failed (${result.status ?? result.signal}); deployment stopped.`);
}

const types = {
  '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8', '.json': 'application/json',
  '.xml': 'application/xml; charset=utf-8', '.txt': 'text/plain; charset=utf-8',
  '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg', '.webp': 'image/webp', '.ico': 'image/x-icon',
  '.woff': 'font/woff', '.woff2': 'font/woff2',
};

async function collect(directory, prefix = '') {
  if (!(await lstat(directory)).isDirectory()) throw new Error(`Not a directory: ${directory}`);
  const files = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const key = prefix + entry.name;
    if (entry.name.startsWith('.') || entry.isSymbolicLink()) throw new Error(`Unexpected release entry: ${key}`);
    if (entry.isDirectory()) files.push(...await collect(join(directory, entry.name), `${key}/`));
    else if (entry.isFile() && types[extname(key)]) files.push(key);
    else throw new Error(`Unsupported release file: ${key}`);
  }
  return files;
}

// Rebuild the production artifact so previews or unrelated dist files cannot be uploaded.
run(process.execPath, [join(root, 'scripts/build-site.mjs'), '--base-url', 'https://haodu.site/']);
const files = await collect(destination);
const priority = key => key === 'index.html' ? 2 : key.endsWith('.html') ? 1 : 0;
files.sort((a, b) => priority(a) - priority(b) || a.localeCompare(b));
const plan = await Promise.all(files.map(async key => {
  const content = await readFile(join(destination, key));
  return {
    key, size: content.length,
    etag: createHash('md5').update(content).digest('hex').toUpperCase(),
    contentType: types[extname(key)],
    cache: key.endsWith('.html') ? 'no-cache'
      : /\.[a-f0-9]{16,}\.(css|js)$/.test(key) ? 'public, max-age=31536000, immutable'
        : 'public, max-age=3600',
  };
}));
console.log(`${values['dry-run'] ? 'Preview' : 'Deploy'}: ${plan.length} files, ${plan.reduce((n, f) => n + f.size, 0)} bytes → oss://${bucket}/`);
for (const file of plan) {
  console.log(`${file.key} (${file.size} bytes; ${file.contentType}; ${file.cache})`);
  if (values['dry-run']) continue;
  run(ossutil, ['api', 'put-object', ...common, '--bucket', bucket, '--key', file.key,
    '--body', `file://${join(destination, file.key)}`, '--content-type', file.contentType,
    '--cache-control', file.cache, '--quiet']);
  // Each file uses a single PutObject, whose ETag is the MD5 of its bytes.
  // Verify every upload before publishing the next page, with the homepage last.
  run(ossutil, ['api', 'head-object', ...common, '--bucket', bucket, '--key', file.key,
    '--if-match', `"${file.etag}"`, '--quiet']);
}
console.log(values['dry-run']
  ? 'Preview complete; no OSS API requests were made.'
  : 'Upload and ETag verification complete. Domain, HTTPS and public-access settings are managed separately.');
