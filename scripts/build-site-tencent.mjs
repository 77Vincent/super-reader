import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { chmod, copyFile, lstat, mkdir, mkdtemp, readFile, readdir, rename, rm, utimes, writeFile } from 'node:fs/promises';
import { dirname, extname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const types = {
  '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8', '.json': 'application/json',
  '.xml': 'application/xml; charset=utf-8', '.txt': 'text/plain; charset=utf-8',
  '.svg': 'image/svg+xml', '.png': 'image/png', '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg', '.webp': 'image/webp', '.ico': 'image/x-icon',
  '.woff': 'font/woff', '.woff2': 'font/woff2',
};
const root = fileURLToPath(new URL('../', import.meta.url));
execFileSync(process.execPath, [join(root, 'scripts/build-site.mjs'), '--base-url', 'https://haodu.site/'], { cwd: root, stdio: 'inherit' });

const source = join(root, 'dist/site');
async function collect(directory, prefix = '') {
  if (!(await lstat(directory)).isDirectory()) throw new Error('Invalid static release directory');
  const files = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const path = prefix + entry.name;
    if (entry.name.startsWith('.') || entry.isSymbolicLink() || !/^[\w.-]+$/.test(entry.name)) {
      throw new Error(`Unexpected release entry: ${path}`);
    }
    if (entry.isDirectory()) files.push(...await collect(join(directory, entry.name), path + '/'));
    else if (entry.isFile() && types[extname(path)]) files.push(path);
    else throw new Error(`Unsupported release file: ${path}`);
  }
  return files.sort();
}

const files = await collect(source);
const temporary = await mkdtemp(join(root, 'dist/.tencent-site-'));
try {
  const unpacked = join(temporary, 'function');
  const entries = [];
  const stamp = new Date(2000, 0, 1);
  for (const path of files) {
    const bytes = await readFile(join(source, path));
    const target = join(unpacked, 'public', path);
    await mkdir(dirname(target), { recursive: true });
    await writeFile(target, bytes, { mode: 0o644 });
    await utimes(target, stamp, stamp);
    entries.push({ path, bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex'),
      contentType: types[extname(path)],
      cacheControl: path.endsWith('.html') ? 'no-cache'
        : /\.[a-f0-9]{16,}\.(css|js)$/.test(path) ? 'public, max-age=31536000, immutable'
          : 'public, max-age=3600',
    });
  }
  const manifest = JSON.stringify({ domain: 'haodu.site', files: entries }, null, 2) + '\n';
  await writeFile(join(unpacked, 'site-manifest.json'), manifest);
  await copyFile(join(root, 'scripts/tencent-site/server.mjs'), join(unpacked, 'server.mjs'));
  await writeFile(join(unpacked, 'scf_bootstrap'), '#!/bin/bash\nexec /var/lang/node20/bin/node /var/user/server.mjs\n');
  for (const name of ['site-manifest.json', 'server.mjs', 'scf_bootstrap']) {
    await chmod(join(unpacked, name), name === 'scf_bootstrap' ? 0o755 : 0o644);
    await utimes(join(unpacked, name), stamp, stamp);
  }
  const archive = join(temporary, 'haodu-site-scf.zip');
  execFileSync('zip', ['-X', '-q', '-9', archive, 'scf_bootstrap', 'server.mjs', 'site-manifest.json', ...files.map(path => 'public/' + path)], { cwd: unpacked });
  const bytes = await readFile(archive);
  if (bytes.length > 15 * 1024 * 1024) throw new Error('Function archive exceeds the inline API deployment limit');
  const report = { functionName: 'haodu-site', region: 'ap-shanghai', namespace: 'default', runtime: 'Nodejs20.19',
    archive: 'haodu-site-scf.zip', archiveBytes: bytes.length,
    archiveSha256: createHash('sha256').update(bytes).digest('hex'),
    release: createHash('sha256').update(manifest).digest('hex'), files: entries,
  };
  await rm(join(root, 'dist/tencent-site'), { recursive: true, force: true });
  await rename(unpacked, join(root, 'dist/tencent-site'));
  await rename(archive, join(root, 'dist/haodu-site-scf.zip'));
  await writeFile(join(root, 'dist/tencent-site-release.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(`SCF package: dist/haodu-site-scf.zip (${entries.length} static files; ${bytes.length} bytes)`);
  console.log(`Release: ${report.release}`);
} finally { await rm(temporary, { recursive: true, force: true }); }
