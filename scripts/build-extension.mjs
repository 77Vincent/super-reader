import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { chmod, copyFile, lstat, mkdir, mkdtemp, readFile, rename, rm, utimes, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Script } from 'node:vm';

const root = fileURLToPath(new URL('../', import.meta.url));
const dist = join(root, 'dist');
// Explicit inputs: never walk the repository or include training data, site,
// dependencies, credentials, tests or historical build output in a release.
const files = [
  'manifest.json',
  'icons/on-16.png', 'icons/on-32.png', 'icons/on-48.png', 'icons/on-128.png',
  'icons/off-16.png', 'icons/off-32.png',
  'src/app/reader.js',
  'src/backend/chunker.js', 'src/backend/inference.js', 'src/boundary-model-data.js',
  'src/content.css',
  'src/frontend/content-changes.js', 'src/frontend/dom-tree.js',
  'src/frontend/processed-text.js', 'src/frontend/read.js',
  'src/frontend/viewport.js', 'src/frontend/visibility.js', 'src/frontend/write.js',
  'src/inference.html', 'src/inference-service.js', 'src/inference-worker.js',
  'src/platform/chrome/background.js', 'src/platform/chrome/content.js',
  'src/platform/chrome/inference.js',
  'src/start-inference.js', 'src/start-reader.js',
].sort();
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');
const manifest = JSON.parse(await readFile(join(root, 'manifest.json'), 'utf8'));
if (manifest.manifest_version !== 3 || !/^\d+(?:\.\d+){0,3}$/.test(manifest.version)) {
  throw new Error('Expected a Manifest V3 extension with a numeric version.');
}
const metadata = JSON.parse(await readFile(join(root, 'package.json'), 'utf8'));
if (metadata.version !== manifest.version) throw new Error('package.json and manifest.json versions differ.');
const references = [manifest.background?.service_worker,
  ...Object.values(manifest.icons || {}), ...Object.values(manifest.action?.default_icon || {}),
  ...(manifest.web_accessible_resources || []).flatMap(({ resources }) => resources),
];
for (const path of references) {
  if (!files.includes(path)) throw new Error(`Manifest resource is missing from the package: ${path}`);
}

await mkdir(dist, { recursive: true });
const temporary = await mkdtemp(join(dist, '.extension-'));
try {
  const unpacked = join(temporary, 'extension');
  const entries = [];
  for (const path of files) {
    const source = join(root, path);
    // Reject symlinks as well as special files instead of following them.
    if (!(await lstat(source)).isFile()) throw new Error(`Not a regular release file: ${path}`);
    const bytes = await readFile(source);
    if (path.endsWith('.js')) new Script(bytes.toString('utf8'), { filename: path });
    const target = join(unpacked, path);
    await mkdir(dirname(target), { recursive: true });
    await copyFile(source, target);
    await chmod(target, 0o644);
    // Stable ZIP timestamps; local time avoids timezone-dependent ZIP dates.
    const stamp = new Date(2000, 0, 1);
    await utimes(target, stamp, stamp);
    entries.push({ path, bytes: bytes.length, sha256: sha256(bytes) });
  }
  const archiveName = `haodu-${manifest.version}.zip`;
  const archive = join(temporary, archiveName);
  // zip is provided by macOS and the Ubuntu release runner. No recursive input.
  execFileSync('zip', ['-X', '-q', '-9', archive, ...files], { cwd: unpacked });
  const archiveBytes = await readFile(archive);
  const report = {
    name: manifest.name, version: manifest.version,
    unpackedBytes: entries.reduce((sum, entry) => sum + entry.bytes, 0),
    archive: archiveName, archiveBytes: archiveBytes.length,
    archiveSha256: sha256(archiveBytes), files: entries,
  };
  await writeFile(join(temporary, 'extension-release.json'), JSON.stringify(report, null, 2) + '\n');
  await writeFile(join(temporary, `${archiveName}.sha256`), `${report.archiveSha256}  ${archiveName}\n`);
  // Only replace owned artifacts after a successful build. Keep older versions.
  await rm(join(dist, 'extension'), { recursive: true, force: true });
  for (const name of ['extension', archiveName, `${archiveName}.sha256`, 'extension-release.json']) {
    await rename(join(temporary, name), join(dist, name));
  }
  console.log(`Extension: dist/extension/ (${entries.length} files, ${(report.unpackedBytes / 1e6).toFixed(2)} MB)`);
  console.log(`Store ZIP: dist/${archiveName} (${(archiveBytes.length / 1e6).toFixed(2)} MB)`);
  console.log(`SHA-256: ${report.archiveSha256}`);
} finally {
  await rm(temporary, { recursive: true, force: true });
}
