import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';

// SCF repackages uploaded ZIPs. Compare every file, including the runtime,
// rather than requiring the container bytes to remain identical.
export function verifyArchiveFiles(expected, actual) {
  const list = path => {
    const entries = execFileSync('unzip', ['-Z1', path], { encoding: 'utf8', maxBuffer: 65536 }).trim().split('\n');
    if (new Set(entries).size !== entries.length || entries.some(name =>
      !/^[\w./-]+$/.test(name) || name.startsWith('/') || name.split('/').some(part => part.startsWith('.')))) {
      throw new Error('Invalid cloud archive entries');
    }
    return entries.filter(name => !name.endsWith('/')).sort();
  };
  const names = list(expected);
  if (JSON.stringify(names) !== JSON.stringify(list(actual))) throw new Error('Cloud archive file list differs');
  for (const name of names) {
    const local = execFileSync('unzip', ['-p', expected, name], { maxBuffer: 16 * 1024 * 1024 });
    const cloud = execFileSync('unzip', ['-p', actual, name], { maxBuffer: Math.max(1024, local.length + 1) });
    if (!local.equals(cloud)) throw new Error(`Cloud archive content differs: ${name}`);
  }
  const mode = execFileSync('unzip', ['-Z', '-l', actual, 'scf_bootstrap'], { encoding: 'utf8' });
  if (!/^-rwx/.test(mode)) throw new Error('Cloud bootstrap is not executable');
}

export async function verifyRepackedCode(code, expected, temporary) {
  // The SDK supplies this short-lived COS URL. Never print it or pass it to a shell.
  const url = new URL(code.Url);
  if (url.protocol !== 'https:' || !url.hostname.endsWith('.myqcloud.com')) {
    throw new Error('Unexpected cloud code download origin');
  }
  let bytes;
  try {
    const response = await fetch(url, { redirect: 'error', signal: AbortSignal.timeout(30000) });
    if (!response.ok || !response.body) throw new Error();
    const parts = [];
    let size = 0;
    for await (const chunk of response.body) {
      size += chunk.length;
      if (size > 16 * 1024 * 1024) throw new Error();
      parts.push(chunk);
    }
    bytes = Buffer.concat(parts);
  } catch { throw new Error('Cloud code download failed; signed URL omitted'); }
  const digest = createHash('sha256').update(bytes).digest('hex');
  if (code.CodeSha256 !== digest && code.CodeSha256 !== Buffer.from(digest, 'hex').toString('base64')) {
    throw new Error('Cloud code download checksum differs');
  }
  const archive = join(temporary, 'cloud.zip');
  await writeFile(archive, bytes, { mode: 0o600 });
  verifyArchiveFiles(expected, archive);
}
