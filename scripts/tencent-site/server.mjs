import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export async function createSiteHandler(directory) {
  const manifestBytes = await readFile(join(directory, 'site-manifest.json'));
  const manifest = JSON.parse(manifestBytes);
  const release = createHash('sha256').update(manifestBytes).digest('hex');
  const files = new Map();
  for (const file of manifest.files) {
    if (!/^[\w./-]+$/.test(file.path) || file.path.split('/').some(part => !part || part.startsWith('.'))) {
      throw new Error('Invalid manifest path');
    }
    const body = await readFile(join(directory, 'public', file.path));
    if (createHash('sha256').update(body).digest('hex') !== file.sha256) throw new Error('Site checksum mismatch');
    files.set('/' + file.path, { ...file, body });
  }
  if (!files.has('/index.html') || !files.has('/404.html')) throw new Error('Missing site pages');

  return (request, response) => {
    const headers = { 'X-Content-Type-Options': 'nosniff', 'X-Site-Release': release };
    const empty = (status, extra = {}) => {
      response.writeHead(status, { ...headers, 'Cache-Control': 'no-store', ...extra });
      response.end();
    };
    if (!['GET', 'HEAD'].includes(request.method)) return empty(405, { Allow: 'GET, HEAD' });
    let path;
    const [rawPath, query] = request.url.split(/\?(.*)/s);
    try { path = decodeURIComponent(rawPath); } catch { return empty(400); }
    if (!path.startsWith('/') || /[\\\x00-\x1f\x7f]/.test(path)
      || path.split('/').some(part => part.startsWith('.'))) return empty(400);
    // Resolve only entries in the release manifest; never expose function code or list directories.
    if (!path.endsWith('/') && files.has(path + '/index.html')) {
      const location = path.split('/').map(encodeURIComponent).join('/') + '/' + (query ? '?' + query : '');
      return empty(308, { Location: location });
    }
    const found = files.get(path.endsWith('/') ? path + 'index.html' : path);
    const file = found || files.get('/404.html');
    const etag = '"' + file.sha256 + '"';
    Object.assign(headers, {
      'Content-Type': file.contentType,
      'Cache-Control': found ? file.cacheControl : 'no-cache',
      ETag: etag,
    });
    const matches = String(request.headers['if-none-match'] || '').split(',')
      .map(value => value.trim().replace(/^W\//, ''));
    if (found && (matches.includes('*') || matches.includes(etag))) {
      response.writeHead(304, headers);
      return response.end();
    }
    response.writeHead(found ? 200 : 404, { ...headers, 'Content-Length': file.body.length });
    response.end(request.method === 'HEAD' ? undefined : file.body);
  };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const handler = await createSiteHandler(dirname(fileURLToPath(import.meta.url)));
  createServer(handler).listen(9000, '0.0.0.0');
}
