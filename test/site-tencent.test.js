const { test } = require('node:test');
const assert = require('node:assert/strict');
const { createServer, request } = require('node:http');
const { createHash } = require('node:crypto');
const { mkdtemp, mkdir, writeFile, rm, copyFile, readFile, utimes } = require('node:fs/promises');
const { tmpdir } = require('node:os');
const { join } = require('node:path');
const { spawnSync } = require('node:child_process');

async function site(t) {
  const directory = await mkdtemp(join(tmpdir(), 'haodu-scf-test-'));
  t.after(() => rm(directory, { recursive: true, force: true }));
  const files = [];
  for (const [path, body, contentType, cacheControl] of [
    ['index.html', '<h1>好读</h1>', 'text/html; charset=utf-8', 'no-cache'],
    ['privacy/index.html', '<p>隐私</p>', 'text/html; charset=utf-8', 'no-cache'],
    ['404.html', '<h1>未找到</h1>', 'text/html; charset=utf-8', 'no-cache'],
    ['favicon.png', Buffer.from([137, 80, 78, 71, 255, 0]), 'image/png', 'public, max-age=3600'],
    ['js/app.0123456789abcdef.js', 'const a=1;', 'text/javascript; charset=utf-8', 'public, max-age=31536000, immutable'],
  ]) {
    await mkdir(join(directory, 'public', path, '..'), { recursive: true });
    await writeFile(join(directory, 'public', path), body);
    files.push({ path, contentType, cacheControl, sha256: createHash('sha256').update(body).digest('hex') });
  }
  await writeFile(join(directory, 'site-manifest.json'), JSON.stringify({ files }));
  // These exist on disk, but are never routable through the manifest.
  await writeFile(join(directory, 'server.mjs'), 'private runtime code');
  await writeFile(join(directory, 'public/unlisted.txt'), 'not in release');
  const { createSiteHandler } = await import('../scripts/tencent-site/server.mjs');
  const server = createServer(await createSiteHandler(directory));
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => new Promise(resolve => server.close(resolve)));
  return {
    directory,
    get: (path, method = 'GET', headers = {}) => new Promise((resolve, reject) => {
      const req = request({ hostname: '127.0.0.1', port: server.address().port, path, method, headers }, res => {
        const chunks = [];
        res.on('data', chunk => chunks.push(chunk));
        res.on('end', () => resolve({ status: res.statusCode, headers: res.headers, body: Buffer.concat(chunks) }));
      });
      req.on('error', reject);
      req.end();
    }),
  };
}

test('SCF serves UTF-8 pages, redirects directories and returns a real 404', async t => {
  const f = await site(t);
  const home = await f.get('/?utm=test');
  assert.equal(home.status, 200);
  assert.equal(home.body.toString(), '<h1>好读</h1>');
  assert.equal(home.headers['content-length'], String(home.body.length));
  assert.equal(home.headers['cache-control'], 'no-cache');
  assert.match(home.headers['x-site-release'], /^[a-f0-9]{64}$/);
  const redirect = await f.get('/privacy?from=home');
  assert.equal(redirect.status, 308);
  assert.equal(redirect.headers.location, '/privacy/?from=home');
  assert.equal((await f.get('/privacy/')).body.toString(), '<p>隐私</p>');
  const missing = await f.get('/unknown/');
  assert.equal(missing.status, 404);
  assert.equal(missing.body.toString(), '<h1>未找到</h1>');
});

test('SCF preserves binary bytes, HEAD and conditional caching', async t => {
  const f = await site(t);
  const png = await f.get('/favicon.png');
  assert.deepEqual(png.body, Buffer.from([137, 80, 78, 71, 255, 0]));
  assert.equal(png.headers['content-type'], 'image/png');
  const head = await f.get('/favicon.png', 'HEAD');
  assert.equal(head.status, 200);
  assert.equal(head.body.length, 0);
  assert.equal(head.headers['content-length'], '6');
  const cached = await f.get('/favicon.png', 'GET', { 'If-None-Match': '"other", W/' + png.headers.etag });
  assert.equal(cached.status, 304);
  assert.equal(cached.body.length, 0);
  assert.match((await f.get('/js/app.0123456789abcdef.js')).headers['cache-control'], /immutable/);
});

test('SCF rejects traversal and non-read methods, and hides files outside the manifest', async t => {
  const f = await site(t);
  for (const path of ['/../server.mjs', '/%2e%2e/server.mjs', '/%5cserver.mjs', '/%00', '/%zz']) {
    assert.equal((await f.get(path)).status, 400, path);
  }
  for (const path of ['/server.mjs', '/site-manifest.json', '/unlisted.txt']) {
    assert.equal((await f.get(path)).status, 404, path);
  }
  const post = await f.get('/', 'POST');
  assert.equal(post.status, 405);
  assert.equal(post.headers.allow, 'GET, HEAD');
});

test('SCF refuses a corrupt release at startup', async t => {
  const f = await site(t);
  await writeFile(join(f.directory, 'public/index.html'), 'wrong bytes');
  const { createSiteHandler } = await import('../scripts/tencent-site/server.mjs');
  await assert.rejects(createSiteHandler(f.directory), /checksum mismatch/);
});

async function deployment(t) {
  const root = await mkdtemp(join(tmpdir(), 'haodu-scf-deploy-test-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  await mkdir(join(root, 'scripts/tencent-site'), { recursive: true });
  await copyFile(join(__dirname, '../scripts/deploy-site-tencent.mjs'), join(root, 'scripts/deploy-site-tencent.mjs'));
  await copyFile(join(__dirname, '../scripts/tencent-site/verify-archive.mjs'), join(root, 'scripts/tencent-site/verify-archive.mjs'));
  await writeFile(join(root, 'scripts/build-site-tencent.mjs'), `
    import {mkdirSync,writeFileSync} from 'node:fs';
    import {createHash} from 'node:crypto';
    mkdirSync('dist',{recursive:true});
    writeFileSync('dist/haodu-site-scf.zip','fixture code');
    writeFileSync('dist/tencent-site-release.json',JSON.stringify({runtime:'Nodejs20.19',archiveSha256:createHash('sha256').update('fixture code').digest('hex')}));
  `);
  const cli = join(root, 'tccli');
  await writeFile(cli, `#!/usr/bin/env node
    const fs=require('node:fs'); const crypto=require('node:crypto');
    const args=process.argv.slice(2); const action=args[1];
    const input=JSON.parse(fs.readFileSync(new URL(args[args.indexOf('--cli-input-json')+1]),'utf8'));
    fs.appendFileSync('calls.jsonl',JSON.stringify({action,args,input})+'\\n');
    if(process.env.DEPLOY_FAIL===action) { console.error('code:UnauthorizedOperation.CAM'); process.exit(1); }
    if(action==='GetFunction') {
      if(process.env.DEPLOY_NEW && !fs.existsSync('created')) { console.error('code:ResourceNotFound.Function'); process.exit(1); }
      console.log(JSON.stringify({Type:'HTTP',Runtime:'Nodejs20.19',Status:'Active'}));
    } else if(action==='CreateFunction') { fs.writeFileSync('created','yes'); console.log('{}'); }
    else if(action==='UpdateFunctionCode') console.log('{}');
    else if(action==='GetFunctionAddress') console.log(JSON.stringify({CodeSha256:crypto.createHash('sha256').update('fixture code').digest('hex'),Url:'https://secret-signed-download.invalid/'}));
    else process.exit(2);
  `, { mode: 0o755 });
  return {
    run: (flags = [], env = {}) => spawnSync(process.execPath, [join(root, 'scripts/deploy-site-tencent.mjs'), '--tccli', cli, ...flags],
      { cwd: root, encoding: 'utf8', env: { ...process.env, ...env } }),
    calls: async () => {
      try { return (await readFile(join(root, 'calls.jsonl'), 'utf8')).trim().split('\n').map(JSON.parse); }
      catch (error) { if (error.code === 'ENOENT') return []; throw error; }
    },
  };
}

test('Tencent dry run does not call the cloud', async t => {
  const f = await deployment(t);
  assert.equal(f.run(['--dry-run']).status, 0);
  assert.deepEqual(await f.calls(), []);
});

test('Tencent deployment updates code and verifies the cloud checksum without exposing signed URLs', async t => {
  const f = await deployment(t);
  const result = f.run();
  assert.equal(result.status, 0, result.stderr);
  const calls = await f.calls();
  assert.deepEqual(calls.map(c => c.action), ['GetFunction', 'UpdateFunctionCode', 'GetFunction', 'GetFunctionAddress']);
  for (const c of calls) {
    assert.equal(c.input.FunctionName, 'haodu-site');
    assert.equal(c.input.Namespace, 'default');
    assert(c.args.includes('ap-shanghai'));
  }
  assert.equal(Buffer.from(calls[1].input.ZipFile, 'base64').toString(), 'fixture code');
  assert.doesNotMatch(result.stdout + result.stderr, /secret-signed-download/);
});

test('Tencent creates only when explicitly requested and fails closed on authorization errors', async t => {
  const f = await deployment(t);
  assert.notEqual(f.run([], { DEPLOY_NEW: '1' }).status, 0);
  const denied = f.run(['--create'], { DEPLOY_FAIL: 'GetFunction' });
  assert.notEqual(denied.status, 0);
  assert(!(await f.calls()).some(c => c.action === 'CreateFunction'));
  const created = f.run(['--create'], { DEPLOY_NEW: '1' });
  assert.equal(created.status, 0, created.stderr);
  const input = (await f.calls()).find(c => c.action === 'CreateFunction').input;
  assert.equal(input.Type, 'HTTP');
  assert.equal(input.MemorySize, 128);
  assert.equal(input.AutoCreateClsTopic, 'FALSE');
});

test('Tencent upload failure stops before checksum verification', async t => {
  const f = await deployment(t);
  const result = f.run([], { DEPLOY_FAIL: 'UpdateFunctionCode' });
  assert.notEqual(result.status, 0);
  assert(!(await f.calls()).some(c => c.action === 'GetFunctionAddress'));
});

test('Cloud ZIP repackaging preserves content; altered and extra files fail verification', async t => {
  const root = await mkdtemp(join(tmpdir(), 'haodu-scf-archive-test-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const { verifyArchiveFiles } = await import('../scripts/tencent-site/verify-archive.mjs');
  await writeFile(join(root, 'scf_bootstrap'), '#!/bin/bash\necho site\n', { mode: 0o755 });
  await writeFile(join(root, 'index.html'), '<p>好读</p>');
  const zip = name => {
    const result = spawnSync('zip', ['-q', name, 'scf_bootstrap', 'index.html'], { cwd: root });
    assert.equal(result.status, 0);
    return join(root, name);
  };
  const local = zip('local.zip');
  await utimes(join(root, 'index.html'), new Date(0), new Date(0));
  const cloud = zip('cloud.zip');
  assert.notDeepEqual(await readFile(local), await readFile(cloud));
  verifyArchiveFiles(local, cloud);
  await writeFile(join(root, 'index.html'), '<p>changed</p>');
  assert.throws(() => verifyArchiveFiles(local, zip('changed.zip')), /content differs/);
  await writeFile(join(root, 'extra.txt'), 'extra');
  spawnSync('zip', ['-q', cloud, 'extra.txt'], { cwd: root });
  assert.throws(() => verifyArchiveFiles(local, cloud), /file list differs/);
});
