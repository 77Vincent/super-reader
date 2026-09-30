const { test } = require('node:test');
const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const { mkdtempSync, mkdirSync, copyFileSync, writeFileSync, readFileSync, existsSync, rmSync } = require('node:fs');
const { tmpdir } = require('node:os');
const { join } = require('node:path');

function fixture(t) {
  const root = mkdtempSync(join(tmpdir(), 'haodu-deploy-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  mkdirSync(join(root, 'scripts'));
  copyFileSync(join(__dirname, '../scripts/deploy-site.mjs'), join(root, 'scripts/deploy-site.mjs'));
  // Exercise deployment independently of Hugo and without calling Alibaba Cloud.
  writeFileSync(join(root, 'scripts/build-site.mjs'), `
    import { mkdirSync, writeFileSync, symlinkSync } from 'node:fs';
    mkdirSync('dist/site/privacy', { recursive: true });
    mkdirSync('dist/site/js', { recursive: true });
    for (const key of ['index.html', '404.html', 'privacy/index.html', 'js/app.0123456789abcdef.js', 'robots.txt']) {
      writeFileSync('dist/site/' + key, key);
    }
    writeFileSync('dist/do-not-upload.txt', 'outside site');
    if (process.env.DEPLOY_TEST_SYMLINK) symlinkSync('../do-not-upload.txt', 'dist/site/linked.txt');
  `);
  const cli = join(root, 'fake-ossutil');
  writeFileSync(cli, `#!/usr/bin/env node
    const fs = require('node:fs');
    const crypto = require('node:crypto');
    const args = process.argv.slice(2);
    const val = flag => args[args.indexOf(flag) + 1];
    fs.appendFileSync('calls.jsonl', JSON.stringify(args) + '\\n');
    if (process.env.DEPLOY_TEST_FAIL && args[1] === process.env.DEPLOY_TEST_FAIL) process.exit(1);
    if (args[1] === 'put-object') {
      const body = fs.readFileSync(val('--body').slice('file://'.length));
      fs.writeFileSync('uploaded-etag', '"' + crypto.createHash('md5').update(body).digest('hex').toUpperCase() + '"');
    } else if (args[1] === 'head-object') {
      if (val('--if-match') !== fs.readFileSync('uploaded-etag', 'utf8')) process.exit(2);
    } else process.exit(3);
  `, { mode: 0o755 });
  return {
    root,
    run: (flags = [], env = {}) => spawnSync(process.execPath,
      [join(root, 'scripts/deploy-site.mjs'), '--ossutil', cli, ...flags],
      { cwd: root, encoding: 'utf8', env: { ...process.env, ...env } }),
    calls: () => existsSync(join(root, 'calls.jsonl'))
      ? readFileSync(join(root, 'calls.jsonl'), 'utf8').trim().split('\n').map(JSON.parse) : [],
  };
}

test('dry run builds the production plan without invoking OSS', t => {
  const f = fixture(t);
  const result = f.run(['--dry-run']);
  assert.equal(result.status, 0, result.stderr);
  assert.match(result.stdout, /Preview: 5 files/);
  assert.doesNotMatch(result.stdout, /do-not-upload/);
  assert.deepEqual(f.calls(), []);
});

test('deployment preserves nested paths, sets cache headers and verifies before publishing the homepage', t => {
  const f = fixture(t);
  const result = f.run(['--profile', 'haodu-site']);
  assert.equal(result.status, 0, result.stderr);
  const calls = f.calls();
  assert.equal(calls.length, 10);
  const value = (args, flag) => args[args.indexOf(flag) + 1];
  const uploads = calls.filter(args => args[1] === 'put-object');
  assert.equal(value(uploads.at(-1), '--key'), 'index.html');
  assert(uploads.some(args => value(args, '--key') === 'privacy/index.html'));
  for (let i = 0; i < calls.length; i += 2) {
    assert.equal(calls[i + 1][1], 'head-object');
    assert.equal(value(calls[i], '--key'), value(calls[i + 1], '--key'));
    assert.equal(value(calls[i], '--bucket'), 'haodu-site');
    assert.equal(value(calls[i], '--profile'), 'haodu-site');
    assert.equal(value(calls[i], '--region'), 'cn-shanghai');
    assert(!calls[i].includes('--object-acl'));
  }
  const html = uploads.filter(args => value(args, '--key').endsWith('.html'));
  for (const args of html) {
    assert.equal(value(args, '--cache-control'), 'no-cache');
    assert.equal(value(args, '--content-type'), 'text/html; charset=utf-8');
  }
  const js = uploads.find(args => value(args, '--key').endsWith('.js'));
  assert.equal(value(js, '--cache-control'), 'public, max-age=31536000, immutable');
});

for (const failure of ['put-object', 'head-object']) {
  test(`a ${failure} failure stops before publishing any HTML`, t => {
    const f = fixture(t);
    assert.notEqual(f.run([], { DEPLOY_TEST_FAIL: failure }).status, 0);
    assert(!f.calls().some(args => args.includes('index.html') || args.includes('privacy/index.html')));
  });
}

test('symlinks cannot upload files from outside the production artifact', t => {
  const f = fixture(t);
  const result = f.run([], { DEPLOY_TEST_SYMLINK: '1' });
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /Unexpected release entry: linked.txt/);
  assert.deepEqual(f.calls(), []);
});
