import { execFileSync, spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';
import { setTimeout } from 'node:timers/promises';
import { verifyRepackedCode } from './tencent-site/verify-archive.mjs';

const { values } = parseArgs({ options: {
  'dry-run': { type: 'boolean', default: false },
  create: { type: 'boolean', default: false },
  profile: { type: 'string', default: 'haodu-site' },
  tccli: { type: 'string' },
} });
const root = fileURLToPath(new URL('../', import.meta.url));
const localCli = join(homedir(), '.local/share/haodu-tencent-tools/bin/tccli');
const tccli = values.tccli || (existsSync(localCli) ? localCli : 'tccli');
execFileSync(process.execPath, [join(root, 'scripts/build-site-tencent.mjs')], { cwd: root, stdio: 'inherit' });
const report = JSON.parse(await readFile(join(root, 'dist/tencent-site-release.json'), 'utf8'));
const target = { FunctionName: 'haodu-site', Namespace: 'default' };
console.log(`${values['dry-run'] ? 'Preview' : 'Deploy'}: ap-shanghai/default/haodu-site (https://haodu.site/)`);
if (values['dry-run']) {
  console.log('Preview complete; no Tencent Cloud API requests were made.');
} else {
  const temporary = await mkdtemp(join(tmpdir(), 'haodu-scf-deploy-'));
  async function api(action, parameters) {
    const input = join(temporary, 'request.json');
    await writeFile(input, JSON.stringify(parameters), { mode: 0o600 });
    const result = spawnSync(tccli, ['scf', action, '--region', 'ap-shanghai',
      '--version', '2018-04-16', '--profile', values.profile, '--output', 'json',
      '--cli-input-json', pathToFileURL(input).href],
    { cwd: root, encoding: 'utf8', timeout: 60000, maxBuffer: 4 * 1024 * 1024 });
    if (result.error) throw result.error;
    let data;
    try { data = JSON.parse(result.stdout); } catch { /* CLI failures may be plain text. */ }
    data = data?.Response || data;
    if (result.status !== 0 || data?.Error || !data) {
      const code = data?.Error?.Code
        || /code[:=]\s*([A-Za-z][A-Za-z.]+)/.exec(result.stderr + result.stdout)?.[1]
        || 'CliFailure';
      // Never log full SDK responses, which can contain signed code-download URLs.
      const error = new Error(`${action} failed: ${code}. Deployment stopped.`);
      error.code = code;
      throw error;
    }
    return data;
  }
  try {
    let current;
    try { current = await api('GetFunction', { ...target, ShowCode: 'FALSE' }); }
    catch (error) {
      if (!['ResourceNotFound.Function', 'ResourceNotFound.FunctionName'].includes(error.code)) throw error;
      if (!values.create) throw new Error('Function does not exist. Use --create for the initial deployment.');
    }
    if (current && (current.Type !== 'HTTP' || current.Runtime !== report.runtime || current.Status !== 'Active')) {
      throw new Error('Existing function is not an active Haodu-compatible Web function; inspect it before updating.');
    }
    const zip = (await readFile(join(root, 'dist/haodu-site-scf.zip'))).toString('base64');
    if (!current) {
      await api('CreateFunction', { ...target, Type: 'HTTP', Runtime: report.runtime,
        Description: 'haodu.site static website', MemorySize: 128, Timeout: 10, InitTimeout: 15,
        Code: { ZipFile: zip }, CodeSource: 'ZipFile', InstallDependency: 'FALSE',
        PublicNetConfig: { PublicNetStatus: 'DISABLE', EipConfig: { EipStatus: 'DISABLE' } },
        AutoCreateClsTopic: 'FALSE', AutoDeployClsTopicIndex: 'FALSE',
      });
    } else {
      await api('UpdateFunctionCode', { ...target, ZipFile: zip, CodeSource: 'ZipFile', InstallDependency: 'FALSE' });
    }
    let verified = false;
    for (let attempt = 0; attempt < 40; attempt++) {
      const state = await api('GetFunction', { ...target, ShowCode: 'FALSE' });
      if (/failed|error/i.test(state.Status || '')) throw new Error(`Function deployment failed (${state.Status}).`);
      if (state.Status === 'Active') {
        const code = await api('GetFunctionAddress', target);
        if (code.CodeSha256?.toLowerCase() === report.archiveSha256
          || code.CodeSha256 === Buffer.from(report.archiveSha256, 'hex').toString('base64')) {
          verified = true;
          break;
        }
        await verifyRepackedCode(code, join(root, 'dist/haodu-site-scf.zip'), temporary);
        verified = true;
        break;
      }
      await setTimeout(3000);
    }
    if (!verified) throw new Error('Cloud archive checksum was not confirmed; do not consider this release deployed.');
    console.log(`SCF code deployed and all release files verified: ${report.release}`);
    console.log('Domain, ICP filing, DNS, HTTPS and CDN must also be configured and checked before the site is live.');
  } finally { await rm(temporary, { recursive: true, force: true }); }
}
