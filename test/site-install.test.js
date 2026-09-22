const test = require('node:test');
const assert = require('node:assert/strict');

const targetModule = import('../site/assets/js/install-target.js');
const links = {
  chromeUrl: 'https://chromewebstore.google.com/detail/test-chrome-listing',
  edgeUrl: 'https://microsoftedge.microsoft.com/addons/detail/test-edge-listing',
  fallbackUrl: 'https://github.com/77Vincent/super-reader',
};
const chromeUA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/140.0.0.0 Safari/537.36';
const fallback = { href: links.fallbackUrl, label: '查看项目与安装方法' };

test('desktop Chrome opens its configured store', async () => {
  const { selectInstallTarget } = await targetModule;
  assert.deepEqual(selectInstallTarget(links, { userAgent: chromeUA }), {
    href: links.chromeUrl, label: '添加到 Chrome',
  });
});

test('Edge takes precedence over Chrome in the same user agent', async () => {
  const { selectInstallTarget } = await targetModule;
  const browser = { userAgent: `${chromeUA} Edg/140.0.0.0` };
  assert.deepEqual(selectInstallTarget(links, browser), {
    href: links.edgeUrl, label: '添加到 Edge',
  });
  assert.deepEqual(selectInstallTarget({ ...links, edgeUrl: '' }, browser), {
    href: links.chromeUrl, label: '添加到 Edge',
  });
});

test('browser hints work regardless of brand order and override generic Chrome UA', async () => {
  const { selectInstallTarget } = await targetModule;
  for (const brand of ['Microsoft Edge', 'Google Chrome']) {
    const brands = [{ brand: 'Chromium' }, { brand: 'Not_A Brand' }, { brand }];
    const expected = brand === 'Microsoft Edge'
      ? { href: links.edgeUrl, label: '添加到 Edge' }
      : { href: links.chromeUrl, label: '添加到 Chrome' };
    for (const order of [brands, [...brands].reverse()]) {
      assert.deepEqual(selectInstallTarget(links, {
        userAgent: chromeUA, userAgentData: { brands: order, mobile: false },
      }), expected);
    }
  }
});

test('unpublished stores fall back to installation instructions', async () => {
  const { selectInstallTarget } = await targetModule;
  for (const userAgent of [chromeUA, `${chromeUA} Edg/140.0.0.0`]) {
    assert.deepEqual(selectInstallTarget({ fallbackUrl: links.fallbackUrl }, { userAgent }), fallback);
  }
  // An Edge-only listing cannot be used as a Chrome installation link.
  assert.deepEqual(selectInstallTarget({ ...links, chromeUrl: '' }, { userAgent: chromeUA }), fallback);
});

test('mobile browsers and iPad desktop mode receive instructions instead of a desktop install CTA', async () => {
  const { selectInstallTarget } = await targetModule;
  for (const browser of [
    { userAgent: 'Mozilla/5.0 (Linux; Android 14) Chrome/140.0.0.0 Mobile Safari/537.36' },
    { userAgent: 'Mozilla/5.0 (iPhone) CriOS/140.0 Mobile Safari/604.1' },
    { userAgent: 'Mozilla/5.0 (iPhone) EdgiOS/140.0 Mobile Safari/604.1' },
    { userAgent: 'Mozilla/5.0 (Linux; Android 14) Chrome/140.0 EdgA/140.0' },
    { userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15) Version/18.0 Safari/605.1.15', maxTouchPoints: 5 },
    { userAgent: chromeUA, userAgentData: { mobile: true, brands: [{ brand: 'Google Chrome' }] } },
  ]) {
    assert.deepEqual(selectInstallTarget(links, browser), fallback);
  }
});

test('unsupported or unidentified browsers keep the installation instructions', async () => {
  const { selectInstallTarget } = await targetModule;
  for (const userAgent of [
    '',
    'Mozilla/5.0 (Macintosh) Version/18.0 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:140.0) Gecko/20100101 Firefox/140.0',
    `${chromeUA} OPR/120.0`,
    `${chromeUA} Vivaldi/7.0`,
    `${chromeUA} Edge/18.19041`,
  ]) {
    assert.deepEqual(selectInstallTarget(links, { userAgent }), fallback);
  }
  assert.deepEqual(selectInstallTarget(links), fallback);
});
