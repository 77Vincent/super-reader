// Low-entropy browser hints are sufficient; no network requests are needed.
export function selectInstallTarget({ chromeUrl = '', edgeUrl = '', fallbackUrl }, browser = {}) {
  const fallback = { href: fallbackUrl, label: '查看项目与安装方法' };
  const userAgent = browser.userAgent || '';
  const hints = browser.userAgentData;
  const hasBrand = (name) => hints?.brands?.some(({ brand }) => brand === name);
  const isMobile = hints?.mobile || /Android|Mobi|iPhone|iPad|iPod/i.test(userAgent)
    || (/Macintosh/i.test(userAgent) && browser.maxTouchPoints > 1);

  // This extension targets desktop Chrome and Chromium-based Edge.
  if (isMobile || /\b(?:Edge|OPR|Opera|Vivaldi)\//.test(userAgent)) return fallback;

  // Edge also includes Chrome in its user agent, so check it first.
  if (hasBrand('Microsoft Edge') || /\bEdg\//.test(userAgent)) {
    const href = edgeUrl || chromeUrl;
    return href ? { href, label: '添加到 Edge' } : fallback;
  }
  if (hasBrand('Google Chrome') || /\bChrome\//.test(userAgent)) {
    return chromeUrl ? { href: chromeUrl, label: '添加到 Chrome' } : fallback;
  }
  return fallback;
}
