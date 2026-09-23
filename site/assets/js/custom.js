import Offcanvas from 'bootstrap/js/dist/offcanvas';
import { selectInstallTarget } from './install-target.js';

// Only choose a store link; installation still happens in the browser's store.
document.querySelectorAll('[data-install-link]').forEach((link) => {
  const target = selectInstallTarget(link.dataset, navigator);
  link.href = target.href;
  link.textContent = target.label;
});

// Close Doks' mobile menu after following an anchor on this single-page site.
const menu = document.querySelector('#offcanvasNavMain');
menu?.addEventListener('click', (event) => {
  const link = event.target.closest('a[href]');
  if (link?.hash && link.origin === location.origin && link.pathname === location.pathname) {
    Offcanvas.getInstance(menu)?.hide();
  }
});

// Static examples only: toggle prepared markers without loading a model.
document.querySelectorAll('[data-reader-demo]').forEach((demo) => {
  const button = demo.querySelector('[data-reader-demo-toggle]');
  const markers = demo.querySelectorAll('.reader-divider');
  button.addEventListener('click', () => {
    const enabled = button.getAttribute('aria-pressed') !== 'true';
    markers.forEach((marker) => { marker.hidden = !enabled; });
    button.setAttribute('aria-pressed', String(enabled));
    button.textContent = enabled ? '关闭阅读辅助' : '启动阅读辅助';
  });
  button.hidden = false;
});
