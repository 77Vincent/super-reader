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

// Toggle model-generated markers in the lead and demos; brand dividers stay visible.
const readerToggle = document.querySelector('[data-reader-toggle]');
if (readerToggle) {
  const markers = document.querySelectorAll('.reader-divider');
  readerToggle.addEventListener('click', () => {
    const enabled = readerToggle.getAttribute('aria-pressed') !== 'true';
    markers.forEach((marker) => { marker.hidden = !enabled; });
    readerToggle.setAttribute('aria-pressed', String(enabled));
    readerToggle.textContent = enabled ? '关闭辅助预览' : '开启辅助预览';
  });
  readerToggle.hidden = false;
}
