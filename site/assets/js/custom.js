import { selectInstallTarget } from './install-target.js';

// Only choose a store link; installation still happens in the browser's store.
document.querySelectorAll('[data-install-link]').forEach((link) => {
  const target = selectInstallTarget(link.dataset, navigator);
  link.href = target.href;
  link.textContent = target.label;
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
