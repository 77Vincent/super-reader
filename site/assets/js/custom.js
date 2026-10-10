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
