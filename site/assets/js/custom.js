// Toggle model-generated markers in the lead and demos; brand dividers stay visible.
const readerToggle = document.querySelector('[data-reader-toggle]');
if (readerToggle) {
  const installButton = document.querySelector('[data-install-button]');
  if (installButton) {
    // Match the rendered grid width, including responsive layout and font changes.
    const syncToggleSize = () => {
      const { width, height } = installButton.getBoundingClientRect();
      readerToggle.style.width = `${width}px`;
      readerToggle.style.height = `${height}px`;
    };
    syncToggleSize();
    new ResizeObserver(syncToggleSize).observe(installButton);
  }
  const markers = document.querySelectorAll('.reader-divider');
  const label = readerToggle.querySelector('[data-reader-label]');
  readerToggle.addEventListener('click', () => {
    const enabled = readerToggle.getAttribute('aria-pressed') !== 'true';
    markers.forEach((marker) => { marker.hidden = !enabled; });
    readerToggle.setAttribute('aria-pressed', String(enabled));
    label.textContent = enabled ? '关闭辅助预览' : '开启辅助预览';
  });
  readerToggle.hidden = false;
}
