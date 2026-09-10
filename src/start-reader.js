(function startReader() {
  "use strict";
  if (globalThis.__superReaderContentLoaded) return;

  // Shared per-page startup; the loaded adapter supplies the implementation.
  const { read, write, clearMarkers, createReader, createReaderAdapter } = globalThis.SuperReader;
  const adapter = createReaderAdapter();
  const reader = createReader({
    read,
    process: adapter.process,
    write,
    clear: clearMarkers,
    publishState: adapter.publishState,
  });
  adapter.connect(reader);
  globalThis.__superReaderContentLoaded = true;
})();
