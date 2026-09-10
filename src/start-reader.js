(function startReader() {
  "use strict";
  if (globalThis.__superReaderContentLoaded) return;

  // Shared per-page startup; the loaded adapter supplies the implementation.
  const {
    read, write, clearMarkers, rememberProcessedText, clearProcessedText,
    watchViewport, createReader, createReaderAdapter,
  } = globalThis.SuperReader;
  const adapter = createReaderAdapter();
  const reader = createReader({
    read,
    process: adapter.process,
    write: (snapshot, results) => write(snapshot, results, adapter.markerStyleUrl),
    clear: clearMarkers,
    remember: rememberProcessedText,
    reset: clearProcessedText,
    watch: (onChange) => watchViewport(document, onChange),
    publishState: adapter.publishState,
  });
  adapter.connect(reader);
  globalThis.__superReaderContentLoaded = true;
})();
