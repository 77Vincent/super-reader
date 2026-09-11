(function startReader() {
  "use strict";
  if (globalThis.__superReaderContentLoaded) return;

  // Shared per-page startup; the loaded adapter supplies the implementation.
  const {
    read, write, clearMarkers, rememberProcessedText, clearProcessedText,
    watchViewport, createContentChanges, createReader, createReaderAdapter,
  } = globalThis.SuperReader;
  const adapter = createReaderAdapter();
  const changes = createContentChanges(document);
  const reader = createReader({
    read: () => read(changes.observeRoot),
    process: adapter.process,
    write: (snapshot, results) => changes.withoutObservation(() => write(snapshot, results, adapter.markerStyleUrl)),
    clear: clearMarkers,
    remember: rememberProcessedText,
    reset: clearProcessedText,
    watch(onChange) {
      const stopContent = changes.watch(onChange);
      const stopViewport = watchViewport(document, onChange);
      return () => { stopContent(); stopViewport(); };
    },
    publishState: adapter.publishState,
  });
  adapter.connect(reader);
  globalThis.__superReaderContentLoaded = true;
})();
