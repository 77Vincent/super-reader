(function startReader() {
  "use strict";
  if (globalThis.__superReaderContentLoaded) return;

  // Shared per-page startup; the loaded adapter supplies the implementation.
  const {
    read, write, clearMarkers, rememberProcessedText, clearProcessedText,
    removeStaleMarkers, forgetProcessedText, markDirtySources,
    watchViewport, createContentChanges, createReader, createReaderAdapter,
  } = globalThis.SuperReader;
  const adapter = createReaderAdapter();
  const changes = createContentChanges(document, markDirtySources);
  const reader = createReader({
    read: () => {
      changes.withoutObservation(() => {
        const { forgotten, preserved } = removeStaleMarkers();
        forgetProcessedText(forgotten);
        rememberProcessedText(preserved);
      });
      return read(changes.observeRoot);
    },
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
