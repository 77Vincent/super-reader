(function startReader() {
  "use strict";
  if (globalThis.__superReaderContentLoaded) return;

  // Shared per-page startup; the loaded adapter supplies the implementation.
  const {
    read, write, clearMarkers, rememberProcessedText, clearProcessedText,
    watchViewport, createDOMChanges, createReader, createReaderAdapter,
  } = globalThis.SuperReader;
  const adapter = createReaderAdapter();
  const changes = createDOMChanges(document);
  const reader = createReader({
    read: () => read(changes.observeRoot),
    process: adapter.process,
    write: (snapshot, results) => changes.mutate(() => write(snapshot, results, adapter.markerStyleUrl)),
    clear: () => changes.mutate(clearMarkers),
    remember: rememberProcessedText,
    reset: clearProcessedText,
    watch(onChange) {
      const stopDOM = changes.watch(onChange);
      const stopViewport = watchViewport(document, onChange);
      return () => { stopViewport(); stopDOM(); };
    },
    publishState: adapter.publishState,
  });
  adapter.connect(reader);
  globalThis.__superReaderContentLoaded = true;
})();
