"use strict";

importScripts("boundary-model-data.js", "backend/inference.js", "backend/chunker.js");

// Each message is a complete viewport. The backend only sees text data.
self.onmessage = (event) => {
  const id = event.data?.id;
  try {
    self.postMessage({
      id,
      offsetsByText: globalThis.SuperReaderChunker.process(event.data.texts),
    });
  } catch (error) {
    self.postMessage({
      id,
      error: error instanceof Error ? error.message : String(error),
    });
  }
};
