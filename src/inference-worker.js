"use strict";

importScripts(
  "boundary-model-data.js",
  "model-backend.js",
  "chunker.js",
);

let inferenceSegmenter = null;

function dividerOffsets(text) {
  if (!inferenceSegmenter) {
    inferenceSegmenter = globalThis.SuperReaderChunker.createSegmenter("zh-CN");
  }
  const chunks = globalThis.SuperReaderChunker.buildVisualChunks(text, {
    segmenter: inferenceSegmenter,
  });
  const offsets = [];
  let offset = 0;

  chunks.forEach((chunk) => {
    if (chunk.separated) offsets.push(offset);
    offset += chunk.text.length;
  });
  return offsets;
}

self.onmessage = (event) => {
  const id = event.data?.id;
  try {
    const texts = event.data?.texts;
    if (
      !Number.isInteger(id) ||
      !Array.isArray(texts) ||
      texts.length !== 1 ||
      typeof texts[0] !== "string" || texts[0].length > 128
    ) {
      throw new TypeError("Super Reader received invalid inference input");
    }
    self.postMessage({ id, offsetsByText: texts.map(dividerOffsets) });
  } catch (error) {
    self.postMessage({
      id,
      error: error instanceof Error ? error.message : String(error),
    });
  }
};
