"use strict";

globalThis.SuperReader ??= {};

/**
 * Uses standard Worker APIs. The host supplies the URL and routes requests.
 * @param {{workerUrl: string}} options
 * @returns {InferenceService}
 */
globalThis.SuperReader.createInferenceService = function createInferenceService({ workerUrl }) {
  let inferenceWorker = null;
  let nextRequestId = 1;
  // Correlate requests from different pages sharing this Worker; no batch queue.
  const pendingRequests = new Map();
  const REQUEST_TIMEOUT_MS = 5000;

  function rejectPendingRequests(error) {
    pendingRequests.forEach(({ reject, timer }) => {
      clearTimeout(timer);
      reject(error);
    });
    pendingRequests.clear();
    inferenceWorker?.terminate();
    inferenceWorker = null;
  }

  function ensureInferenceWorker() {
    if (inferenceWorker) return inferenceWorker;

    inferenceWorker = new Worker(workerUrl);
    inferenceWorker.onmessage = (event) => {
      const request = pendingRequests.get(event.data?.id);
      if (!request) return;
      pendingRequests.delete(event.data.id);
      clearTimeout(request.timer);
      if (event.data.error) request.reject(new Error(event.data.error));
      else request.resolve(event.data.offsetsByText);
    };
    inferenceWorker.onerror = (event) => {
      const error = new Error(event.message || "Super Reader inference worker failed");
      rejectPendingRequests(error);
    };
    return inferenceWorker;
  }

  function runInference(texts) {
    return new Promise((resolve, reject) => {
      if (!Array.isArray(texts) || texts.some((text) => typeof text !== "string")) {
        throw new TypeError("Super Reader received invalid inference input");
      }
      if (texts.length === 0) { resolve([]); return; }
      const worker = ensureInferenceWorker();
      const id = nextRequestId;
      nextRequestId += 1;
      const timer = setTimeout(() => {
        rejectPendingRequests(new Error("Super Reader inference timed out"));
      }, REQUEST_TIMEOUT_MS);
      pendingRequests.set(id, { reject, resolve, timer });
      try {
        worker.postMessage({ id, texts });
      } catch (error) {
        rejectPendingRequests(error);
      }
    });
  }

  return { runInference };
};
