(function initializeInferenceService() {
  "use strict";

  let inferenceWorker = null;
  let nextRequestId = 1;
  const pendingRequests = new Map();

  function rejectPendingRequests(error) {
    pendingRequests.forEach(({ reject }) => reject(error));
    pendingRequests.clear();
  }

  function ensureInferenceWorker() {
    if (inferenceWorker) return inferenceWorker;

    inferenceWorker = new Worker(
      chrome.runtime.getURL("src/inference-worker.js"),
    );
    inferenceWorker.onmessage = (event) => {
      const request = pendingRequests.get(event.data?.id);
      if (!request) return;
      pendingRequests.delete(event.data.id);
      if (event.data.error) request.reject(new Error(event.data.error));
      else request.resolve(event.data.offsetsByText);
    };
    inferenceWorker.onerror = (event) => {
      const error = new Error(event.message || "Super Reader inference worker failed");
      rejectPendingRequests(error);
      inferenceWorker?.terminate();
      inferenceWorker = null;
    };
    return inferenceWorker;
  }

  function runInference(texts) {
    return new Promise((resolve, reject) => {
      const id = nextRequestId;
      nextRequestId += 1;
      pendingRequests.set(id, { reject, resolve });
      ensureInferenceWorker().postMessage({ id, texts });
    });
  }

  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (
      message?.target !== "offscreen" ||
      message?.type !== "SUPER_READER_RUN_INFERENCE"
    ) {
      return undefined;
    }

    const texts = message.texts;
    if (
      sender.id !== chrome.runtime.id ||
      !Array.isArray(texts) ||
      texts.some((text) => typeof text !== "string")
    ) {
      sendResponse({ error: "Super Reader received invalid inference input" });
      return false;
    }

    void runInference(texts).then(
      (offsetsByText) => sendResponse({ offsetsByText }),
      (error) => sendResponse({
        error: error instanceof Error ? error.message : String(error),
      }),
    );
    return true;
  });
})();
