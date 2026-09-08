(function initializeInferenceService() {
  "use strict";

  let inferenceWorker = null;
  let nextRequestId = 1;
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

    inferenceWorker = new Worker(
      chrome.runtime.getURL("src/inference-worker.js"),
    );
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
      texts.length !== 1 ||
      typeof texts[0] !== "string" || texts[0].length > 128
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
