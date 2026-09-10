"use strict";

globalThis.SuperReader ??= {};

/** @returns {InferenceAdapter} */
globalThis.SuperReader.createInferenceAdapter = function createChromeInferenceAdapter() {
  return {
    workerUrl: chrome.runtime.getURL("src/inference-worker.js"),
    connect(service) {
      chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
        if (message?.target !== "offscreen" || message?.type !== "SUPER_READER_RUN_INFERENCE") {
          return undefined;
        }
        if (sender.id !== chrome.runtime.id) return false;

        void service.runInference(message.texts).then(
          (offsetsByText) => sendResponse({ offsetsByText }),
          (error) => sendResponse({
            error: error instanceof Error ? error.message : String(error),
          }),
        );
        return true;
      });
    },
  };
};
