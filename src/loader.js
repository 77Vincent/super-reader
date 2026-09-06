(function initializeSuperReaderLoader() {
  "use strict";

  if (globalThis.__superReaderLoaderLoaded) return;
  globalThis.__superReaderLoaderLoaded = true;

  const RUNTIME_FILES = Object.freeze([
    "src/content.js",
  ]);
  let runtimePromise = null;
  let runtimeReady = Boolean(globalThis.__superReaderContentLoaded);

  function ensureRuntime() {
    if (runtimeReady) return Promise.resolve();
    if (runtimePromise) return runtimePromise;

    runtimePromise = RUNTIME_FILES.reduce(
      (previous, file) => previous.then(() => import(chrome.runtime.getURL(file))),
      Promise.resolve(),
    ).then(() => {
      runtimeReady = true;
    }).catch((error) => {
      runtimePromise = null;
      console.error("Super Reader failed to load its reading runtime", error);
    });

    return runtimePromise;
  }

  chrome.storage.sync.get({ enabled: false }, ({ enabled }) => {
    if (enabled) void ensureRuntime();
  });

  chrome.storage.onChanged.addListener((changes, areaName) => {
    if (areaName === "sync" && changes.enabled?.newValue) {
      void ensureRuntime();
    }
  });

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message?.type === "SUPER_READER_PING") {
      sendResponse({
        ready: true,
        runtimeReady,
        loading: Boolean(runtimePromise) && !runtimeReady,
      });
    }
  });
})();
