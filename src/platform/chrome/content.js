"use strict";

globalThis.SuperReader ??= {};

/** @returns {ReaderAdapter} */
globalThis.SuperReader.createReaderAdapter = function createChromeReaderAdapter() {
  return {
    markerStyleUrl: chrome.runtime.getURL("src/content.css"),
    // The page's callbacks may outlive the extension context after a reload.
    isAvailable: () => Boolean(chrome.runtime?.id),
    async process(texts) {
      const response = await chrome.runtime.sendMessage({
        type: "SUPER_READER_PROCESS", texts,
      });
      if (response?.error) throw new Error(response.error);
      if (!Array.isArray(response?.offsetsByText) || response.offsetsByText.length !== texts.length) {
        throw new Error("Super Reader received an incomplete viewport result");
      }
      return response.offsetsByText;
    },
    publishState(state) {
      try {
        void chrome.runtime.sendMessage({ type: "SUPER_READER_STATE", ...state }).catch(() => {});
      } catch (_) { /* An extension reload can invalidate this page's context. */ }
    },
    connect(reader) {
      chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
        if (message?.type === "SUPER_READER_PING") sendResponse(reader.status());
        if (message?.type === "SUPER_READER_TOGGLE") sendResponse(reader.toggle());
      });
    },
  };
};
