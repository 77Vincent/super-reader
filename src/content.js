(function initializeSuperReader() {
  "use strict";

  if (globalThis.__superReaderContentLoaded) return;
  globalThis.__superReaderContentLoaded = true;

  const DEFAULTS = {
    enabled: false,
  };

  const IGNORED_TAGS = new Set([
    "SCRIPT",
    "STYLE",
    "NOSCRIPT",
    "INPUT",
    "TEXTAREA",
    "SELECT",
    "OPTION",
    "BUTTON",
    "CODE",
    "PRE",
    "KBD",
    "SAMP",
    "VAR",
    "SVG",
    "MATH",
    "CANVAS",
  ]);

  const state = {
    enabled: false,
    observer: null,
    settings: { ...DEFAULTS },
    segmenter: SuperReaderChunker.createSegmenter("zh-CN"),
    pendingRoots: new Set(),
    flushScheduled: false,
  };

  function isIgnored(element) {
    if (!(element instanceof Element)) return true;
    if (IGNORED_TAGS.has(element.tagName)) return true;
    if (element.closest(".super-reader-chunk")) return true;
    if (element.closest("[contenteditable]:not([contenteditable='false'])")) return true;
    if (element.closest("[aria-hidden='true']")) return true;
    return false;
  }

  function shouldProcess(textNode) {
    if (!textNode.isConnected || !textNode.parentElement) return false;
    if (isIgnored(textNode.parentElement)) return false;

    const text = textNode.nodeValue;
    if (!text || !/[\u3400-\u9fff]/u.test(text)) return false;

    return SuperReaderChunker.visualLength(text) >= 2;
  }

  function processTextNode(textNode) {
    if (!shouldProcess(textNode)) return;

    const chunks = SuperReaderChunker.buildVisualChunks(textNode.nodeValue, {
      segmenter: state.segmenter,
    });
    if (!chunks.length) return;

    const fragment = document.createDocumentFragment();
    chunks.forEach((chunk) => {
      if (!chunk.processed) {
        fragment.append(document.createTextNode(chunk.text));
        return;
      }

      const span = document.createElement("span");
      span.className = chunk.separated
        ? "super-reader-chunk super-reader-chunk--separated"
        : "super-reader-chunk";
      span.dataset.superReaderChunk = "true";
      span.textContent = chunk.text;
      fragment.append(span);
    });

    textNode.replaceWith(fragment);
  }

  function processRoot(root) {
    if (!state.enabled || !root || !root.isConnected) return;

    if (root.nodeType === Node.TEXT_NODE) {
      processTextNode(root);
      return;
    }

    if (!(root instanceof Element) || isIgnored(root)) return;

    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        return shouldProcess(node)
          ? NodeFilter.FILTER_ACCEPT
          : NodeFilter.FILTER_REJECT;
      },
    });
    const textNodes = [];
    while (walker.nextNode()) textNodes.push(walker.currentNode);
    textNodes.forEach(processTextNode);
  }

  function flushPendingRoots() {
    state.flushScheduled = false;
    const roots = Array.from(state.pendingRoots);
    state.pendingRoots.clear();
    roots.forEach(processRoot);
  }

  function queueRoot(root) {
    if (!root || !root.isConnected) return;
    state.pendingRoots.add(root);
    if (state.flushScheduled) return;
    state.flushScheduled = true;
    queueMicrotask(flushPendingRoots);
  }

  function startObserver() {
    if (state.observer || !document.documentElement) return;

    state.observer = new MutationObserver((mutations) => {
      if (!state.enabled) return;
      for (const mutation of mutations) {
        if (mutation.type === "characterData") {
          queueRoot(mutation.target);
        }
        mutation.addedNodes.forEach(queueRoot);
      }
    });
    state.observer.observe(document.documentElement, {
      characterData: true,
      childList: true,
      subtree: true,
    });
  }

  function stopObserver() {
    state.observer?.disconnect();
    state.observer = null;
    state.pendingRoots.clear();
    state.flushScheduled = false;
  }

  function restoreDocument() {
    stopObserver();
    const parents = new Set();

    document.querySelectorAll(".super-reader-chunk").forEach((span) => {
      if (span.parentNode) parents.add(span.parentNode);
      span.replaceWith(document.createTextNode(span.textContent || ""));
    });

    parents.forEach((parent) => parent.normalize());
  }

  function enable() {
    if (state.enabled) return;
    state.enabled = true;
    processRoot(document.body || document.documentElement);
    startObserver();
  }

  function disable() {
    state.enabled = false;
    restoreDocument();
  }

  function applySettings(nextSettings) {
    const wasEnabled = state.enabled;
    state.settings = {
      enabled: Object.prototype.hasOwnProperty.call(nextSettings, "enabled")
        ? Boolean(nextSettings.enabled)
        : state.settings.enabled,
    };

    if (wasEnabled && !state.settings.enabled) {
      disable();
    } else if (!wasEnabled && state.settings.enabled) {
      enable();
    }
  }

  chrome.storage.sync.get(DEFAULTS, applySettings);

  chrome.storage.onChanged.addListener((changes, areaName) => {
    if (areaName !== "sync" || !changes.enabled) return;
    applySettings({ enabled: changes.enabled.newValue });
  });

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message?.type === "SUPER_READER_PING") {
      sendResponse({ ready: true, enabled: state.enabled });
    }
  });
})();
