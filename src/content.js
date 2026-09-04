(function initializeSuperReader() {
  "use strict";

  if (globalThis.__superReaderContentLoaded) return;
  globalThis.__superReaderContentLoaded = true;

  const DEFAULTS = {
    enabled: false,
    targetLength: 7,
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

    const minimum = Math.max(2, state.settings.targetLength - 2);
    return SuperReaderChunker.visualLength(text) >= minimum;
  }

  function processTextNode(textNode) {
    if (!shouldProcess(textNode)) return;

    const chunks = SuperReaderChunker.buildVisualChunks(textNode.nodeValue, {
      targetLength: state.settings.targetLength,
    });
    if (!chunks.length) return;

    const fragment = document.createDocumentFragment();
    chunks.forEach((chunk) => {
      const runs = SuperReaderChunker.splitUnderlineRuns(chunk.text);

      runs.forEach((run) => {
        if (!run.underlinable) {
          fragment.append(document.createTextNode(run.text));
          return;
        }
        const span = document.createElement("span");
        const variant = chunk.underlined ? "a" : "b";
        span.className = `super-reader-chunk super-reader-chunk--${variant}`;
        span.dataset.superReaderChunk = variant;
        span.textContent = run.text;
        fragment.append(span);
      });
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
    const targetChanged = Object.prototype.hasOwnProperty.call(nextSettings, "targetLength")
      && nextSettings.targetLength !== state.settings.targetLength;

    state.settings = { ...state.settings, ...nextSettings };

    if (wasEnabled && !state.settings.enabled) {
      disable();
    } else if (!wasEnabled && state.settings.enabled) {
      enable();
    } else if (state.enabled && targetChanged) {
      disable();
      enable();
    }
  }

  chrome.storage.sync.get(DEFAULTS, applySettings);

  chrome.storage.onChanged.addListener((changes, areaName) => {
    if (areaName !== "sync") return;
    const nextSettings = {};
    for (const [key, change] of Object.entries(changes)) {
      nextSettings[key] = change.newValue;
    }
    applySettings(nextSettings);
  });

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message?.type === "SUPER_READER_PING") {
      sendResponse({ ready: true, enabled: state.enabled });
    }
  });
})();
