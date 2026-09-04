(function initializeSuperReader() {
  "use strict";

  if (globalThis.__superReaderContentLoaded) return;
  globalThis.__superReaderContentLoaded = true;

  const DEFAULTS = {
    enabled: false,
    dividerWidth: 2,
    dividerColor: "text",
  };

  const DIVIDER_COLORS = Object.freeze({
    red: "#ff1744",
    yellow: "#ffd600",
    blue: "#2979ff",
    green: "#00e676",
    text: "currentColor",
  });
  const COLORED_DIVIDER_FILTER = [
    "drop-shadow(-0.5px 0 0 rgba(0, 0, 0, 0.65))",
    "drop-shadow(0.5px 0 0 rgba(255, 255, 255, 0.7))",
  ].join(" ");

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

  function normalizeDividerWidth(value) {
    const width = Number(value);
    return Number.isFinite(width) ? Math.min(4, Math.max(1, width)) : DEFAULTS.dividerWidth;
  }

  function normalizeDividerColor(value) {
    return Object.hasOwn(DIVIDER_COLORS, value) ? value : DEFAULTS.dividerColor;
  }

  function applyDividerStyle(element) {
    const colorName = state.settings.dividerColor;
    element.style.setProperty(
      "--super-reader-divider-width",
      `${state.settings.dividerWidth}px`,
    );
    element.style.setProperty(
      "--super-reader-divider-color",
      DIVIDER_COLORS[colorName],
    );
    element.style.setProperty(
      "--super-reader-divider-filter",
      colorName === "text" ? "none" : COLORED_DIVIDER_FILTER,
    );
  }

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
      applyDividerStyle(span);
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

  function updateDividerStyles() {
    document.querySelectorAll(".super-reader-chunk").forEach((span) => {
      applyDividerStyle(span);
    });
  }

  function applySettings(nextSettings) {
    const wasEnabled = state.enabled;
    const dividerWidth = Object.prototype.hasOwnProperty.call(nextSettings, "dividerWidth")
      ? normalizeDividerWidth(nextSettings.dividerWidth)
      : state.settings.dividerWidth;
    const dividerColor = Object.prototype.hasOwnProperty.call(nextSettings, "dividerColor")
      ? normalizeDividerColor(nextSettings.dividerColor)
      : state.settings.dividerColor;
    const dividerWidthChanged = dividerWidth !== state.settings.dividerWidth;
    const dividerColorChanged = dividerColor !== state.settings.dividerColor;
    state.settings = {
      enabled: Object.prototype.hasOwnProperty.call(nextSettings, "enabled")
        ? Boolean(nextSettings.enabled)
        : state.settings.enabled,
      dividerWidth,
      dividerColor,
    };

    if (wasEnabled && !state.settings.enabled) {
      disable();
    } else if (!wasEnabled && state.settings.enabled) {
      enable();
    } else if (state.enabled && (dividerWidthChanged || dividerColorChanged)) {
      updateDividerStyles();
    }
  }

  chrome.storage.sync.get(DEFAULTS, applySettings);

  chrome.storage.onChanged.addListener((changes, areaName) => {
    if (areaName !== "sync") return;
    const nextSettings = {};
    if (changes.enabled) nextSettings.enabled = changes.enabled.newValue;
    if (changes.dividerWidth) {
      nextSettings.dividerWidth = changes.dividerWidth.newValue;
    }
    if (changes.dividerColor) {
      nextSettings.dividerColor = changes.dividerColor.newValue;
    }
    if (Object.keys(nextSettings).length) applySettings(nextSettings);
  });

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message?.type === "SUPER_READER_PING") {
      sendResponse({ ready: true, enabled: state.enabled });
    }
  });
})();
