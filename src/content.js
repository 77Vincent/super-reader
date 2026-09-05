(function initializeSuperReader() {
  "use strict";

  if (globalThis.__superReaderContentLoaded) return;
  globalThis.__superReaderContentLoaded = true;

  const DEFAULTS = {
    enabled: false,
    dividerWidth: 3,
    dividerColor: "red",
  };

  const DIVIDER_COLORS = Object.freeze({
    red: "#ff1744",
    yellow: "#ffd600",
    blue: "#2979ff",
    green: "#00e676",
    text: "currentColor",
  });
  const COLORED_DIVIDER_FILTER = [
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
  const FLOW_BREAK_TAGS = new Set(["BR", "WBR"]);
  const INLINE_FLOW_DISPLAYS = new Set([
    "contents",
    "inline",
    "ruby",
    "ruby-base",
    "ruby-base-container",
    "ruby-text",
    "ruby-text-container",
  ]);
  const OBSERVER_OPTIONS = Object.freeze({
    characterData: true,
    childList: true,
    subtree: true,
  });

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
    return Number.isFinite(width)
      ? Math.min(5, Math.max(1, Math.round(width)))
      : DEFAULTS.dividerWidth;
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

  function isReadableTextNode(textNode) {
    return (
      textNode?.nodeType === Node.TEXT_NODE &&
      textNode.isConnected &&
      Boolean(textNode.parentElement) &&
      !isIgnored(textNode.parentElement) &&
      Boolean(textNode.nodeValue)
    );
  }

  function isInlineFlowElement(element) {
    return INLINE_FLOW_DISPLAYS.has(getComputedStyle(element).display);
  }

  function resolveProcessingScope(root) {
    let element = root?.nodeType === Node.TEXT_NODE
      ? root.parentElement
      : root;
    if (!(element instanceof Element) || !element.isConnected || isIgnored(element)) {
      return null;
    }

    while (
      element.parentElement &&
      element !== document.body &&
      element !== document.documentElement &&
      isInlineFlowElement(element)
    ) {
      element = element.parentElement;
      if (isIgnored(element)) return null;
    }

    return element;
  }

  function collectTextRuns(scope) {
    const runs = [];
    let currentRun = [];

    const flush = () => {
      if (currentRun.length) runs.push(currentRun);
      currentRun = [];
    };

    const visit = (node, isScope = false) => {
      if (node.nodeType === Node.TEXT_NODE) {
        if (isReadableTextNode(node)) currentRun.push(node);
        else flush();
        return;
      }

      if (!(node instanceof Element)) return;
      if (!isScope && isIgnored(node)) {
        flush();
        return;
      }
      if (FLOW_BREAK_TAGS.has(node.tagName)) {
        flush();
        return;
      }

      const createsFlowBoundary = !isScope && !isInlineFlowElement(node);
      if (createsFlowBoundary) flush();
      Array.from(node.childNodes).forEach((child) => visit(child));
      if (createsFlowBoundary) flush();
    };

    visit(scope, true);
    flush();
    return runs;
  }

  function createDividerMarker() {
    const marker = document.createElement("span");
    marker.className = "super-reader-chunk super-reader-chunk--separated";
    marker.dataset.superReaderDivider = "true";
    marker.setAttribute("aria-hidden", "true");
    applyDividerStyle(marker);
    return marker;
  }

  function dividerOffsets(chunks) {
    const offsets = [];
    let offset = 0;

    chunks.forEach((chunk) => {
      if (chunk.separated) offsets.push(offset);
      offset += chunk.text.length;
    });
    return offsets;
  }

  function insertDividerMarkers(textNodes, offsets) {
    const placements = new Map();
    let sourceOffset = 0;
    let nodeIndex = 0;
    const entries = textNodes.map((textNode) => {
      const start = sourceOffset;
      sourceOffset += textNode.nodeValue.length;
      return { textNode, start, end: sourceOffset };
    });

    offsets.forEach((offset) => {
      while (nodeIndex < entries.length && offset >= entries[nodeIndex].end) {
        nodeIndex += 1;
      }
      const entry = entries[nodeIndex];
      if (!entry || offset < entry.start) return;
      const localOffset = offset - entry.start;
      const positions = placements.get(entry.textNode) || [];
      positions.push(localOffset);
      placements.set(entry.textNode, positions);
    });

    placements.forEach((positions, textNode) => {
      [...new Set(positions)]
        .sort((left, right) => right - left)
        .forEach((localOffset) => {
          if (!textNode.isConnected || !textNode.parentNode) return;
          const reference = localOffset === 0
            ? textNode
            : textNode.splitText(localOffset);
          reference.before(createDividerMarker());
        });
    });
  }

  function processTextRun(textNodes) {
    const text = textNodes.map((textNode) => textNode.nodeValue).join("");
    if (!/[\u3400-\u9fff]/u.test(text)) return;
    if (SuperReaderChunker.visualLength(text) < 2) return;

    const chunks = SuperReaderChunker.buildVisualChunks(text, {
      segmenter: state.segmenter,
    });
    insertDividerMarkers(textNodes, dividerOffsets(chunks));
  }

  function removeDividerMarkers(scope) {
    const parents = new Set();
    scope.querySelectorAll("[data-super-reader-divider='true']").forEach((marker) => {
      if (marker.parentNode) parents.add(marker.parentNode);
      marker.remove();
    });
    parents.forEach((parent) => parent.normalize());
  }

  function processRoot(root) {
    if (!state.enabled || !root || !root.isConnected) return;
    const scope = resolveProcessingScope(root);
    if (!scope) return;

    removeDividerMarkers(scope);
    collectTextRuns(scope).forEach(processTextRun);
  }

  function flushPendingRoots() {
    state.flushScheduled = false;
    const roots = Array.from(state.pendingRoots);
    state.pendingRoots.clear();
    if (!state.enabled || !roots.length) return;

    const observer = state.observer;
    observer?.disconnect();
    try {
      roots.forEach(processRoot);
    } finally {
      observer?.takeRecords();
      if (observer && observer === state.observer && state.enabled) {
        observer.observe(document.documentElement, OBSERVER_OPTIONS);
      }
    }
  }

  function queueRoot(root) {
    const scope = resolveProcessingScope(root);
    if (!scope) return;

    for (const pendingScope of state.pendingRoots) {
      if (pendingScope === scope || pendingScope.contains(scope)) return;
      if (scope.contains(pendingScope)) state.pendingRoots.delete(pendingScope);
    }
    state.pendingRoots.add(scope);
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
        if (mutation.removedNodes.length) queueRoot(mutation.target);
      }
    });
    state.observer.observe(document.documentElement, OBSERVER_OPTIONS);
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
      if (span.dataset.superReaderDivider === "true") {
        span.remove();
      } else {
        // Clean up text-wrapping spans created by earlier extension versions.
        span.replaceWith(document.createTextNode(span.textContent || ""));
      }
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
