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
  const VISIBILITY_OBSERVER_OPTIONS = Object.freeze({
    root: null,
    rootMargin: "0px",
    threshold: 0,
  });

  const state = {
    enabled: false,
    observer: null,
    visibilityObserver: null,
    visibilityTargets: new Set(),
    shadowRoots: new Set(),
    settings: { ...DEFAULTS },
    segmenter: SuperReaderChunker.createSegmenter("zh-CN"),
    pendingRoots: new Set(),
    flushScheduled: false,
    flushHandle: null,
    flushKind: null,
  };

  function isOpenShadowRoot(root) {
    return (
      root?.nodeType === Node.DOCUMENT_FRAGMENT_NODE &&
      root.host instanceof Element &&
      root.host.shadowRoot === root
    );
  }

  function observeMutationTarget(root, observer = state.observer) {
    if (!observer) return;
    const target = root === document ? document.documentElement : root;
    if (!target) return;
    observer.observe(target, OBSERVER_OPTIONS);
  }

  function observeRegisteredMutationTargets(observer = state.observer) {
    if (!observer) return;
    observeMutationTarget(document, observer);
    state.shadowRoots.forEach((root) => {
      if (root.host.isConnected) observeMutationTarget(root, observer);
    });
  }

  function registerShadowRoot(root) {
    if (!isOpenShadowRoot(root) || state.shadowRoots.has(root)) return false;
    state.shadowRoots.add(root);
    observeMutationTarget(root);
    return true;
  }

  function pruneDisconnectedShadowRoots() {
    let removed = false;
    state.shadowRoots.forEach((root) => {
      if (root.host.isConnected) return;
      state.shadowRoots.delete(root);
      removed = true;
    });
    return removed;
  }

  function registeredQueryRoots() {
    return [document, ...state.shadowRoots];
  }

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

  function applyShadowMarkerLayout(marker) {
    marker.style.setProperty("display", "inline-block", "important");
    marker.style.setProperty("width", "0", "important");
    marker.style.setProperty("height", "1em", "important");
    marker.style.setProperty("margin", "0 0.08em", "important");
    marker.style.setProperty(
      "border-inline-start",
      "var(--super-reader-divider-width, 3px) solid var(--super-reader-divider-color, #ff1744)",
      "important",
    );
    marker.style.setProperty("vertical-align", "-0.15em", "important");
    marker.style.setProperty(
      "filter",
      "var(--super-reader-divider-filter, none)",
      "important",
    );
    marker.style.setProperty("pointer-events", "none", "important");
  }

  function createDividerMarker(root = document) {
    const marker = document.createElement("span");
    marker.className = "super-reader-chunk super-reader-chunk--separated";
    marker.dataset.superReaderDivider = "true";
    marker.setAttribute("aria-hidden", "true");
    if (isOpenShadowRoot(root)) applyShadowMarkerLayout(marker);
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
          reference.before(createDividerMarker(reference.getRootNode()));
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

  function scheduleFlush() {
    if (state.flushScheduled || !state.enabled) return;
    state.flushScheduled = true;

    if (typeof requestIdleCallback === "function") {
      state.flushKind = "idle";
      state.flushHandle = requestIdleCallback(flushPendingRoots, { timeout: 150 });
      return;
    }

    state.flushKind = "timeout";
    state.flushHandle = setTimeout(() => {
      flushPendingRoots({ didTimeout: true, timeRemaining: () => 0 });
    }, 0);
  }

  function cancelScheduledFlush() {
    if (!state.flushScheduled) return;
    if (state.flushKind === "idle" && typeof cancelIdleCallback === "function") {
      cancelIdleCallback(state.flushHandle);
    } else if (state.flushKind === "timeout") {
      clearTimeout(state.flushHandle);
    }
    state.flushScheduled = false;
    state.flushHandle = null;
    state.flushKind = null;
  }

  function flushPendingRoots(deadline) {
    state.flushScheduled = false;
    state.flushHandle = null;
    state.flushKind = null;
    const roots = Array.from(state.pendingRoots);
    state.pendingRoots.clear();
    if (!state.enabled || !roots.length) return;

    const observer = state.observer;
    observer?.disconnect();
    try {
      let processed = 0;
      for (let index = 0; index < roots.length; index += 1) {
        const shouldYield = processed > 0 && (
          processed >= 4 ||
          (!deadline?.didTimeout && deadline?.timeRemaining() < 4)
        );
        if (shouldYield) {
          roots.slice(index).forEach((root) => state.pendingRoots.add(root));
          break;
        }
        processRoot(roots[index]);
        processed += 1;
      }
    } finally {
      observer?.takeRecords();
      if (observer && observer === state.observer && state.enabled) {
        pruneDisconnectedShadowRoots();
        observeRegisteredMutationTargets(observer);
      }
    }

    if (state.pendingRoots.size) scheduleFlush();
  }

  function queueRoot(root) {
    const scope = resolveProcessingScope(root);
    if (!scope) return;

    for (const pendingScope of state.pendingRoots) {
      if (pendingScope === scope || pendingScope.contains(scope)) return;
      if (scope.contains(pendingScope)) state.pendingRoots.delete(pendingScope);
    }
    state.pendingRoots.add(scope);
    scheduleFlush();
  }

  function isObservableScope(scope) {
    return (
      scope instanceof Element &&
      scope.isConnected &&
      scope !== document.body &&
      scope !== document.documentElement &&
      !isIgnored(scope)
    );
  }

  function observeScopeWhenVisible(scope) {
    if (!isObservableScope(scope)) return;
    for (const pendingScope of state.pendingRoots) {
      if (pendingScope === scope || pendingScope.contains(scope)) return;
    }
    if (!state.visibilityObserver) {
      queueRoot(scope);
      return;
    }
    if (state.visibilityTargets.has(scope)) return;

    state.visibilityTargets.add(scope);
    state.visibilityObserver.observe(scope);
  }

  function observeRootWhenVisible(root) {
    observeScopeWhenVisible(resolveProcessingScope(root));
  }

  function observeTextScopes(root) {
    if (!root) return;
    const scopeCache = new Map();

    const visit = (node, isRoot = false) => {
      if (node.nodeType === Node.TEXT_NODE) {
        if (!/[\u3400-\u9fff]/u.test(node.nodeValue || "")) return;
        const parent = node.parentElement;
        if (!parent || !isReadableTextNode(node)) return;
        if (!scopeCache.has(parent)) {
          scopeCache.set(parent, resolveProcessingScope(node));
        }
        observeScopeWhenVisible(scopeCache.get(parent));
        return;
      }

      if (isOpenShadowRoot(node)) {
        registerShadowRoot(node);
        for (let child = node.firstChild; child; child = child.nextSibling) {
          visit(child);
        }
        return;
      }

      if (!(node instanceof Element)) return;
      const ignored = isIgnored(node);
      if (!isRoot && ignored) return;
      if (!ignored && node.shadowRoot) visit(node.shadowRoot, true);
      for (let child = node.firstChild; child; child = child.nextSibling) {
        visit(child);
      }
    };

    if (root.nodeType === Node.TEXT_NODE) {
      visit(root, true);
    } else if (root instanceof Element || isOpenShadowRoot(root)) {
      visit(root, true);
    }
  }

  function removeDisconnectedVisibilityTargets() {
    for (const target of state.visibilityTargets) {
      if (target.isConnected) continue;
      state.visibilityObserver?.unobserve(target);
      state.visibilityTargets.delete(target);
    }
  }

  function affectsCurrentTextRun(node) {
    return (
      node.nodeType === Node.TEXT_NODE ||
      (node instanceof Element && isInlineFlowElement(node))
    );
  }

  function startVisibilityObserver() {
    if (state.visibilityObserver || typeof IntersectionObserver !== "function") return;

    state.visibilityObserver = new IntersectionObserver((entries) => {
      if (!state.enabled) return;
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        state.visibilityObserver?.unobserve(entry.target);
        state.visibilityTargets.delete(entry.target);
        queueRoot(entry.target);
      });
    }, VISIBILITY_OBSERVER_OPTIONS);
  }

  function stopVisibilityObserver() {
    state.visibilityObserver?.disconnect();
    state.visibilityObserver = null;
    state.visibilityTargets.clear();
  }

  function startObserver() {
    if (state.observer || !document.documentElement) return;

    state.observer = new MutationObserver((mutations) => {
      if (!state.enabled) return;
      for (const mutation of mutations) {
        if (mutation.type === "characterData") {
          observeRootWhenVisible(mutation.target);
        }
        mutation.addedNodes.forEach(observeTextScopes);
        if (
          mutation.type === "childList" &&
          [...mutation.addedNodes, ...mutation.removedNodes].some(affectsCurrentTextRun)
        ) {
          observeRootWhenVisible(mutation.target);
        }
      }
      removeDisconnectedVisibilityTargets();
      if (pruneDisconnectedShadowRoots()) {
        state.observer.disconnect();
        observeRegisteredMutationTargets();
      }
    });
    observeRegisteredMutationTargets();
  }

  function stopObserver() {
    state.observer?.disconnect();
    state.observer = null;
    stopVisibilityObserver();
    cancelScheduledFlush();
    state.pendingRoots.clear();
  }

  function restoreDocument() {
    const queryRoots = registeredQueryRoots();
    stopObserver();
    const parents = new Set();

    queryRoots.forEach((root) => {
      root.querySelectorAll(".super-reader-chunk").forEach((span) => {
        if (span.parentNode) parents.add(span.parentNode);
        if (span.dataset.superReaderDivider === "true") {
          span.remove();
        } else {
          // Clean up text-wrapping spans created by earlier extension versions.
          span.replaceWith(document.createTextNode(span.textContent || ""));
        }
      });
    });

    parents.forEach((parent) => parent.normalize());
    state.shadowRoots.clear();
  }

  function enable() {
    if (state.enabled) return;
    state.enabled = true;
    startVisibilityObserver();
    startObserver();
    observeTextScopes(document.body || document.documentElement);
  }

  function disable() {
    state.enabled = false;
    restoreDocument();
  }

  function updateDividerStyles() {
    registeredQueryRoots().forEach((root) => {
      root.querySelectorAll(".super-reader-chunk").forEach((span) => {
        applyDividerStyle(span);
      });
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
