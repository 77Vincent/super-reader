(function initializeReader() {
  "use strict";
  if (globalThis.__superReaderContentLoaded) return;
  globalThis.__superReaderContentLoaded = true;

  // One request and its synchronous DOM update form one locked batch.
  const MAX_BATCH_LENGTH = 128; // UTF-16 units; also checked by the Worker.
  const BATCH_PAUSE_MS = 50; // Give toolbar clicks a turn between batches.
  const IGNORED = [
    "script", "style", "noscript", "input", "textarea", "select", "button",
    "code", "pre", "svg", "math", "canvas", "[hidden]", "[aria-hidden='true']",
    "[contenteditable]:not([contenteditable='false'])", ".super-reader-divider",
  ].join(",");
  let enabled = false;
  let busy = false;
  let batches = null;
  let timer = null;

  function status() {
    return { enabled, busy };
  }

  function publishStatus(error) {
    try {
      void chrome.runtime.sendMessage({
        type: "SUPER_READER_STATE", ...status(), error,
      }).catch(() => {});
    } catch (_) { /* An extension reload can invalidate this page's context. */ }
  }

  function collectTextNodes() {
    const nodes = [];
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      const node = walker.currentNode;
      const parent = node.parentElement;
      if (!parent || parent.closest(IGNORED) || !/\p{Script=Han}/u.test(node.nodeValue)) continue;
      const style = getComputedStyle(parent);
      if (!parent.getClientRects().length || style.visibility === "hidden") continue;
      nodes.push({ node, text: node.nodeValue, parent });
    }
    return nodes;
  }

  function* planBatches(nodes) {
    for (const source of nodes) {
      // Work backwards so splitting the text node leaves earlier offsets valid.
      for (let end = source.text.length; end > 0;) {
        let start = Math.max(0, end - MAX_BATCH_LENGTH);
        if (start > 0 && /[\uDC00-\uDFFF]/u.test(source.text[start])) start += 1;
        const text = source.text.slice(start, end);
        if (/\p{Script=Han}/u.test(text)) yield { ...source, start, end, input: text };
        end = start;
      }
    }
  }

  function sourceIsCurrent(batch) {
    return batch.node.isConnected && batch.node.parentNode === batch.parent &&
      batch.node.nodeValue.slice(0, batch.end) === batch.text.slice(0, batch.end);
  }

  function renderBatch(batch, offsets) {
    // Validate the whole result before making any DOM changes.
    if (!Array.isArray(offsets) || offsets.some((offset, index) => (
      !Number.isInteger(offset) || offset <= 0 || offset >= batch.input.length ||
      (index > 0 && offset <= offsets[index - 1])
    ))) throw new Error("Invalid divider offsets");
    if (!sourceIsCurrent(batch)) return;

    for (const offset of [...offsets].reverse()) {
      const right = batch.node.splitText(batch.start + offset);
      const marker = document.createElement("span");
      marker.className = "super-reader-divider";
      marker.setAttribute("aria-hidden", "true");
      right.before(marker);
    }
  }

  function clearPage() {
    clearTimeout(timer);
    timer = null;
    batches = null;
    const parents = new Set();
    document.querySelectorAll(".super-reader-divider").forEach((marker) => {
      parents.add(marker.parentNode);
      marker.remove();
    });
    parents.forEach((parent) => parent.normalize());
  }

  async function processNextBatch() {
    timer = null;
    if (!enabled) return;
    const next = batches.next();
    if (next.done) {
      batches = null;
      return;
    }
    busy = true;
    publishStatus();
    let error;
    try {
      if (sourceIsCurrent(next.value)) {
        const response = await chrome.runtime.sendMessage({
          type: "SUPER_READER_SPLIT_TEXTS", texts: [next.value.input],
        });
        if (response?.error) throw new Error(response.error);
        renderBatch(next.value, response?.offsetsByText?.[0]);
      }
    } catch (failure) {
      error = failure instanceof Error ? failure.message : String(failure);
      enabled = false;
      clearPage();
      console.error("Super Reader stopped", failure);
    } finally {
      busy = false;
      publishStatus(error);
    }
    if (enabled) timer = setTimeout(processNextBatch, BATCH_PAUSE_MS);
  }

  function toggle() {
    if (busy) return status();
    if (enabled) {
      enabled = false;
      clearPage();
    } else {
      batches = planBatches(collectTextNodes());
      enabled = true;
      timer = setTimeout(processNextBatch, BATCH_PAUSE_MS);
    }
    return status();
  }

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message?.type === "SUPER_READER_PING") sendResponse(status());
    if (message?.type === "SUPER_READER_TOGGLE") sendResponse(toggle());
  });
})();
