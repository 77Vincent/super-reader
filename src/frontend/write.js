(function defineDOMWrite() {
  "use strict";

  globalThis.SuperReader ??= {};

  const SUPER_READER_SELECTOR = "super-reader-divider";

  function sourceIsCurrent(source) {
    return source.node.isConnected && source.node.parentNode === source.parent &&
      source.node.nodeValue === source.text;
  }

  // Adds markers to the document at the specified offsets within a text node.
  globalThis.SuperReader.addMarkers = function addMarkers(source, offsets) {
    // Offsets follow the processing contract; only the live DOM can become stale.
    if (!sourceIsCurrent(source)) return;

    for (const offset of [...offsets].reverse()) {
      const right = source.node.splitText(source.start + offset);
      const marker = document.createElement("span");
      marker.className = SUPER_READER_SELECTOR;
      marker.setAttribute("aria-hidden", "true");
      right.before(marker);
      globalThis.SuperReader.rememberProcessedText(right);
    }
    // Include the remaining left fragment and nodes requiring no markers.
    globalThis.SuperReader.rememberProcessedText(source.node);
  };

  /** Apply a complete result synchronously to the unchanged source nodes. */
  globalThis.SuperReader.write = function write(snapshot, results) {
    for (let index = snapshot.sources.length - 1; index >= 0; index -= 1) {
      globalThis.SuperReader.addMarkers(snapshot.sources[index], results[index]);
    }
  };

  // Removes all markers from the document and normalizes the parent nodes.
  globalThis.SuperReader.clearMarkers = function clearMarkers() {
    const parents = new Set();
    document.querySelectorAll(`.${SUPER_READER_SELECTOR}`).forEach((marker) => {
      parents.add(marker.parentNode);
      marker.remove();
    });
    parents.forEach((parent) => parent.normalize());
    globalThis.SuperReader.clearProcessedText();
  };
})();
