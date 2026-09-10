(function defineDOMWrite() {
  "use strict";

  globalThis.SuperReader ??= {};
  if (globalThis.SuperReader.write) return;

  const SUPER_READER_SELECTOR = "super-reader-divider";
  const markers = new Set();
  const shadowStyles = new Map();

  function ensureShadowStyle(root, styleUrl) {
    if (!root.host) return;
    if (shadowStyles.get(root)?.parentNode === root) return;
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = styleUrl;
    root.append(link);
    shadowStyles.set(root, link);
  }

  function sourceIsCurrent(source) {
    return source.node.isConnected && source.node.parentNode === source.parent &&
      source.node.nodeValue === source.text;
  }

  // Adds markers to the document at the specified offsets within a text node.
  globalThis.SuperReader.addMarkers = function addMarkers(source, offsets, styleUrl) {
    // Offsets follow the processing contract; only the live DOM can become stale.
    if (!sourceIsCurrent(source)) return [];
    const written = [source.node];
    if (offsets.length > 0) ensureShadowStyle(source.node.getRootNode(), styleUrl);

    for (const offset of [...offsets].reverse()) {
      const right = source.node.splitText(source.start + offset);
      const marker = document.createElement("span");
      marker.className = SUPER_READER_SELECTOR;
      marker.setAttribute("aria-hidden", "true");
      right.before(marker);
      markers.add(marker);
      written.push(right);
    }
    return written;
  };

  /** Apply a complete result synchronously to the unchanged source nodes. */
  globalThis.SuperReader.write = function write(snapshot, results, styleUrl) {
    const written = [];
    for (let index = snapshot.sources.length - 1; index >= 0; index -= 1) {
      written.push(...globalThis.SuperReader.addMarkers(snapshot.sources[index], results[index], styleUrl));
    }
    return written;
  };

  // Removes all markers from the document and normalizes the parent nodes.
  globalThis.SuperReader.clearMarkers = function clearMarkers() {
    const parents = new Set();
    markers.forEach((marker) => {
      if (marker.parentNode) parents.add(marker.parentNode);
      marker.remove();
    });
    markers.clear();
    parents.forEach((parent) => parent.normalize());
    shadowStyles.forEach((link) => link.remove());
    shadowStyles.clear();
  };
})();
