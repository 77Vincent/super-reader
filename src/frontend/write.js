(function defineDOMWrite() {
  "use strict";

  globalThis.SuperReader ??= {};
  if (globalThis.SuperReader.write) return;

  const SUPER_READER_SELECTOR = "super-reader-divider";
  const markers = new Set();
  const markedSources = new Set();
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
    const sequence = [source.node];
    const sourceMarkers = [];
    if (offsets.length > 0) ensureShadowStyle(source.node.getRootNode(), styleUrl);

    for (const offset of [...offsets].reverse()) {
      const right = source.node.splitText(source.start + offset);
      const marker = document.createElement("span");
      marker.className = SUPER_READER_SELECTOR;
      marker.setAttribute("aria-hidden", "true");
      right.before(marker);
      markers.add(marker);
      sourceMarkers.push(marker);
      sequence.splice(1, 0, marker, right);
      written.push(right);
    }
    if (sourceMarkers.length) markedSources.add({
      parent: source.parent,
      markers: sourceMarkers,
      sequence: sequence.map((node) => ({ node, text: node.nodeValue })),
    });
    return written;
  };

  /** Remove obsolete splits before reading page edits; preserve the page's current text. */
  globalThis.SuperReader.removeStaleMarkers = function removeStaleMarkers() {
    const parents = new Set();
    for (const source of markedSources) {
      const current = source.sequence.every(({ node, text }, index, sequence) => (
        node.isConnected && node.parentNode === source.parent && node.nodeValue === text &&
        (index === sequence.length - 1 || node.nextSibling === sequence[index + 1].node)
      ));
      if (current) continue;
      parents.add(source.parent);
      for (const marker of source.markers) {
        if (marker.parentNode) parents.add(marker.parentNode);
      }
    }
    const forgotten = [];
    // Clear all splits in an affected parent before normalize() rejoins its text.
    // Other parents keep their markers and processed records.
    for (const source of markedSources) {
      if (!parents.has(source.parent)) continue;
      for (const marker of source.markers) {
        marker.remove();
        markers.delete(marker);
      }
      forgotten.push(...source.sequence.filter(({ node }) => node.nodeType === 3).map(({ node }) => node));
      markedSources.delete(source);
    }
    for (const parent of parents) {
      forgotten.push(...Array.from(parent.childNodes).filter((node) => node.nodeType === 3));
      parent.normalize();
    }
    return forgotten;
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
    markedSources.clear();
    parents.forEach((parent) => parent.normalize());
    shadowStyles.forEach((link) => link.remove());
    shadowStyles.clear();
  };
})();
