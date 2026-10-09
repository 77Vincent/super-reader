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
    const nextSibling = source.node.nextSibling;
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
      nextSibling,
      markers: sourceMarkers,
      sequence: sequence.map((node) => ({ node, text: node.nodeValue, dirty: false, moved: false, reinserted: false })),
    });
    return written;
  };

  function sourceIsIntact(source) {
    return source.sequence.every(({ node, text, dirty, moved }, index, sequence) => (
      !dirty && !moved && node.isConnected && node.parentNode === source.parent && node.nodeValue === text &&
      (index === sequence.length - 1 || node.nextSibling === sequence[index + 1].node)
    ));
  }

  /** Page writes count even when they assign the same fragment value. */
  globalThis.SuperReader.markDirtySources = function markDirtySources(records) {
    if (!records.length) return false;
    const edited = new Set(records.filter((record) => record.type === "characterData").map((record) => record.target));
    const removed = new Set(records.flatMap((record) => Array.from(record.removedNodes || [])));
    const added = new Set(records.flatMap((record) => Array.from(record.addedNodes || [])));
    let changed = false;
    for (const source of markedSources) {
      for (const entry of source.sequence) {
        if (entry.node.nodeType !== 3) continue;
        if (edited.has(entry.node)) entry.dirty = true;
        if (removed.has(entry.node)) entry.moved = true;
        if (added.has(entry.node)) entry.reinserted = true;
      }
      if (!sourceIsIntact(source)) changed = true;
    }
    return changed;
  };

  // Missing or relocated markers do not prevent restoring an intact text group.
  // Foreign nodes between its text fragments do: never merge across page edits.
  function textsAreTogether(source, entries) {
    const ownMarkers = new Set(source.markers);
    return entries.every(({ node }, index) => {
      if (node.parentNode !== source.parent) return false;
      if (index === entries.length - 1) return true;
      let next = node.nextSibling;
      while (ownMarkers.has(next)) next = next.nextSibling;
      return next === entries[index + 1].node;
    });
  }

  function removeGroupMarkers(source) {
    for (const marker of source.markers) {
      marker.remove();
      markers.delete(marker);
    }
    markedSources.delete(source);
  }

  /** Drop only the nodes created by this write; the page owns the anchor. */
  function discard(source, texts) {
    for (const { node } of texts.slice(1)) node.remove();
    removeGroupMarkers(source);
  }

  /** Invert an unchanged write, retaining the original Text object's identity. */
  function unwrite(source, texts) {
    texts[0].node.nodeValue = texts.map(({ text }) => text).join("");
    discard(source, texts);
  }

  // Some page renderers retain their original Text object and prune injected
  // siblings. Recover only our untouched, detached fragments, without merging
  // across replacement nodes or undoing a page edit/move of the original text.
  function canRestoreRemovedFragments(source, texts) {
    const [anchor, ...fragments] = texts;
    if (anchor.dirty || anchor.moved || anchor.node.parentNode !== source.parent ||
        anchor.node.nodeValue !== anchor.text) return false;
    if (!fragments.some(({ node }) => node.parentNode === null)) return false;
    if (!fragments.every(({ node, text, dirty, moved, reinserted }) => (
      !dirty && !reinserted && node.nodeValue === text &&
      (node.parentNode === null || (!moved && node.parentNode === source.parent))
    ))) return false;

    const remaining = texts.filter(({ node }) => node.parentNode === source.parent);
    if (!textsAreTogether(source, remaining)) return false;
    // A new sibling in the old group's place may already replace the removed
    // text. Require the original end boundary to avoid restoring it twice.
    const ownMarkers = new Set(source.markers);
    let next = remaining[remaining.length - 1].node.nextSibling;
    while (ownMarkers.has(next)) next = next.nextSibling;
    return next === source.nextSibling;
  }

  function releaseSource(source) {
    const texts = source.sequence.filter(({ node }) => node.nodeType === 3);
    const [anchor, ...fragments] = texts;
    const fragmentsUntouched = fragments.every(({ node, text, dirty, moved }) => (
      !dirty && !moved && node.nodeValue === text && node.parentNode === source.parent
    ));
    const anchorInPlace = !anchor.moved && anchor.node.parentNode === source.parent;
    const anchorDirty = anchor.dirty || anchor.node.nodeValue !== anchor.text;
    let preserved = [];

    if (canRestoreRemovedFragments(source, texts)) {
      unwrite(source, texts);
      // Keep the restored source intact until the page changes it again.
      if (anchor.node.isConnected) preserved = [anchor.node];
    } else if (fragmentsUntouched && anchorInPlace && textsAreTogether(source, texts)) {
      // Contract: a page write to the original anchor replaces its whole source.
      if (anchorDirty) discard(source, texts);
      else unwrite(source, texts);
    } else if (fragmentsUntouched && !anchor.node.parentNode && textsAreTogether(source, fragments)) {
      discard(source, texts);
    } else {
      // The page has adopted fragments or changed their structure. Keep its DOM
      // and pause these text nodes until their value or parent changes again.
      removeGroupMarkers(source);
      preserved = texts.map(({ node }) => node).filter((node) => node.isConnected);
    }
    return { forgotten: texts.map(({ node }) => node), preserved };
  }

  /** Reconcile obsolete groups before reading, without normalizing page parents. */
  globalThis.SuperReader.removeStaleMarkers = function removeStaleMarkers() {
    const result = { forgotten: [], preserved: [] };
    for (const source of markedSources) {
      if (sourceIsIntact(source)) continue;
      const released = releaseSource(source);
      result.forgotten.push(...released.forgotten);
      result.preserved.push(...released.preserved);
    }
    return result;
  };

  /** Apply a complete result synchronously to the unchanged source nodes. */
  globalThis.SuperReader.write = function write(snapshot, results, styleUrl) {
    const written = [];
    for (let index = snapshot.sources.length - 1; index >= 0; index -= 1) {
      written.push(...globalThis.SuperReader.addMarkers(snapshot.sources[index], results[index], styleUrl));
    }
    return written;
  };

  // Use the same group rules on shutdown, including roots detached by the page.
  globalThis.SuperReader.clearMarkers = function clearMarkers() {
    for (const source of markedSources) releaseSource(source);
    markers.forEach((marker) => marker.remove());
    markers.clear();
    shadowStyles.forEach((link) => link.remove());
    shadowStyles.clear();
  };
})();
