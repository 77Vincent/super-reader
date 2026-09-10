(function defineDOMRead() {
  "use strict";

  globalThis.SuperReader ??= {};

  /**
   * @typedef {Object} TextSource
   * @property {Text} node Original text node; read() never splits it.
   * @property {Element} parent Original parent, used to validate a later write.
   * @property {string} text Original full node value, captured before processing.
   * @property {number} start Inclusive UTF-16 offset in the original node.
   * @property {number} end Exclusive UTF-16 offset in the original node.
   */

  /**
   * Capture whole unprocessed text nodes with any text visible, without DOM writes.
   * The viewport selects nodes; it does not limit their string lengths.
   * Visibility is geometric: viewport and rectangular overflow clipping, not
   * occlusion by overlays or CSS masks. Only the current document's light DOM is read.
   * @returns {ViewportSnapshot & {sources: TextSource[], viewport: Object}}
   */
  globalThis.SuperReader.read = function read() {
    const { readViewport, createVisibilityFilter, isProcessedText } = globalThis.SuperReader;
    const viewport = readViewport(document);
    const snapshot = { texts: [], sources: [], viewport };
    if (!document.body || viewport.right <= viewport.left || viewport.bottom <= viewport.top) return snapshot;

    const { shouldSkipSubtree, getVisibleArea } = createVisibilityFilter(document, viewport);
    // TreeWalker does not filter its root or the root's ancestors.
    for (let element = document.body; element; element = element.parentElement) {
      if (shouldSkipSubtree(element)) return snapshot;
    }
    const domRange = document.createRange();
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        if (node.nodeType === 1) {
          return shouldSkipSubtree(node) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_SKIP;
        }
        return NodeFilter.FILTER_ACCEPT;
      },
    });
    while (walker.nextNode()) {
      const node = walker.currentNode;
      if (isProcessedText(node)) continue;
      const text = node.nodeValue;
      if (!text || !/\p{Script=Han}/u.test(text)) continue;
      const visibleArea = getVisibleArea(node);
      if (!visibleArea) continue;
      domRange.selectNodeContents(node);
      // Check text fragments, so a gap between offscreen lines is not mistaken
      // for visible text. One intersecting fragment includes the whole node.
      for (const rect of domRange.getClientRects()) {
        if (rect.width <= 0 || rect.height <= 0 ||
            rect.right <= visibleArea.left || rect.left >= visibleArea.right ||
            rect.bottom <= visibleArea.top || rect.top >= visibleArea.bottom) continue;
        snapshot.texts.push(text);
        snapshot.sources.push({ node, parent: node.parentElement, text, start: 0, end: text.length });
        break;
      }
    }
    return snapshot;
  };
})();
