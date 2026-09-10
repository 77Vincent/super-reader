(function defineDOMTree() {
  "use strict";

  globalThis.SuperReader ??= {};

  /** Follow the rendered ancestry through slots and shadow hosts. */
  globalThis.SuperReader.parentElementAcrossRoots = function parentElementAcrossRoots(node) {
    const parent = node.assignedSlot || node.parentElement;
    if (parent) return parent;
    // Only a shadow root supplies a host element. A link's .host is a URL string.
    const root = node.nodeType === 11 ? node : node.parentNode;
    return root?.nodeType === 11 ? root.host || null : null;
  };

  /** Walk ordinary children and nested open shadow roots, visiting each node once. */
  globalThis.SuperReader.walkDOM = function* walkDOM(root, shouldSkipSubtree = () => false) {
    if (root.nodeType === 1 && shouldSkipSubtree(root)) return;
    yield root;
    if (root.nodeType === 3) return;
    if (root.shadowRoot) yield* walkDOM(root.shadowRoot, shouldSkipSubtree);
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        return node.nodeType === 1 && shouldSkipSubtree(node)
          ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT;
      },
    });
    while (walker.nextNode()) {
      const node = walker.currentNode;
      yield node;
      if (node.shadowRoot) yield* walkDOM(node.shadowRoot, shouldSkipSubtree);
    }
  };
})();
