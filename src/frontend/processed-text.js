(function defineProcessedText() {
  "use strict";

  globalThis.SuperReader ??= {};
  if (globalThis.SuperReader.isProcessedText) return;

  // Weak keys allow removed text nodes to be collected. No page markup is added.
  let processedNodes = new WeakMap();

  globalThis.SuperReader.isProcessedText = function isProcessedText(node) {
    const previous = processedNodes.get(node);
    return previous !== undefined && previous.text === node.nodeValue && previous.parent === node.parentNode;
  };

  globalThis.SuperReader.rememberProcessedText = function rememberProcessedText(node) {
    processedNodes.set(node, { text: node.nodeValue, parent: node.parentNode });
  };

  globalThis.SuperReader.clearProcessedText = function clearProcessedText() {
    processedNodes = new WeakMap();
  };
})();
