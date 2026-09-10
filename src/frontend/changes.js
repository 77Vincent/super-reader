(function defineDOMChanges() {
  "use strict";

  globalThis.SuperReader ??= {};

  /** Observe content and shadow-root changes; scheduling belongs to the reader. */
  globalThis.SuperReader.createDOMChanges = function createDOMChanges(document) {
    const { walkDOM, parentElementAcrossRoots } = globalThis.SuperReader;
    const roots = new Set();
    const options = {
      subtree: true, childList: true, characterData: true, attributes: true,
      attributeFilter: ["class", "style", "hidden", "aria-hidden", "contenteditable", "slot"],
    };
    const ignored = "script,style,link,noscript,code,pre,input,textarea,select,button,[contenteditable]:not([contenteditable='false'])";
    let observer = null;
    let notify = null;

    function ignoreContent(node) {
      let element = node.nodeType === 1 ? node : parentElementAcrossRoots(node);
      while (element) {
        if (element.matches(ignored)) return true;
        element = parentElementAcrossRoots(element);
      }
      return false;
    }

    function observeRoot(root) {
      if (!observer || roots.has(root) || (root.host && !root.host.isConnected)) return;
      roots.add(root);
      observer.observe(root, options);
      if (root.host) {
        root.addEventListener("scroll", changed, { capture: true, passive: true });
        root.addEventListener("slotchange", changed);
      }
    }

    function changed() { notify?.(); }

    function hasTextOrRoot(node) {
      if (ignoreContent(node)) return false;
      let relevant = false;
      for (const child of walkDOM(node, (element) => element.matches(ignored))) {
        if (child.nodeType === 11 && child.host) {
          observeRoot(child);
          relevant = true;
        }
        if (child.nodeType === 3 && /\p{Script=Han}/u.test(child.nodeValue)) relevant = true;
      }
      return relevant;
    }

    function handleMutations(records) {
      let relevant = false;
      for (const record of records) {
        if (ignoreContent(record.target)) continue;
        if (record.type === "childList") {
          for (const node of [...record.addedNodes, ...record.removedNodes]) {
            if (hasTextOrRoot(node)) relevant = true;
          }
        } else if (hasTextOrRoot(record.target)) {
          relevant = true;
        }
      }
      // MutationObserver has no unobserve(root); reconnect the remaining roots.
      if ([...roots].some((root) => root.host && !root.host.isConnected)) {
        observer.disconnect();
        reconnect();
      }
      if (relevant) changed();
    }

    function removeRoot(root) {
      if (root.host) {
        root.removeEventListener("scroll", changed, { capture: true });
        root.removeEventListener("slotchange", changed);
      }
      roots.delete(root);
    }

    function reconnect() {
      for (const root of roots) {
        if (root.host && !root.host.isConnected) removeRoot(root);
        else observer.observe(root, options);
      }
    }

    function watch(onChange) {
      notify = onChange;
      observer = new document.defaultView.MutationObserver(handleMutations);
      observeRoot(document.documentElement);
      return () => {
        observer.disconnect();
        observer = null;
        notify = null;
        for (const root of roots) removeRoot(root);
      };
    }

    /** Suppress only our synchronous DOM writes, preserving queued page changes. */
    function mutate(operation) {
      if (!observer) return operation();
      handleMutations(observer.takeRecords());
      observer.disconnect();
      try { return operation(); }
      finally { reconnect(); }
    }

    return { watch, observeRoot, mutate };
  };
})();
