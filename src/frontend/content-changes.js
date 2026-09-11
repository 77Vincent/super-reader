(function defineContentChanges() {
  "use strict";

  globalThis.SuperReader ??= {};

  /** Detect page text replacement; scheduling and visibility belong to the reader. */
  globalThis.SuperReader.createContentChanges = function createContentChanges(document) {
    const { walkDOM, parentElementAcrossRoots } = globalThis.SuperReader;
    const ignored = "script,style,link,noscript,input,textarea,select,button,code,pre,[contenteditable]:not([contenteditable='false']),.super-reader-divider";
    const options = { subtree: true, childList: true, characterData: true };
    const roots = new Set();
    let observer = null;
    let notify = null;

    function isIgnored(node) {
      // Only Elements have matches(); detached links must not supply URL hosts.
      for (let element = node.nodeType === 1 ? node : parentElementAcrossRoots(node);
        element; element = parentElementAcrossRoots(element)) {
        if (element.matches(ignored)) return true;
      }
      return false;
    }

    function hasContent(node) {
      for (const child of walkDOM(node, (element) => element.matches(ignored))) {
        // An added open root may contain comments not present in textContent.
        if (child.nodeType === 11 ||
            (child.nodeType === 3 && /\p{Script=Han}/u.test(child.nodeValue))) return true;
      }
      return false;
    }

    function reconnect() {
      for (const root of roots) {
        if (root.isConnected) observer.observe(root, options);
        else roots.delete(root);
      }
    }

    function handleChanges(records) {
      const changed = records.some((record) => !isIgnored(record.target) && (
        record.type === "characterData" ? hasContent(record.target) :
          [...record.addedNodes, ...record.removedNodes].some(hasContent)
      ));
      // Release removed comment roots instead of retaining them for the session.
      if ([...roots].some((root) => !root.isConnected)) {
        observer.disconnect();
        reconnect();
      }
      if (changed) notify();
    }

    function observeRoot(root) {
      if (!observer || roots.has(root)) return;
      roots.add(root);
      observer.observe(root, options);
    }

    function watch(onChange) {
      notify = onChange;
      observer = new document.defaultView.MutationObserver(handleChanges);
      observeRoot(document.documentElement);
      return () => {
        observer.disconnect();
        observer = null;
        notify = null;
        roots.clear();
      };
    }

    /** Exclude only synchronous marker writes, keeping already queued page changes. */
    function withoutObservation(write) {
      if (!observer) return write();
      handleChanges(observer.takeRecords());
      observer.disconnect();
      try { return write(); }
      finally { reconnect(); }
    }

    return { watch, observeRoot, withoutObservation };
  };
})();
