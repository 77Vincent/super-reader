(function defineVisibilityFilter() {
  "use strict";

  globalThis.SuperReader ??= {};

  /**
   * Share eligibility checks and cached styles for one synchronous read.
   * shouldSkipSubtree(element) excludes whole branches during traversal.
   * getVisibleArea(node) returns the viewport restricted by ancestor overflow,
   * or null for hidden text. It does not measure the text itself.
   * @param {Document} document
   * @param {{left: number, top: number, right: number, bottom: number}} viewport
   */
  globalThis.SuperReader.createVisibilityFilter = function createVisibilityFilter(document, viewport) {
    const view = document.defaultView;
    const ignored = [
      "script", "style", "noscript", "input", "textarea", "select", "button",
      "code", "pre", "kbd", "samp", "var", "svg", "math", "canvas", "iframe",
      "[hidden]", "[aria-hidden='true']", "[contenteditable]:not([contenteditable='false'])",
      ".super-reader-divider",
    ].join(",");
    const styleCache = new Map();
    const visibleAreaCache = new Map();

    function getElementStyle(element) {
      if (!styleCache.has(element)) styleCache.set(element, view.getComputedStyle(element));
      return styleCache.get(element);
    }

    function shouldSkipSubtree(element) {
      if (element.matches(ignored)) return true;
      const style = getElementStyle(element);
      // visibility:hidden is overridable by descendants; do not prune it.
      return style.display === "none" || style.contentVisibility === "hidden" || style.opacity === "0";
    }

    function getElementVisibleArea(element) {
      if (!element) return viewport;
      if (visibleAreaCache.has(element)) return visibleAreaCache.get(element);
      if (shouldSkipSubtree(element)) {
        visibleAreaCache.set(element, null);
        return null;
      }
      const parentVisibleArea = getElementVisibleArea(element.parentElement);
      const style = getElementStyle(element);
      let visibleArea = parentVisibleArea && { ...parentVisibleArea };
      // The root scrolling area is already represented by the viewport.
      if (visibleArea && element !== document.documentElement && element !== document.body &&
          style.display !== "inline" && style.display !== "contents") {
        const clipsHorizontally = /^(hidden|clip|auto|scroll)$/.test(style.overflowX);
        const clipsVertically = /^(hidden|clip|auto|scroll)$/.test(style.overflowY);
        if (clipsHorizontally || clipsVertically) {
          const elementRect = element.getBoundingClientRect();
          // Client sizes are unscaled; range rectangles are in viewport coordinates.
          const scaleX = element.offsetWidth ? elementRect.width / element.offsetWidth : 1;
          const scaleY = element.offsetHeight ? elementRect.height / element.offsetHeight : 1;
          const contentLeft = elementRect.left + element.clientLeft * scaleX;
          const contentTop = elementRect.top + element.clientTop * scaleY;
          if (clipsHorizontally) {
            visibleArea.left = Math.max(visibleArea.left, contentLeft);
            visibleArea.right = Math.min(visibleArea.right, contentLeft + element.clientWidth * scaleX);
          }
          if (clipsVertically) {
            visibleArea.top = Math.max(visibleArea.top, contentTop);
            visibleArea.bottom = Math.min(visibleArea.bottom, contentTop + element.clientHeight * scaleY);
          }
          if (visibleArea.right <= visibleArea.left || visibleArea.bottom <= visibleArea.top) visibleArea = null;
        }
      }
      visibleAreaCache.set(element, visibleArea);
      return visibleArea;
    }

    function getVisibleArea(node) {
      const parent = node.parentElement;
      const text = node.nodeValue;
      if (!parent || !text) return null;
      const visibleArea = getElementVisibleArea(parent);
      if (!visibleArea) return null;
      const style = getElementStyle(parent);
      // Check computed visibility on the text's parent: a descendant can override
      // visibility:hidden inherited from an ancestor.
      if (style.visibility === "hidden" || style.visibility === "collapse") return null;
      return visibleArea;
    }

    return { shouldSkipSubtree, getVisibleArea };
  };
})();
