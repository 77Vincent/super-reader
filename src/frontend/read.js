(function defineDOMRead() {
  "use strict";

  globalThis.SuperReader ??= {};

  const SUPER_READER_SELECTOR = "super-reader-divider";

  /**
   * @typedef {Object} TextSource
   * @property {Text} node Original text node; read() never splits it.
   * @property {Element} parent Original parent, used to validate a later write.
   * @property {string} text Original full node value, captured before processing.
   * @property {number} start Inclusive UTF-16 offset in the original node.
   * @property {number} end Exclusive UTF-16 offset in the original node.
   */

  /**
   * Capture one viewport's eligible text without changing the DOM.
   * texts[i] is sources[i].text.slice(sources[i].start, sources[i].end).
   * Visibility is geometric: viewport and rectangular overflow clipping, not
   * occlusion by overlays or CSS masks. Only the current document's light DOM is read.
   * @returns {ViewportSnapshot & {sources: TextSource[], viewport: Object}}
   */
  globalThis.SuperReader.read = function read() {
    const view = document.defaultView;
    const visual = view.visualViewport;
    const left = visual?.offsetLeft ?? 0;
    const top = visual?.offsetTop ?? 0;
    const viewport = {
      left, top,
      right: left + (visual?.width ?? document.documentElement.clientWidth),
      bottom: top + (visual?.height ?? document.documentElement.clientHeight),
    };
    const snapshot = { texts: [], sources: [], viewport };
    if (!document.body || viewport.right <= left || viewport.bottom <= top) return snapshot;

    const ignored = [
      "script", "style", "noscript", "input", "textarea", "select", "button",
      "code", "pre", "kbd", "samp", "var", "svg", "math", "canvas", "iframe",
      "[hidden]", "[aria-hidden='true']", "[contenteditable]:not([contenteditable='false'])",
      `.${SUPER_READER_SELECTOR}`,
    ].join(",");
    const han = /\p{Script=Han}/u;
    const range = document.createRange();
    const styles = new Map();
    const clips = new Map();

    function styleFor(element) {
      if (!styles.has(element)) styles.set(element, view.getComputedStyle(element));
      return styles.get(element);
    }

    function clipFor(element) {
      if (!element) return viewport;
      if (clips.has(element)) return clips.get(element);
      const inherited = clipFor(element.parentElement);
      const style = styleFor(element);
      let clip = inherited && { ...inherited };
      if (style.display === "none" || style.contentVisibility === "hidden" || style.opacity === "0") {
        clip = null;
      }
      // The root scrolling area is already represented by the viewport.
      if (clip && element !== document.documentElement && element !== document.body &&
          style.display !== "inline" && style.display !== "contents") {
        const clipX = /^(hidden|clip|auto|scroll)$/.test(style.overflowX);
        const clipY = /^(hidden|clip|auto|scroll)$/.test(style.overflowY);
        if (clipX || clipY) {
          const box = element.getBoundingClientRect();
          // Client sizes are unscaled; range rectangles are in viewport coordinates.
          const scaleX = element.offsetWidth ? box.width / element.offsetWidth : 1;
          const scaleY = element.offsetHeight ? box.height / element.offsetHeight : 1;
          const x = box.left + element.clientLeft * scaleX;
          const y = box.top + element.clientTop * scaleY;
          if (clipX) {
            clip.left = Math.max(clip.left, x);
            clip.right = Math.min(clip.right, x + element.clientWidth * scaleX);
          }
          if (clipY) {
            clip.top = Math.max(clip.top, y);
            clip.bottom = Math.min(clip.bottom, y + element.clientHeight * scaleY);
          }
          if (clip.right <= clip.left || clip.bottom <= clip.top) clip = null;
        }
      }
      clips.set(element, clip);
      return clip;
    }

    function visibleRanges(node, text, clip) {
      const spans = [];
      function include(start, end) {
        const previous = spans[spans.length - 1];
        if (previous?.end === start) previous.end = end;
        else spans.push({ start, end });
      }
      function visit(start, end) {
        range.setStart(node, start);
        range.setEnd(node, end);
        const box = range.getBoundingClientRect();
        if (box.width <= 0 || box.height <= 0 ||
            box.right <= clip.left || box.left >= clip.right ||
            box.bottom <= clip.top || box.top >= clip.bottom) return;
        if (box.left >= clip.left && box.right <= clip.right &&
            box.top >= clip.top && box.bottom <= clip.bottom) {
          include(start, end);
          return;
        }

        // Narrow only ranges crossing an edge. Whole offscreen ranges are discarded
        // above, so long nodes do not require a JavaScript scan of every character.
        let middle = Math.floor((start + end) / 2);
        if (middle > start && /[\uD800-\uDBFF]/u.test(text[middle - 1]) &&
            /[\uDC00-\uDFFF]/u.test(text[middle])) middle += 1;
        if (middle === start || middle === end) {
          include(start, end); // Include a partly visible character as a whole.
          return;
        }
        visit(start, middle);
        visit(middle, end);
      }
      visit(0, text.length);
      return spans;
    }

    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    while (walker.nextNode()) {
      const node = walker.currentNode;
      const parent = node.parentElement;
      const text = node.nodeValue;
      if (!parent || !text || parent.closest(ignored)) continue;
      const style = styleFor(parent);
      // Check computed visibility on the text's parent: a descendant can override
      // visibility:hidden inherited from an ancestor.
      if (style.visibility === "hidden" || style.visibility === "collapse") continue;
      const clip = clipFor(parent);
      if (!clip) continue;
      for (const { start, end } of visibleRanges(node, text, clip)) {
        const input = text.slice(start, end);
        if (!han.test(input)) continue;
        snapshot.texts.push(input);
        snapshot.sources.push({ node, parent, text, start, end });
      }
    }
    return snapshot;
  };

})();
