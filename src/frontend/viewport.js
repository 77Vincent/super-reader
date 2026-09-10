(function defineViewport() {
  "use strict";

  globalThis.SuperReader ??= {};

  /** Capture viewport bounds as plain numbers, independent of subsequent changes. */
  globalThis.SuperReader.readViewport = function readViewport(document) {
    const view = document.defaultView;
    const visual = view.visualViewport;
    const left = visual?.offsetLeft ?? 0;
    const top = visual?.offsetTop ?? 0;
    return {
      left, top,
      right: left + (visual?.width ?? document.documentElement.clientWidth),
      bottom: top + (visual?.height ?? document.documentElement.clientHeight),
    };
  };

  /** Notify changes immediately; the reader owns debouncing and scheduling. */
  globalThis.SuperReader.watchViewport = function watchViewport(document, onChange) {
    const view = document.defaultView;
    const visual = view.visualViewport;
    // Capture also receives scroll events from nested scrolling containers.
    view.addEventListener("scroll", onChange, { capture: true, passive: true });
    view.addEventListener("resize", onChange);
    visual?.addEventListener("scroll", onChange, { passive: true });
    visual?.addEventListener("resize", onChange);

    return () => {
      view.removeEventListener("scroll", onChange, true);
      view.removeEventListener("resize", onChange);
      visual?.removeEventListener("scroll", onChange);
      visual?.removeEventListener("resize", onChange);
    };
  };
})();
