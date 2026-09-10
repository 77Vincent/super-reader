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
})();
