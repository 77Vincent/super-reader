const test = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const vm = require("node:vm");

const context = vm.createContext({});
vm.runInContext(readFileSync(join(__dirname, "../src/frontend/viewport.js"), "utf8"), context);
const { readViewport, watchViewport } = context.SuperReader;

test("viewport capture uses visual viewport offsets and preserves the captured bounds", () => {
  const visualViewport = { offsetLeft: 12, offsetTop: 34, width: 200, height: 300 };
  const document = { defaultView: { visualViewport } };
  const first = readViewport(document);
  visualViewport.offsetLeft = 25;
  visualViewport.width = 100;
  assert.deepEqual({ ...first }, { left: 12, top: 34, right: 212, bottom: 334 });
  assert.deepEqual({ ...readViewport(document) }, { left: 25, top: 34, right: 125, bottom: 334 });
});

test("without a visual viewport, capture uses document client dimensions, including zero", () => {
  const document = { defaultView: {}, documentElement: { clientWidth: 640, clientHeight: 480 } };
  assert.deepEqual({ ...readViewport(document) }, { left: 0, top: 0, right: 640, bottom: 480 });
  document.documentElement.clientHeight = 0;
  assert.equal(readViewport(document).bottom, 0);
});

function eventTarget() {
  const listeners = [];
  return {
    listeners,
    addEventListener(type, callback, options) {
      listeners.push({ type, callback, capture: options?.capture === true, passive: options?.passive === true });
    },
    removeEventListener(type, callback, options) {
      const capture = options === true || options?.capture === true;
      const index = listeners.findIndex((entry) => entry.type === type && entry.callback === callback && entry.capture === capture);
      if (index !== -1) listeners.splice(index, 1);
    },
    emit(type) { listeners.filter((entry) => entry.type === type).forEach(({ callback }) => callback()); },
  };
}

for (const hasVisualViewport of [false, true]) {
  test(`viewport subscriptions capture scrolling, report changes immediately, and unsubscribe (visual viewport: ${hasVisualViewport})`, () => {
    const view = eventTarget();
    if (hasVisualViewport) view.visualViewport = eventTarget();
    let changes = 0;
    const stop = watchViewport({ defaultView: view }, () => { changes += 1; });
    const scroll = view.listeners.find(({ type }) => type === "scroll");
    assert.equal(scroll.capture, true, "nested scrolling must reach the window listener");
    assert.equal(scroll.passive, true);
    view.emit("scroll");
    view.emit("resize");
    assert.equal(changes, 2);
    view.visualViewport?.emit("scroll");
    view.visualViewport?.emit("resize");
    const expected = hasVisualViewport ? 4 : 2;
    assert.equal(changes, expected);
    stop();
    assert.equal(view.listeners.length, 0);
    assert.equal(view.visualViewport?.listeners.length ?? 0, 0);
    view.emit("scroll");
    view.emit("resize");
    view.visualViewport?.emit("scroll");
    view.visualViewport?.emit("resize");
    assert.equal(changes, expected);
  });
}
