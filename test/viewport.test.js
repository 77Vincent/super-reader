const test = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const vm = require("node:vm");

const context = vm.createContext({});
vm.runInContext(readFileSync(join(__dirname, "../src/frontend/viewport.js"), "utf8"), context);
const { readViewport } = context.SuperReader;

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
