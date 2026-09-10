const test = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const vm = require("node:vm");

const context = vm.createContext({});
vm.runInContext(readFileSync(join(__dirname, "../src/frontend/visibility.js"), "utf8"), context);
const { createVisibilityFilter } = context.SuperReader;
const viewport = Object.freeze({ left: 0, top: 0, right: 100, bottom: 100 });
const box = (left, top, width, height) => ({
  left, top, width, height, right: left + width, bottom: top + height,
});

// Only element styles and boxes are needed. No tree traversal or text Range API.
function environment() {
  const styleReads = new Map();
  function element(parentElement = null, style = {}, rect = box(0, 0, 100, 100)) {
    return {
      parentElement, style,
      matches: () => false,
      getBoundingClientRect: () => rect,
      clientLeft: 0, clientTop: 0, clientWidth: rect.width, clientHeight: rect.height,
      offsetWidth: rect.width, offsetHeight: rect.height,
    };
  }
  const documentElement = element();
  const body = element(documentElement);
  const document = {
    documentElement, body,
    defaultView: {
      getComputedStyle(node) {
        styleReads.set(node, (styleReads.get(node) || 0) + 1);
        return {
          display: "block", visibility: "visible", opacity: "1",
          contentVisibility: "visible", overflowX: "visible", overflowY: "visible",
          ...node.style,
        };
      },
    },
  };
  return { document, body, element, styleReads, lookup: () => createVisibilityFilter(document, viewport).getVisibleArea };
}
const text = (parentElement, nodeValue = "中文") => ({ parentElement, nodeValue });

test("missing parents, empty text, and excluded subtrees need no style or geometry reads", () => {
  const page = environment();
  const parent = page.element(page.body);
  parent.matches = () => true;
  parent.getBoundingClientRect = () => assert.fail("excluded element should not be measured");
  const getVisibleArea = page.lookup();
  assert.equal(getVisibleArea(text(null)), null);
  assert.equal(getVisibleArea(text(page.body, "")), null);
  assert.equal(getVisibleArea(text(parent)), null);
  assert.equal(page.styleReads.size, 0);
});

test("nested ancestors clip their respective axes without changing the viewport", () => {
  const page = environment();
  const horizontal = page.element(page.body, { overflowX: "hidden" }, box(20, 0, 40, 100));
  const vertical = page.element(horizontal, { overflowY: "auto" }, box(0, 30, 100, 40));
  assert.deepEqual({ ...page.lookup()(text(vertical)) }, { left: 20, top: 30, right: 60, bottom: 70 });
  const outside = page.element(horizontal, { overflowX: "clip" }, box(80, 0, 10, 10));
  assert.equal(page.lookup()(text(outside)), null);
});

test("scaled client boxes include border offsets and intersect the viewport", () => {
  const page = environment();
  const parent = page.element(page.body, { overflowX: "auto", overflowY: "hidden" }, box(10, 20, 100, 80));
  Object.assign(parent, {
    offsetWidth: 50, offsetHeight: 40, clientLeft: 2, clientTop: 3, clientWidth: 44, clientHeight: 32,
  });
  assert.deepEqual({ ...page.lookup()(text(parent)) }, { left: 14, top: 26, right: 100, bottom: 90 });
});

test("inline, contents, and root boxes do not introduce overflow clipping", () => {
  const page = environment();
  Object.assign(page.body.style, { overflowX: "hidden", overflowY: "hidden" });
  page.body.clientWidth = 0;
  page.body.clientHeight = 0;
  for (const display of ["inline", "contents"]) {
    const parent = page.element(page.body, { display, overflowX: "hidden", overflowY: "hidden" }, box(0, 0, 0, 0));
    assert.deepEqual({ ...page.lookup()(text(parent)) }, viewport);
  }
});

test("visible descendants may override visibility but cannot override a hidden or transparent ancestor", () => {
  const page = environment();
  const hidden = page.element(page.body, { visibility: "hidden" });
  const visible = page.element(hidden, { visibility: "visible" });
  const getVisibleArea = page.lookup();
  assert.equal(getVisibleArea(text(hidden)), null);
  assert.deepEqual({ ...getVisibleArea(text(visible)) }, viewport);
  for (const style of [{ display: "none" }, { opacity: "0" }, { contentVisibility: "hidden" }]) {
    const ancestor = page.element(page.body, style);
    assert.equal(page.lookup()(text(page.element(ancestor))), null);
  }
});

test("visibility caches are shared within one read and refreshed for the next snapshot", () => {
  const page = environment();
  const parent = page.element(page.body);
  const firstRead = page.lookup();
  firstRead(text(parent, "第一段"));
  firstRead(text(parent, "第二段"));
  assert.equal(page.styleReads.get(parent), 1);
  parent.style.opacity = "0";
  assert.equal(page.lookup()(text(parent)), null);
  assert.equal(page.styleReads.get(parent), 2);
});
