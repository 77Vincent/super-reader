const test = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const vm = require("node:vm");

const scripts = ["viewport.js", "visibility.js", "processed-text.js", "read.js"].map((name) => {
  const filename = join(__dirname, "../src/frontend", name);
  return { filename, source: readFileSync(filename, "utf8") };
});
const box = (left, top, width, height) => ({
  left, top, width, height, right: left + width, bottom: top + height,
});
function cells(rectangles) {
  return (start, end) => rectangles.slice(start, end).filter(Boolean);
}

// Only layout geometry is supplied by the test. Browser fixtures separately check
// native Range measurements and CSS; this harness needs no DOM library or model.
function createPage() {
  const measurements = [];
  const visited = [];
  function element(tag = "p", options = {}) {
    const rect = options.rect || box(0, 0, 100, 100);
    const node = {
      nodeType: 1, tag, parentElement: options.parent || null, childNodes: [],
      style: options.style || {}, attributes: options.attributes || {},
      className: options.className || "", clientLeft: 0, clientTop: 0,
      clientWidth: rect.width, clientHeight: rect.height,
      getBoundingClientRect: () => rect,
      matches(selector) {
        return selector.split(",").some((part) => {
          if (part === this.tag) return true;
          if (part.startsWith(".")) return this.className.split(" ").includes(part.slice(1));
          if (part === "[hidden]") return "hidden" in this.attributes;
          if (part === "[aria-hidden='true']") return this.attributes["aria-hidden"] === "true";
          return part === "[contenteditable]:not([contenteditable='false'])" &&
            "contenteditable" in this.attributes && this.attributes.contenteditable !== "false";
        });
      },
    };
    node.parentElement?.childNodes.push(node);
    return node;
  }
  const root = element("html");
  const body = element("body", { parent: root });
  function computedStyle(node) {
    return {
      display: "block", visibility: node.parentElement ? computedStyle(node.parentElement).visibility : "visible",
      opacity: "1", contentVisibility: "visible", overflowX: "visible", overflowY: "visible",
      ...node.style,
    };
  }
  const view = { getComputedStyle: computedStyle };
  const document = {
    body, documentElement: root, defaultView: view,
    createTreeWalker(root, whatToShow, filter) {
      function* visit(parent) {
        for (const node of parent.childNodes) {
          visited.push(node);
          const result = ((1 << (node.nodeType - 1)) & whatToShow) ? filter.acceptNode(node) : 3;
          if (result === 2) continue;
          if (result === 1) yield node;
          if (node.nodeType === 1) yield* visit(node);
        }
      }
      const iterator = visit(root);
      return { nextNode() { this.currentNode = iterator.next().value; return this.currentNode || null; } };
    },
    createRange() {
      let node;
      return {
        selectNodeContents(source) { node = source; },
        getClientRects() {
          measurements.push({ node, start: 0, end: node.nodeValue.length });
          const rectangles = node.geometry(0, node.nodeValue.length);
          return Array.isArray(rectangles) ? rectangles : [rectangles];
        },
      };
    },
  };
  const context = vm.createContext({ document, NodeFilter: { SHOW_ELEMENT: 1, SHOW_TEXT: 4, FILTER_ACCEPT: 1, FILTER_REJECT: 2, FILTER_SKIP: 3 } });
  for (const { filename, source } of scripts) vm.runInContext(source, context, { filename });
  return {
    document, view, body, root, measurements, visited,
    read: context.SuperReader.read,
    element: (tag, options = {}) => element(tag, { parent: body, ...options }),
    text(value, geometry = () => box(0, 0, 40, 20), parent = body) {
      const node = { nodeType: 3, nodeValue: value, parentElement: parent, parentNode: parent, geometry };
      parent.childNodes.push(node);
      return node;
    },
  };
}

test("read returns an empty snapshot before body exists or when the viewport is empty", () => {
  const page = createPage();
  page.document.body = null;
  assert.deepEqual(Array.from(page.read().texts), []);
  page.document.body = page.body;
  page.root.clientHeight = 0;
  page.text("这里有中文");
  assert.deepEqual(Array.from(page.read().sources), []);
  assert.equal(page.measurements.length, 0);
});

test("read preserves DOM order, text, inline nodes, and source mapping without DOM writes", () => {
  const page = createPage();
  const first = page.text("第一段中文");
  const strong = page.element("strong");
  const second = page.text("加粗的中文", undefined, strong);
  page.text("English only");
  page.text("");
  const originalChildren = [...page.body.childNodes];
  const snapshot = page.read();

  assert.deepEqual(Array.from(snapshot.texts), [first.nodeValue, second.nodeValue]);
  for (const [index, node] of [first, second].entries()) {
    assert.deepEqual({ ...snapshot.sources[index] }, {
      node, parent: node.parentElement, text: node.nodeValue, start: 0, end: node.nodeValue.length,
    });
  }
  assert.deepEqual(page.body.childNodes, originalChildren);
  assert.equal(first.nodeValue, "第一段中文");
});

test("read excludes code, controls, editable and hidden subtrees, and existing markers", () => {
  const page = createPage();
  for (const tag of ["script", "style", "noscript", "input", "textarea", "select", "button",
    "code", "pre", "kbd", "samp", "var", "svg", "math", "canvas", "iframe"]) {
    const parent = page.element(tag);
    page.text("不应被读取", undefined, page.element("span", { parent }));
  }
  for (const options of [
    { attributes: { hidden: "" } }, { attributes: { "aria-hidden": "true" } },
    { attributes: { contenteditable: "" } }, { attributes: { contenteditable: "plaintext-only" } },
    { className: "super-reader-divider" },
    { style: { display: "none" } }, { style: { visibility: "hidden" } },
    { style: { visibility: "collapse" } }, { style: { opacity: "0" } },
    { style: { contentVisibility: "hidden" } },
  ]) {
    page.text("不应被读取", undefined, page.element("div", options));
  }
  page.text("可以读取", undefined, page.element("div", { attributes: { contenteditable: "false" } }));
  assert.deepEqual(Array.from(page.read().texts), ["可以读取"]);
});

test("visibility overrides and display:contents work while ancestor opacity still hides text", () => {
  const page = createPage();
  const hidden = page.element("div", { style: { visibility: "hidden" } });
  page.text("仍然隐藏", undefined, hidden);
  page.text("重新可见", undefined, page.element("span", { parent: hidden, style: { visibility: "visible" } }));
  page.text("没有父级盒子", undefined, page.element("span", { style: { display: "contents" }, rect: box(0, 0, 0, 0) }));
  const transparent = page.element("div", { style: { opacity: "0" } });
  page.text("依旧透明", undefined, page.element("span", { parent: transparent, style: { opacity: "1" } }));
  assert.deepEqual(Array.from(page.read().texts), ["重新可见", "没有父级盒子"]);
});

test("excluded branches are pruned before their descendants are traversed", () => {
  const page = createPage();
  const excluded = [
    page.element("pre"),
    page.element("div", { style: { display: "none" } }),
    page.element("div", { style: { opacity: "0" } }),
    page.element("div", { style: { contentVisibility: "hidden" } }),
    page.element("div", { attributes: { contenteditable: "true" } }),
  ];
  for (const element of excluded) {
    Object.defineProperty(element, "childNodes", {
      get() { assert.fail("traversed descendants of an excluded branch"); },
    });
  }
  const visible = page.text("仍然读取正文");
  assert.deepEqual(Array.from(page.read().texts), [visible.nodeValue]);
  assert.deepEqual(page.visited, [...excluded, visible]);
});

test("excluded body or html prevents traversal even though TreeWalker does not filter its root", () => {
  for (const target of ["body", "root"]) {
    const page = createPage();
    page[target].attributes.hidden = "";
    page.text("不应该进入遍历");
    assert.deepEqual(Array.from(page.read().texts), []);
    assert.equal(page.visited.length, 0);
  }
});

test("an offscreen parent does not prune a visible positioned descendant", () => {
  const page = createPage();
  const offscreen = page.element("div", { rect: box(0, 200, 100, 100) });
  page.text("屏幕之外", () => box(0, 200, 40, 20), offscreen);
  const fixed = page.element("span", { parent: offscreen, style: { position: "fixed" } });
  page.text("屏幕之内", () => box(0, 0, 40, 20), fixed);
  assert.deepEqual(Array.from(page.read().texts), ["屏幕之内"]);
});

test("text fully clipped by a scroll container is excluded even inside the viewport", () => {
  const page = createPage();
  const scroller = page.element("div", {
    rect: box(20, 20, 40, 20), style: { overflowX: "hidden", overflowY: "auto" },
  });
  page.text("没有显示", () => box(20, 60, 40, 20), scroller);
  page.text("显示一半也保留全部", () => box(20, 30, 40, 20), scroller);
  assert.deepEqual(Array.from(page.read().texts), ["显示一半也保留全部"]);
});

test("viewport edges exclude offscreen and zero-area text and retain partially visible nodes", () => {
  const page = createPage();
  for (const rect of [box(-10, 0, 10, 10), box(100, 0, 10, 10), box(0, -10, 10, 10),
    box(0, 100, 10, 10), box(0, 0, 0, 10), box(0, 0, 10, 0)]) {
    page.text("外", () => rect);
  }
  page.text("上", () => box(0, -5, 10, 10));
  page.text("下", () => box(0, 95, 10, 10));
  assert.deepEqual(Array.from(page.read().texts), ["上", "下"]);
});

test("a partially visible 100,000-character node is captured whole with one range measurement", () => {
  const page = createPage();
  const node = page.text("汉".repeat(100_000), (start, end) => box(0, start * 10 - 500_000, 10, (end - start) * 10));
  const snapshot = page.read();
  assert.deepEqual(Array.from(snapshot.texts), [node.nodeValue]);
  assert.equal(snapshot.sources[0].start, 0);
  assert.equal(snapshot.sources[0].end, 100_000);
  assert.equal(snapshot.sources[0].node, node);
  assert.equal(node.nodeValue.length, 100_000);
  assert.equal(page.measurements.length, 1);
});

test("a node intersecting a scroll container is captured once, including its clipped characters", () => {
  const page = createPage();
  const parent = page.element("div", {
    rect: box(20, 0, 20, 40), style: { overflowX: "hidden", overflowY: "auto" },
  });
  const rectangles = Array.from({ length: 12 }, (_, i) => box((i % 6) * 10, Math.floor(i / 6) * 20, 10, 20));
  const node = page.text("甲乙丙丁戊己庚辛壬癸子丑", cells(rectangles), parent);
  const snapshot = page.read();
  assert.deepEqual(Array.from(snapshot.texts), [node.nodeValue]);
  assert.deepEqual(Array.from(snapshot.sources, ({ start, end }) => [start, end]), [[0, node.nodeValue.length]]);
  assert.ok(snapshot.sources.every((source) => source.node === node));
});

test("a bounding rectangle crossing the viewport does not imply its characters are visible", () => {
  const page = createPage();
  page.text("甲乙", cells([box(0, -20, 10, 10), box(0, 110, 10, 10)]));
  assert.deepEqual(Array.from(page.read().texts), []);
});

test("a partially visible mixed-language node is captured whole even when its Han text is offscreen", () => {
  const page = createPage();
  page.text("abc中文", cells([
    box(0, 0, 10, 10), box(10, 0, 10, 10), box(20, 0, 10, 10),
    box(100, 0, 10, 10), box(110, 0, 10, 10),
  ]));
  assert.deepEqual(Array.from(page.read().texts), ["abc中文"]);
});

test("partly visible supplementary Han characters keep both UTF-16 code units", () => {
  const page = createPage();
  page.view.visualViewport = { offsetLeft: 5, offsetTop: 0, width: 5, height: 10 };
  page.text("𠀀甲", cells([box(0, 0, 10, 10), box(0, 0, 10, 10), box(10, 0, 10, 10)]));
  const snapshot = page.read();
  assert.deepEqual(Array.from(snapshot.texts), ["𠀀甲"]);
  assert.equal(snapshot.sources[0].start, 0);
  assert.equal(snapshot.sources[0].end, 3);
  assert.ok(page.measurements.every(({ start, end }) => start !== 1 && end !== 1));
});

test("snapshot text, source offsets, and visual viewport bounds stay fixed after the page changes", () => {
  const page = createPage();
  page.view.visualViewport = { offsetLeft: 20, offsetTop: 30, width: 20, height: 20 };
  const node = page.text("甲乙丙丁戊己", (start, end) => box(start * 10, 30, (end - start) * 10, 10));
  const first = page.read();
  node.nodeValue = "子丑寅卯辰巳";
  page.view.visualViewport.offsetLeft = 40;
  const second = page.read();
  assert.deepEqual(Array.from(first.texts), ["甲乙丙丁戊己"]);
  assert.equal(first.sources[0].text, "甲乙丙丁戊己");
  assert.deepEqual({ ...first.viewport }, { left: 20, top: 30, right: 40, bottom: 50 });
  assert.deepEqual(Array.from(second.texts), ["子丑寅卯辰巳"]);
});
