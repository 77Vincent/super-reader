const test = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const vm = require("node:vm");

const sampleText = "这里是一段用于检查开启和关闭的中文文本";
const drainPromises = () => new Promise((resolve) => setImmediate(resolve));

// Run the reader without extension APIs; optionally connect the Chrome adapter.
// Exercise the real DOM modules, reader, startup, and adapter together.
// Layout geometry and inference responses are supplied by this harness.
function createPage(text = sampleText, tag = "p", useChromeAdapter = false) {
  let document;
  let reader;
  const requests = [];
  const states = [];

  class DomNode {
    constructor(nodeType) { this.nodeType = nodeType; this.parentNode = null; }
    get parentElement() { return this.parentNode; }
    getRootNode() { return this.parentNode?.getRootNode() || this; }
    get isConnected() { return this === document.documentElement || Boolean(this.parentNode?.isConnected); }
    remove() {
      if (!this.parentNode) return;
      const siblings = this.parentNode.childNodes;
      siblings.splice(siblings.indexOf(this), 1);
      this.parentNode = null;
    }
    before(node) {
      const parent = this.parentNode;
      parent.childNodes.splice(parent.childNodes.indexOf(this), 0, node);
      node.parentNode = parent;
    }
  }
  class TextNode extends DomNode {
    constructor(value) {
      super(3); this.nodeValue = value;
      this.geometry = () => ({ left: 0, top: 0, right: 40, bottom: 20, width: 40, height: 20 });
    }
    get textContent() { return this.nodeValue; }
    splitText(offset) {
      assert.ok(offset <= this.nodeValue.length);
      const right = new TextNode(this.nodeValue.slice(offset));
      this.nodeValue = this.nodeValue.slice(0, offset);
      const parent = this.parentNode;
      parent.childNodes.splice(parent.childNodes.indexOf(this) + 1, 0, right);
      right.parentNode = parent;
      return right;
    }
  }
  class Element extends DomNode {
    constructor(name) {
      super(1);
      this.tagName = name.toUpperCase();
      this.childNodes = [];
      this.attributes = {};
      this.className = "";
    }
    get textContent() { return this.childNodes.map((node) => node.textContent).join(""); }
    append(node) { node.remove(); this.childNodes.push(node); node.parentNode = this; }
    setAttribute(name, value) { this.attributes[name] = value; }
    getBoundingClientRect() { return { left: 0, top: 0, right: 800, bottom: 600, width: 800, height: 600 }; }
    matches() {
      return ["PRE", "CODE", "BUTTON"].includes(this.tagName) || this.attributes["aria-hidden"] === "true";
    }
    querySelectorAll() {
      const matches = [];
      for (const node of this.childNodes) {
        if (!(node instanceof Element)) continue;
        if (node.className === "super-reader-divider") matches.push(node);
        matches.push(...node.querySelectorAll());
      }
      return matches;
    }
    normalize() {
      for (let index = this.childNodes.length - 1; index > 0; index -= 1) {
        const left = this.childNodes[index - 1];
        const right = this.childNodes[index];
        if (left instanceof TextNode && right instanceof TextNode) {
          left.nodeValue += right.nodeValue;
          right.remove();
        }
      }
    }
  }

  const paragraph = new Element(tag);
  paragraph.append(new TextNode(text));
  const body = new Element("body");
  body.append(paragraph);
  const html = new Element("html");
  html.append(body);
  document = {
    body, documentElement: html,
    defaultView: Object.assign(new EventTarget(), {
      visualViewport: Object.assign(new EventTarget(), { offsetLeft: 0, offsetTop: 0, width: 800, height: 600 }),
      MutationObserver: class {
        constructor() { throw new Error("The reader must not observe DOM mutations"); }
      },
      getComputedStyle: () => ({
        display: "block", visibility: "visible", opacity: "1",
        contentVisibility: "visible", overflowX: "visible", overflowY: "visible",
      }),
    }),
    createRange() {
      let node;
      return {
        selectNodeContents(source) { node = source; },
        getClientRects() { return [node.geometry(0, node.nodeValue.length)]; },
      };
    },
    createElement: (name) => new Element(name),
    querySelectorAll: () => html.querySelectorAll(),
    createTreeWalker(root, whatToShow, filter) {
      function* visit(parent) {
        for (const node of parent.childNodes) {
          const result = ((1 << (node.nodeType - 1)) & whatToShow) ? filter.acceptNode(node) : 3;
          if (result === 2) continue; // FILTER_REJECT prunes descendants.
          if (result === 1) yield node;
          if (node.nodeType === 1) yield* visit(node);
        }
      }
      const iterator = visit(root);
      return { nextNode() { this.currentNode = iterator.next().value; return this.currentNode || null; } };
    },
  };
  const page = {
    paragraph, document, requests, states,
    infer: async (texts) => ({ offsetsByText: texts.map((text) => text.length > 4 ? [4] : []) }),
    markers: () => document.querySelectorAll(),
    toggle: () => reader.toggle(),
    finish: drainPromises,
    scroll: () => document.defaultView.dispatchEvent(new Event("scroll")),
    resize: () => document.defaultView.dispatchEvent(new Event("resize")),
    async settle() { await new Promise((resolve) => setTimeout(resolve, 220)); await drainPromises(); },
    text(value, parent = paragraph) { const node = new TextNode(value); parent.append(node); return node; },
    element(tag, parent = paragraph) { const node = new Element(tag); parent.append(node); return node; },
  };
  const context = vm.createContext({ document, setTimeout, clearTimeout, NodeFilter: { SHOW_ELEMENT: 1, SHOW_TEXT: 4, FILTER_ACCEPT: 1, FILTER_REJECT: 2, FILTER_SKIP: 3 } });
  const run = (path) => vm.runInContext(
    readFileSync(join(__dirname, "..", path), "utf8"), context, { filename: path },
  );
  run("src/frontend/dom-tree.js");
  run("src/frontend/viewport.js");
  run("src/frontend/visibility.js");
  run("src/frontend/processed-text.js");
  run("src/frontend/read.js");
  run("src/frontend/write.js");
  run("src/app/reader.js");
  if (useChromeAdapter) {
    const listeners = [];
    context.chrome = { runtime: {
      getURL: (path) => `chrome-extension://test-extension/${path}`,
      onMessage: { addListener: (callback) => listeners.push(callback) },
      async sendMessage(message) {
        if (message.type === "SUPER_READER_STATE") { states.push(message); return {}; }
        requests.push(Array.from(message.texts));
        return page.infer(message.texts);
      },
    } };
    run("src/platform/chrome/content.js");
    run("src/start-reader.js");
    // Reinjecting must not register a second reader or message handler.
    run("src/frontend/dom-tree.js");
    run("src/frontend/viewport.js");
    run("src/frontend/visibility.js");
    run("src/frontend/processed-text.js");
    run("src/frontend/read.js");
    run("src/frontend/write.js");
    run("src/app/reader.js");
    run("src/platform/chrome/content.js");
    run("src/start-reader.js");
    assert.equal(listeners.length, 1);
    reader = { toggle() {
      let response;
      listeners[0]({ type: "SUPER_READER_TOGGLE" }, {}, (value) => { response = value; });
      return response;
    } };
  } else {
    // A second adapter implementation uses local callbacks, with no Chrome API.
    context.SuperReader.createReaderAdapter = () => ({
      markerStyleUrl: "http://localhost/src/content.css",
      async process(texts) {
        requests.push(Array.from(texts));
        const response = await page.infer(texts);
        if (response?.error) throw new Error(response.error);
        return response.offsetsByText;
      },
      publishState: (state) => states.push(state),
      connect: (instance) => { reader = instance; },
    });
    run("src/start-reader.js");
  }
  page.write = context.SuperReader.write;
  page.clear = context.SuperReader.clearMarkers;
  return page;
}

test("the page is inert until toggled, then renders and restores its original text", async () => {
  const page = createPage();
  assert.equal(page.requests.length, 0);
  assert.equal(page.toggle().enabled, true);
  await page.finish();
  assert.deepEqual(page.requests, [[sampleText]]);
  assert.equal(page.markers().length, 1);
  assert.equal(page.paragraph.textContent, sampleText);
  assert.equal(page.toggle().enabled, false);
  assert.equal(page.markers().length, 0);
  assert.equal(page.paragraph.childNodes.length, 1);
  assert.equal(page.paragraph.textContent, sampleText);
});

test("one request includes all visible texts, including strings longer than 128 UTF-16 units", async () => {
  const text = "甲".repeat(127) + "𠀀" + "乙".repeat(260);
  const page = createPage(text, "p", true);
  const strong = page.element("strong");
  page.text(sampleText, strong);
  const offscreen = page.text("屏幕外的文字不参与本次处理");
  offscreen.geometry = () => ({ left: 0, top: 700, right: 100, bottom: 720, width: 100, height: 20 });
  page.toggle();
  await page.finish();
  assert.deepEqual(page.requests, [[text, sampleText]]);
  assert.deepEqual(page.states.map(({ busy }) => busy), [true, false]);
  assert.equal(page.markers().length, 2);
  assert.equal(strong.querySelectorAll().length, 1);
  assert.equal(strong.textContent, sampleText);
  page.toggle();
  assert.equal(page.paragraph.childNodes[0].nodeValue, text);
  assert.equal(page.paragraph.childNodes[1], strong);
});

test("toggle is rejected for the entire viewport and all results are written before unlock", async () => {
  const page = createPage();
  page.text("这里是屏幕里的第二段中文文本");
  let complete;
  page.infer = () => new Promise((resolve) => { complete = resolve; });
  page.toggle();
  assert.equal(page.requests.length, 1);
  assert.equal(page.markers().length, 0);
  for (let click = 0; click < 3; click += 1) {
    assert.equal(page.toggle().busy, true);
    assert.equal(page.toggle().enabled, true);
  }
  assert.equal(page.states.length, 1);
  complete({ offsetsByText: [[4], [5]] });
  await page.finish();
  assert.equal(page.markers().length, 2);
  assert.deepEqual(page.states.map(({ busy }) => busy), [true, false]);
  assert.equal(page.toggle().enabled, false);
  assert.equal(page.markers().length, 0);
});

test("the original viewport snapshot is used even if the viewport moves during processing", async () => {
  const text = "甲".repeat(1_000);
  const page = createPage(text);
  page.paragraph.childNodes[0].geometry = (start, end) => ({
    left: 0, right: 10, width: 10,
    top: start * 10 - 2_000, bottom: end * 10 - 2_000, height: (end - start) * 10,
  });
  let complete;
  page.infer = () => new Promise((resolve) => { complete = resolve; });
  page.toggle();
  assert.deepEqual(page.requests, [[text]]);
  page.document.defaultView.visualViewport.offsetTop = 100;
  complete({ offsetsByText: [[4]] });
  await page.finish();
  assert.equal(page.paragraph.childNodes[0].nodeValue.length, 4);
  assert.equal(page.paragraph.textContent, text);
  assert.equal(page.requests.length, 1);
});

test("whole-node offsets are written from the end without shifting earlier positions", () => {
  const text = "甲乙丙丁戊己庚辛壬癸子丑寅卯辰巳";
  const page = createPage(text);
  const node = page.paragraph.childNodes[0];
  const snapshot = {
    texts: [text],
    sources: [
      { node, parent: page.paragraph, text, start: 0, end: text.length },
    ],
  };
  page.write(snapshot, [[3, 5, 11, 13]]);
  let offset = 0;
  const positions = [];
  for (const child of page.paragraph.childNodes) {
    if (child.nodeType === 3) offset += child.nodeValue.length;
    else {
      positions.push(offset);
      assert.equal(child.attributes["aria-hidden"], "true");
    }
  }
  assert.deepEqual(positions, [3, 5, 11, 13]);
  page.clear();
  assert.equal(page.paragraph.childNodes.length, 1);
  assert.equal(page.paragraph.textContent, text);
});

test("a failed operation disables and unlocks the reader, retains the error, and permits retry", async () => {
  const page = createPage(sampleText, "p", true);
  page.infer = async () => ({ error: "inference timed out" });
  page.toggle();
  await page.finish();
  const failed = page.states.at(-1);
  assert.equal(failed.enabled, false);
  assert.equal(failed.busy, false);
  assert.match(failed.error, /timed out/u);
  assert.equal(page.markers().length, 0);
  await page.finish();
  assert.equal(page.requests.length, 1); // No automatic retry.
  page.infer = async () => ({ offsetsByText: [[4]] });
  assert.equal(page.toggle().error, null);
  await page.finish();
  assert.equal(page.markers().length, 1);
});

for (const mutation of ["change", "append", "detach", "reparent"]) {
  test(`results for text that the page did ${mutation} during inference are discarded`, async () => {
    const page = createPage();
    let complete;
    page.infer = () => new Promise((resolve) => { complete = resolve; });
    page.toggle();
    const node = page.paragraph.childNodes[0];
    if (mutation === "change") node.nodeValue = "页面更新了这段中文";
    if (mutation === "append") node.nodeValue += "后面新增的文字";
    if (mutation === "detach") node.remove();
    if (mutation === "reparent") page.element("strong").append(node);
    complete({ offsetsByText: [[4]] });
    await page.finish();
    assert.equal(page.markers().length, 0);
    assert.equal(page.states.at(-1).busy, false);
  });
}

test("pages have independent switches and excluded text never enters the processing input", async () => {
  const first = createPage();
  const second = createPage();
  first.toggle();
  await first.finish();
  assert.equal(second.requests.length, 0);
  assert.equal(second.markers().length, 0);
  for (const tag of ["code", "pre", "button"]) {
    const page = createPage(sampleText, tag);
    page.toggle();
    await page.finish();
    assert.deepEqual(page.requests, []);
    assert.equal(page.markers().length, 0);
    assert.equal(page.states.at(-1).busy, false);
  }
});

test("Chrome rejects an incomplete viewport response before any markers are written", async () => {
  const page = createPage(sampleText, "p", true);
  page.text("第二段文字也需要一个完整的结果");
  page.infer = async () => ({ offsetsByText: [[4]] });
  page.toggle();
  await page.finish();
  assert.match(page.states.at(-1).error, /incomplete viewport result/u);
  assert.equal(page.states.at(-1).enabled, false);
  assert.equal(page.markers().length, 0);
});

test("scroll and resize process newly visible text once and leave overlapping markers unchanged", async () => {
  const page = createPage(sampleText, "p", true);
  const nextParent = page.element("strong");
  const next = page.text("滚动后新出现的文字需要处理", nextParent);
  next.geometry = () => ({ left: 0, top: 700, right: 80, bottom: 720, width: 80, height: 20 });
  page.toggle();
  await page.finish();
  const originalMarker = page.markers()[0];
  next.geometry = () => ({ left: 0, top: 100, right: 80, bottom: 120, width: 80, height: 20 });
  for (let i = 0; i < 5; i += 1) { page.scroll(); page.resize(); }
  assert.equal(page.requests.length, 1);
  await page.settle();
  assert.deepEqual(page.requests, [[sampleText], ["滚动后新出现的文字需要处理"]]);
  assert.equal(page.markers().length, 2);
  assert.equal(page.markers()[0], originalMarker);
  page.document.defaultView.visualViewport.dispatchEvent(new Event("resize"));
  await page.settle();
  assert.equal(page.requests.length, 2);
  assert.equal(page.markers().length, 2);
  page.toggle();
});

test("nodes needing no markers are remembered, while changed text is processed again", async () => {
  const page = createPage();
  page.infer = async (texts) => ({ offsetsByText: texts.map(() => []) });
  page.toggle();
  await page.finish();
  page.resize();
  await page.settle();
  assert.deepEqual(page.requests, [[sampleText]]);
  page.paragraph.childNodes[0].nodeValue = "页面已更新这段中文文字";
  page.scroll();
  await page.settle();
  assert.deepEqual(page.requests, [[sampleText], ["页面已更新这段中文文字"]]);
  page.toggle();
});

test("disable cancels scheduled refreshes, removes listeners, and resets processed text for re-enable", async () => {
  const page = createPage();
  page.toggle();
  await page.finish();
  page.scroll();
  page.toggle();
  page.scroll();
  page.resize();
  await page.settle();
  assert.equal(page.requests.length, 1);
  assert.equal(page.markers().length, 0);
  page.toggle();
  await page.finish();
  assert.deepEqual(page.requests, [[sampleText], [sampleText]]);
  assert.equal(page.markers().length, 1);
  page.toggle();
});

test("a stale result is not remembered and the pending refresh processes the changed node", async () => {
  const page = createPage();
  let finishFirst;
  page.infer = () => new Promise((resolve) => { finishFirst = resolve; });
  page.toggle();
  page.paragraph.childNodes[0].nodeValue = "处理期间页面更新的新中文文字";
  page.scroll();
  await page.settle();
  assert.equal(page.requests.length, 1);
  page.infer = async () => ({ offsetsByText: [[4]] });
  finishFirst({ offsetsByText: [[4]] });
  await page.finish();
  assert.deepEqual(page.requests, [[sampleText], ["处理期间页面更新的新中文文字"]]);
  assert.equal(page.markers().length, 1);
  page.toggle();
});
