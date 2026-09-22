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
function createPage(text = sampleText, tag = "p", useChromeAdapter = false, optedOut = false) {
  let document;
  let reader;
  const requests = [];
  const states = [];
  const layoutObservers = new Set();
  let contentObserver;
  let bodyHeight = 600;

  class DomNode {
    constructor(nodeType) { this.nodeType = nodeType; this.parentNode = null; }
    get parentElement() { return this.parentNode; }
    get nextSibling() {
      const siblings = this.parentNode?.childNodes;
      return siblings?.[siblings.indexOf(this) + 1] || null;
    }
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
  body.getBoundingClientRect = () => ({ left: 0, top: 0, right: 800, bottom: bodyHeight, width: 800, height: bodyHeight });
  body.append(paragraph);
  const html = new Element("html");
  html.append(body);
  document = {
    body, documentElement: html,
    querySelector(selector) {
      assert.equal(selector, 'meta[name="super-reader"][content="off"]');
      return optedOut ? {} : null;
    },
    defaultView: Object.assign(new EventTarget(), {
      visualViewport: Object.assign(new EventTarget(), { offsetLeft: 0, offsetTop: 0, width: 800, height: 600 }),
      MutationObserver: class {
        // Delivery is driven explicitly here; native fixtures exercise browser delivery.
        constructor(callback) { this.callback = callback; this.records = []; contentObserver = this; }
        observe() { this.observing = true; }
        disconnect() { this.observing = false; this.records = []; }
        takeRecords() { const records = this.records; this.records = []; return records; }
        queue(record) { if (this.observing) this.records.push(record); }
        deliver() { const records = this.takeRecords(); if (records.length) this.callback(records); }
      },
      ResizeObserver: class {
        constructor(callback) { this.callback = callback; }
        observe(target) { assert.equal(target, body); layoutObservers.add(this); }
        disconnect() { layoutObservers.delete(this); }
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
    click() {
      const state = reader.status();
      if (state.busy) return state;
      return reader.toggle(!state.enabled);
    },
    finish: drainPromises,
    scroll: () => document.defaultView.dispatchEvent(new Event("scroll")),
    resize: () => document.defaultView.dispatchEvent(new Event("resize")),
    grow() {
      bodyHeight += 100;
      layoutObservers.forEach((observer) => observer.callback());
    },
    async settle() { await new Promise((resolve) => setTimeout(resolve, 220)); await drainPromises(); },
    edit(node, value) {
      const oldValue = node.nodeValue;
      node.nodeValue = value;
      contentObserver.queue({ type: "characterData", target: node, oldValue });
    },
    deliver: () => contentObserver.deliver(),
    queueMutation: (record) => contentObserver.queue(record),
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
  run("src/frontend/content-changes.js");
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
    let globalEnabled = false;
    page.applySetting = (enabled) => {
      globalEnabled = enabled;
      let response;
      listeners[0]({ type: "SUPER_READER_APPLY_SETTING", enabled }, {}, (value) => { response = value; });
      return response;
    };
    reader = {
      status() {
        let response;
        listeners[0]({ type: "SUPER_READER_PING" }, {}, (value) => { response = value; });
        return response;
      },
      toggle(enabled) { return page.applySetting(enabled); },
    };
    page.click = () => {
      const state = reader.status();
      if (state.busy) return state;
      return reader.toggle(!globalEnabled);
    };
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
  assert.equal(page.click().enabled, true);
  await page.finish();
  assert.deepEqual(page.requests, [[sampleText]]);
  assert.equal(page.markers().length, 1);
  assert.equal(page.paragraph.textContent, sampleText);
  assert.equal(page.click().enabled, false);
  assert.equal(page.markers().length, 0);
  assert.equal(page.paragraph.childNodes.length, 1);
  assert.equal(page.paragraph.textContent, sampleText);
});

test("the Chrome adapter leaves a demo page's own markers and controls independent", async () => {
  const page = createPage(sampleText, "p", true, true);
  assert.equal(page.applySetting(true).enabled, false);
  await page.finish();
  assert.equal(page.requests.length, 0);
  assert.equal(page.markers().length, 0);
  assert.equal(page.paragraph.textContent, sampleText);
});

test("one request includes all visible texts, including strings longer than 128 UTF-16 units", async () => {
  const text = "甲".repeat(127) + "𠀀" + "乙".repeat(260);
  const page = createPage(text, "p", true);
  const strong = page.element("strong");
  page.text(sampleText, strong);
  const offscreen = page.text("屏幕外的文字不参与本次处理");
  offscreen.geometry = () => ({ left: 0, top: 700, right: 100, bottom: 720, width: 100, height: 20 });
  page.click();
  await page.finish();
  assert.deepEqual(page.requests, [[text, sampleText]]);
  assert.deepEqual(page.states.map(({ busy }) => busy), [true, false]);
  assert.equal(page.markers().length, 2);
  assert.equal(strong.querySelectorAll().length, 1);
  assert.equal(strong.textContent, sampleText);
  page.click();
  assert.equal(page.paragraph.childNodes[0].nodeValue, text);
  assert.equal(page.paragraph.childNodes[1], strong);
});

test("button clicks are ignored for the entire viewport and all results are written before unlock", async () => {
  const page = createPage();
  page.text("这里是屏幕里的第二段中文文本");
  let complete;
  page.infer = () => new Promise((resolve) => { complete = resolve; });
  page.click();
  assert.equal(page.requests.length, 1);
  assert.equal(page.markers().length, 0);
  for (let click = 0; click < 3; click += 1) {
    assert.equal(page.click().busy, true);
    assert.equal(page.click().enabled, true);
  }
  assert.equal(page.states.length, 1);
  complete({ offsetsByText: [[4], [5]] });
  await page.finish();
  assert.equal(page.markers().length, 2);
  assert.deepEqual(page.states.map(({ busy }) => busy), [true, false]);
  assert.equal(page.click().enabled, false);
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
  page.click();
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
  page.click();
  await page.finish();
  const failed = page.states.at(-1);
  assert.equal(failed.enabled, false);
  assert.equal(failed.busy, false);
  assert.match(failed.error, /timed out/u);
  assert.equal(page.markers().length, 0);
  await page.finish();
  assert.equal(page.requests.length, 1); // No automatic retry.
  page.infer = async () => ({ offsetsByText: [[4]] });
  page.click(); // Global OFF; failure only stopped this page's reader.
  assert.equal(page.click().error, null);
  await page.finish();
  assert.equal(page.markers().length, 1);
});

for (const mutation of ["change", "append", "detach", "reparent"]) {
  test(`results for text that the page did ${mutation} during inference are discarded`, async () => {
    const page = createPage();
    let complete;
    page.infer = () => new Promise((resolve) => { complete = resolve; });
    page.click();
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
  first.click();
  await first.finish();
  assert.equal(second.requests.length, 0);
  assert.equal(second.markers().length, 0);
  for (const tag of ["code", "pre", "button"]) {
    const page = createPage(sampleText, tag);
    page.click();
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
  page.click();
  await page.finish();
  assert.match(page.states.at(-1).error, /incomplete viewport result/u);
  assert.equal(page.states.at(-1).enabled, false);
  assert.equal(page.markers().length, 0);
});

test("global ON and OFF commands are idempotent and an ON command doesn't retry a failed page", async () => {
  const page = createPage(sampleText, "p", true);
  page.applySetting(true);
  page.applySetting(true);
  await page.finish();
  const marker = page.markers()[0];
  page.applySetting(true);
  assert.equal(page.requests.length, 1);
  assert.equal(page.markers().length, 1);
  assert.equal(page.markers()[0], marker);
  page.applySetting(false);
  page.applySetting(false);
  assert.equal(page.markers().length, 0);
  assert.equal(page.states.at(-1).enabled, false);

  page.infer = async () => ({ error: "failed" });
  page.applySetting(true);
  await page.finish();
  page.applySetting(true);
  await page.finish();
  assert.equal(page.requests.length, 2, "repeated global synchronization must not retry failures");
  assert.equal(page.states.at(-1).error, "failed");
  page.applySetting(false);
  page.infer = async () => ({ offsetsByText: [[4]] });
  page.applySetting(true);
  await page.finish();
  assert.equal(page.requests.length, 3);
  assert.equal(page.markers().length, 1);
  page.applySetting(false);
});

test("a global OFF received while busy is ignored until another idle command", async () => {
  const page = createPage(sampleText, "p", true);
  let finish;
  page.infer = () => new Promise((resolve) => { finish = resolve; });
  page.applySetting(true);
  const stillRunning = page.applySetting(false);
  assert.equal(stillRunning.busy, true);
  assert.equal(stillRunning.enabled, true);
  assert.equal(page.states.length, 1);
  finish({ offsetsByText: [[4]] });
  await page.finish();
  assert.equal(page.requests.length, 1);
  assert.equal(page.markers().length, 1);
  assert.equal(page.states.at(-1).enabled, true);
  assert.equal(page.states.at(-1).busy, false);
  page.applySetting(false);
  assert.equal(page.markers().length, 0);
  assert.equal(page.states.at(-1).enabled, false);
});

test("Chrome translates state commands and publishing only delivers the supplied state", async () => {
  let listener;
  const commands = [];
  const messages = [];
  const state = { enabled: true, busy: true, error: null };
  const context = vm.createContext({ document: { querySelector: () => null }, chrome: { runtime: {
    getURL: (path) => path,
    onMessage: { addListener(callback) { listener = callback; } },
    async sendMessage(message) { messages.push(message); return {}; },
  } } });
  vm.runInContext(readFileSync(join(__dirname, "../src/platform/chrome/content.js"), "utf8"), context);
  const adapter = context.SuperReader.createReaderAdapter();
  adapter.connect({
    status() { throw new Error("Only a PING should ask for status"); },
    toggle(value) { commands.push(value); return state; },
  });
  let response;
  listener({ type: "SUPER_READER_APPLY_SETTING", enabled: false }, {}, (value) => { response = value; });
  assert.equal(response, state);
  assert.deepEqual(commands, [false]);
  const finished = { enabled: false, busy: false, error: null };
  adapter.publishState(finished);
  await drainPromises();
  assert.deepEqual(commands, [false], "publishing must not issue another control command");
  assert.deepEqual({ ...messages[0] }, { type: "SUPER_READER_STATE", ...finished });
});

test("an article arriving after the initial empty read is processed when the page grows", async () => {
  const page = createPage("", "p", true);
  page.applySetting(true);
  await page.finish();
  assert.equal(page.requests.length, 0);
  assert.equal(page.states.at(-1).enabled, true);
  page.text(sampleText);
  page.grow();
  page.grow();
  await page.settle();
  assert.deepEqual(page.requests, [[sampleText]]);
  const marker = page.markers()[0];
  assert.ok(marker);
  page.grow(); // Later layout updates preserve already processed text.
  await page.settle();
  assert.equal(page.requests.length, 1);
  assert.equal(page.markers()[0], marker);
  page.applySetting(false);
  const states = page.states.length;
  page.text("关闭之后新增的文字不应该被处理");
  page.grow();
  await page.settle();
  assert.equal(page.states.length, states);
  assert.equal(page.requests.length, 1);
});

test("scroll and resize process newly visible text once and leave overlapping markers unchanged", async () => {
  const page = createPage(sampleText, "p", true);
  const nextParent = page.element("strong");
  const next = page.text("滚动后新出现的文字需要处理", nextParent);
  next.geometry = () => ({ left: 0, top: 700, right: 80, bottom: 720, width: 80, height: 20 });
  page.click();
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
  page.click();
});

test("nodes needing no markers are remembered, while changed text is processed again", async () => {
  const page = createPage();
  page.infer = async (texts) => ({ offsetsByText: texts.map(() => []) });
  page.click();
  await page.finish();
  page.resize();
  await page.settle();
  assert.deepEqual(page.requests, [[sampleText]]);
  page.paragraph.childNodes[0].nodeValue = "页面已更新这段中文文字";
  page.scroll();
  await page.settle();
  assert.deepEqual(page.requests, [[sampleText], ["页面已更新这段中文文字"]]);
  page.click();
});

test("a title rewrite preserves adopted text, removes a leading marker, and leaves other groups intact", async () => {
  const page = createPage("我的数学家朋友集体破防AI又双叒叕来毁灭人类了？", "h1");
  const other = page.element("p", page.document.body);
  page.text(sampleText, other);
  page.click();
  await page.finish();
  const otherMarker = other.querySelectorAll()[0];
  const [left, marker, right] = page.paragraph.childNodes;
  left.remove();
  right.nodeValue = "AI又双叒叕来毁灭人类了？";
  const prefix = page.text("我的数学家朋友集体破防");
  prefix.remove();
  right.before(prefix);
  assert.equal(page.paragraph.childNodes[0], marker);
  const currentTitle = "我的数学家朋友集体破防AI又双叒叕来毁灭人类了？";
  page.infer = async (texts) => ({ offsetsByText: texts.map(() => []) });
  page.resize();
  await page.settle();
  assert.deepEqual(page.requests[1], [prefix.nodeValue], "adopted right fragment is paused; new page text is read independently");
  assert.equal(page.paragraph.querySelectorAll().length, 0);
  assert.equal(page.paragraph.textContent, currentTitle);
  assert.equal(other.querySelectorAll()[0], otherMarker);
  page.resize();
  await page.settle();
  assert.equal(page.requests.length, 2, "reconciled text is remembered");
  page.click();
});

test("an edited derived fragment pauses its group until a later page edit", async () => {
  const page = createPage();
  page.infer = async (texts) => ({ offsetsByText: texts.map(() => [4, 8]) });
  page.click();
  await page.finish();
  const oldMarkers = page.markers();
  const [anchor, , middle, , tail] = page.paragraph.childNodes;
  middle.nodeValue = "页面改写后的中间片段";
  const current = sampleText.slice(0, 4) + middle.nodeValue + sampleText.slice(8);
  page.resize();
  await page.settle();
  assert.equal(page.requests.length, 1, "conflicted fragments are not immediately split again");
  assert.equal(page.markers().length, 0);
  assert.ok(oldMarkers.every((marker) => !marker.isConnected));
  assert.deepEqual(page.paragraph.childNodes, [anchor, middle, tail]);
  assert.equal(page.paragraph.textContent, current);
  middle.nodeValue = "页面再次修改的独立文字";
  page.infer = async (texts) => ({ offsetsByText: texts.map(() => []) });
  page.resize();
  await page.settle();
  assert.deepEqual(page.requests[1], ["页面再次修改的独立文字"]);
  page.click();
});

test("removing the anchor discards its untouched derived text and markers", async () => {
  const page = createPage();
  page.click();
  await page.finish();
  page.paragraph.childNodes[0].remove();
  page.infer = async () => ({ offsetsByText: [[]] });
  page.resize();
  await page.settle();
  assert.equal(page.requests.length, 1, "old derived text must not become a fresh source");
  assert.equal(page.markers().length, 0);
  assert.equal(page.paragraph.textContent, "");
  assert.equal(page.paragraph.childNodes.length, 0);
  page.click();
});

test("unwrite preserves adjacent original Text objects, empty nodes, and nested markup", async () => {
  const page = createPage();
  const anchor = page.paragraph.childNodes[0];
  const second = page.text("第二个原始文本节点必须独立保留");
  const empty = page.text("");
  const strong = page.element("strong");
  const nested = page.text("嵌套原始文字也要还原", strong);
  page.infer = async (texts) => ({ offsetsByText: texts.map(() => [4, 8]) });
  page.paragraph.normalize = () => { throw new Error("must not normalize page parents"); };
  for (let iteration = 0; iteration < 2; iteration += 1) {
    page.click();
    await page.finish();
    page.click();
    assert.deepEqual(page.paragraph.childNodes, [anchor, second, empty, strong]);
    assert.equal(anchor.nodeValue, sampleText);
    assert.equal(second.nodeValue, "第二个原始文本节点必须独立保留");
    assert.deepEqual(strong.childNodes, [nested]);
    assert.equal(nested.nodeValue, "嵌套原始文字也要还原");
  }
});

test("anchor replacement and clearing discard every old tail before rereading", async () => {
  for (const updated of ["这是新标题", ""]) {
    const page = createPage("今天我们讨论人工智能", "h1");
    const anchor = page.paragraph.childNodes[0];
    page.infer = async (texts) => ({ offsetsByText: texts.map(() => [4, 6]) });
    page.click();
    await page.finish();
    const oldNodes = page.paragraph.childNodes.slice(1);
    page.infer = async (texts) => ({ offsetsByText: texts.map(() => []) });
    page.edit(anchor, updated);
    page.deliver();
    await page.settle();
    assert.deepEqual(page.paragraph.childNodes, [anchor]);
    assert.equal(page.paragraph.textContent, updated);
    assert.ok(oldNodes.every((node) => !node.isConnected));
    assert.deepEqual(page.requests.slice(1), updated ? [[updated]] : []);
    page.click();
    assert.equal(anchor.nodeValue, updated, "OFF must not resurrect the old source");
  }
});

test("same-value anchor writes are dirty even for an ASCII prefix", async () => {
  const page = createPage("1234后面的中文不应残留");
  const anchor = page.paragraph.childNodes[0];
  page.click();
  await page.finish();
  page.edit(anchor, "1234");
  page.deliver();
  await page.settle();
  assert.deepEqual(page.paragraph.childNodes, [anchor]);
  assert.equal(page.paragraph.textContent, "1234");
  assert.equal(page.requests.length, 1);
  page.click();
});

test("OFF consumes queued anchor writes, including writes back to the saved prefix", async () => {
  for (const intermediate of [null, "中途更新的文字"]) {
    const page = createPage("今天我们讨论人工智能");
    const anchor = page.paragraph.childNodes[0];
    page.click();
    await page.finish();
    if (intermediate) page.edit(anchor, intermediate);
    page.edit(anchor, "今天我们");
    // No observer delivery or viewport refresh before shutdown.
    page.click();
    assert.deepEqual(page.paragraph.childNodes, [anchor]);
    assert.equal(anchor.nodeValue, "今天我们");
  }
});

test("missing or relocated dividers restore only their own text group", async () => {
  const page = createPage();
  const anchor = page.paragraph.childNodes[0];
  const other = page.text("同一个父元素里的另一个独立原文");
  page.click();
  await page.finish();
  const [divider, otherDivider] = page.markers();
  const destination = page.element("div", page.document.body);
  destination.append(divider);
  page.infer = async (texts) => ({ offsetsByText: texts.map(() => []) });
  page.resize();
  await page.settle();
  assert.deepEqual(page.requests[1], [sampleText]);
  assert.equal(anchor.nodeValue, sampleText);
  assert.equal(divider.parentNode, null);
  assert.equal(otherDivider.parentNode, page.paragraph);
  assert.equal(other.nodeValue, "同一个父");
  page.click();
  assert.deepEqual(page.paragraph.childNodes, [anchor, other]);
  assert.equal(other.nodeValue, "同一个父元素里的另一个独立原文");
});

test("a same-value write to a derived fragment is treated as page adoption", async () => {
  const page = createPage();
  page.click();
  await page.finish();
  const [anchor, , tail] = page.paragraph.childNodes;
  page.edit(tail, tail.nodeValue);
  page.deliver();
  await page.settle();
  assert.deepEqual(page.paragraph.childNodes, [anchor, tail]);
  assert.equal(page.requests.length, 1);
  assert.equal(page.paragraph.textContent, sampleText);
  page.click();
  assert.deepEqual(page.paragraph.childNodes, [anchor, tail], "adopted text must not be merged on OFF");
});

test("independent fragment moves preserve page structure and pause the conflicted group", async () => {
  const page = createPage();
  page.click();
  await page.finish();
  const [anchor, , tail] = page.paragraph.childNodes;
  const destination = page.element("p", page.document.body);
  destination.append(tail);
  page.resize();
  await page.settle();
  assert.deepEqual(page.paragraph.childNodes, [anchor]);
  assert.deepEqual(destination.childNodes, [tail]);
  assert.equal(page.requests.length, 1);
  page.click();
  assert.equal(anchor.nodeValue, sampleText.slice(0, 4));
  assert.equal(tail.nodeValue, sampleText.slice(4));
});

test("a fragment moved away and back cannot be mistaken for an untouched group", async () => {
  const page = createPage();
  page.click();
  await page.finish();
  const [anchor, , tail] = page.paragraph.childNodes;
  tail.remove();
  page.paragraph.append(tail);
  page.queueMutation({ type: "childList", target: page.paragraph, removedNodes: [tail], addedNodes: [] });
  page.click();
  assert.deepEqual(page.paragraph.childNodes, [anchor, tail]);
  assert.equal(page.paragraph.textContent, sampleText);
});

test("a foreign node inserted inside a group survives cleanup without merging its neighbors", async () => {
  const page = createPage();
  page.click();
  await page.finish();
  const [anchor, , tail] = page.paragraph.childNodes;
  const inserted = page.text("页面插入的新文字");
  inserted.remove();
  tail.before(inserted);
  page.click();
  assert.deepEqual(page.paragraph.childNodes, [anchor, inserted, tail]);
  assert.equal(page.paragraph.textContent, sampleText.slice(0, 4) + "页面插入的新文字" + sampleText.slice(4));
});

test("detached intact subtrees can still be unwritten and reused by the page", async () => {
  const page = createPage();
  const anchor = page.paragraph.childNodes[0];
  page.click();
  await page.finish();
  page.paragraph.remove();
  page.resize();
  await page.settle();
  assert.deepEqual(page.paragraph.childNodes, [anchor]);
  assert.equal(anchor.nodeValue, sampleText);
  page.click();
});

test("disable cancels scheduled refreshes, removes listeners, and resets processed text for re-enable", async () => {
  const page = createPage();
  page.click();
  await page.finish();
  page.scroll();
  page.click();
  page.scroll();
  page.resize();
  await page.settle();
  assert.equal(page.requests.length, 1);
  assert.equal(page.markers().length, 0);
  page.click();
  await page.finish();
  assert.deepEqual(page.requests, [[sampleText], [sampleText]]);
  assert.equal(page.markers().length, 1);
  page.click();
});

test("a stale result is not remembered and the pending refresh processes the changed node", async () => {
  const page = createPage();
  let finishFirst;
  page.infer = () => new Promise((resolve) => { finishFirst = resolve; });
  page.click();
  page.paragraph.childNodes[0].nodeValue = "处理期间页面更新的新中文文字";
  page.scroll();
  await page.settle();
  assert.equal(page.requests.length, 1);
  page.infer = async () => ({ offsetsByText: [[4]] });
  finishFirst({ offsetsByText: [[4]] });
  await page.finish();
  assert.deepEqual(page.requests, [[sampleText], ["处理期间页面更新的新中文文字"]]);
  assert.equal(page.markers().length, 1);
  page.click();
});
