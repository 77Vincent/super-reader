const test = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const vm = require("node:vm");

const sampleText = "这里是一段用于检查开启和关闭的中文文本";
const drainPromises = () => new Promise((resolve) => setImmediate(resolve));

// A small DOM + extension-message harness. Run the unmodified content script
// and observe messages, markers, and timers rather than its private state.
function createPage(text = sampleText, tag = "p") {
  let document;
  let listener;
  let nextTimer = 1;
  const timers = new Map();
  const requests = [];
  const states = [];

  class DomNode {
    constructor(nodeType) { this.nodeType = nodeType; this.parentNode = null; }
    get parentElement() { return this.parentNode; }
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
    constructor(value) { super(3); this.nodeValue = value; }
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
    getClientRects() { return [{}]; }
    closest() {
      for (let node = this; node; node = node.parentElement) {
        if (["PRE", "CODE", "BUTTON"].includes(node.tagName) || node.attributes["aria-hidden"] === "true") return node;
      }
      return null;
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
    createElement: (name) => new Element(name),
    querySelectorAll: () => html.querySelectorAll(),
    createTreeWalker(root) {
      const nodes = [];
      function visit(node) {
        if (node.nodeType === 3) nodes.push(node);
        else node.childNodes.forEach(visit);
      }
      visit(root);
      let index = 0;
      return { nextNode() { this.currentNode = nodes[index++]; return Boolean(this.currentNode); } };
    },
  };
  const page = {
    paragraph, timers, requests, states,
    infer: async (texts) => ({ offsetsByText: texts.map((text) => text.length > 4 ? [4] : []) }),
    markers: () => paragraph.querySelectorAll(),
    toggle() {
      let response;
      listener({ type: "SUPER_READER_TOGGLE" }, {}, (value) => { response = value; });
      return response;
    },
    async step() {
      const next = timers.entries().next().value;
      assert.ok(next, "expected another batch");
      timers.delete(next[0]);
      void next[1]();
      await drainPromises();
    },
    async finish() {
      for (let index = 0; timers.size && index < 30; index += 1) await this.step();
      assert.equal(timers.size, 0);
    },
  };
  const context = vm.createContext({
    document, NodeFilter: { SHOW_TEXT: 4 },
    getComputedStyle: () => ({ visibility: "visible" }),
    console: { error() {} },
    setTimeout(callback) { const id = nextTimer++; timers.set(id, callback); return id; },
    clearTimeout: (id) => timers.delete(id),
    chrome: { runtime: {
      onMessage: { addListener: (callback) => { listener = callback; } },
      async sendMessage(message) {
        if (message.type === "SUPER_READER_STATE") { states.push(message); return {}; }
        requests.push(message.texts);
        return page.infer(message.texts);
      },
    } },
  });
  vm.runInContext(readFileSync(join(__dirname, "../src/content.js"), "utf8"), context);
  return page;
}

test("the page is inert until toggled, then renders and restores its original text", async () => {
  const page = createPage();
  assert.equal(page.requests.length, 0);
  assert.equal(page.toggle().enabled, true);
  await page.finish();
  assert.equal(page.markers().length, 1);
  assert.equal(page.paragraph.textContent, sampleText);
  assert.equal(page.toggle().enabled, false);
  assert.equal(page.markers().length, 0);
  assert.equal(page.paragraph.childNodes.length, 1);
  assert.equal(page.paragraph.textContent, sampleText);
});

test("toggle is rejected throughout inference and becomes available after DOM insertion", async () => {
  const page = createPage();
  let complete;
  page.infer = () => new Promise((resolve) => { complete = resolve; });
  page.toggle();
  await page.step();
  assert.equal(page.toggle().busy, true);
  assert.equal(page.toggle().enabled, true);
  complete({ offsetsByText: [[4]] });
  await drainPromises();
  assert.equal(page.markers().length, 1);
  assert.equal(page.states.at(-1).busy, false);
  assert.equal(page.toggle().enabled, false);
  assert.equal(page.timers.size, 0);
});

test("long text is bounded per batch without splitting surrogate pairs or losing text", async () => {
  const text = "甲".repeat(127) + "😀" + "乙".repeat(260);
  const page = createPage(text);
  page.toggle();
  await page.finish();
  const inputs = page.requests.map((texts) => {
    assert.equal(texts.length, 1);
    assert.ok(texts[0].length <= 128);
    assert.doesNotMatch(texts[0], /^[\uDC00-\uDFFF]|[\uD800-\uDBFF]$/u);
    return texts[0];
  });
  assert.equal(inputs.reverse().join(""), text);
  assert.equal(page.paragraph.textContent, text);
  page.toggle();
  assert.equal(page.paragraph.textContent, text);
});

test("turning off between batches drops the rest of the queue", async () => {
  const page = createPage("甲".repeat(400));
  page.toggle();
  await page.step();
  assert.equal(page.requests.length, 1);
  assert.equal(page.toggle().enabled, false);
  await page.finish();
  assert.equal(page.requests.length, 1);
  assert.equal(page.markers().length, 0);
});

test("a failed batch clears partial rendering, unlocks, and permits retry", async () => {
  const page = createPage("甲".repeat(260));
  page.toggle();
  await page.step();
  assert.equal(page.markers().length, 1);
  page.infer = async () => ({ error: "inference timed out" });
  await page.step();
  assert.equal(page.states.at(-1).enabled, false);
  assert.equal(page.states.at(-1).busy, false);
  assert.match(page.states.at(-1).error, /timed out/u);
  assert.equal(page.markers().length, 0);
  page.infer = async (texts) => ({ offsetsByText: texts.map((text) => text.length > 4 ? [4] : []) });
  assert.equal(page.toggle().enabled, true);
  await page.finish();
  assert.ok(page.markers().length > 0);
});

test("a result for text changed by the page is discarded", async () => {
  const page = createPage();
  let complete;
  page.infer = () => new Promise((resolve) => { complete = resolve; });
  page.toggle();
  await page.step();
  page.paragraph.childNodes[0].nodeValue = "页面更新了这段中文";
  complete({ offsetsByText: [[4]] });
  await drainPromises();
  assert.equal(page.markers().length, 0);
  assert.equal(page.paragraph.textContent, "页面更新了这段中文");
});

test("pages have independent switches and excluded text is never sent for inference", async () => {
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
    assert.equal(page.requests.length, 0);
  }
});
