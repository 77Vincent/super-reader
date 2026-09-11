const test = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const vm = require("node:vm");

function setup() {
  let observer;
  let notifications = 0;
  const element = (tag = "div", parent = null) => ({
    nodeType: 1, parentElement: parent, isConnected: true, childNodes: [],
    matches: () => ["script", "style", "link", "pre", "button", "marker"].includes(tag),
  });
  const text = (value, parent = null) => ({ nodeType: 3, nodeValue: value, parentElement: parent });
  const html = element("html");
  const document = { documentElement: html, defaultView: { MutationObserver: class {
    constructor(callback) { this.callback = callback; this.targets = new Set(); this.records = []; observer = this; }
    observe(root, options) {
      assert.deepEqual({ ...options }, { subtree: true, childList: true, characterData: true });
      this.targets.add(root);
    }
    disconnect() { this.targets.clear(); this.records = []; }
    takeRecords() { const records = this.records; this.records = []; return records; }
    queue(record) { if (this.targets.size) this.records.push(record); }
    deliver() { const records = this.takeRecords(); if (records.length) this.callback(records); }
  } } };
  const context = vm.createContext({ document });
  for (const file of ["dom-tree.js", "content-changes.js"]) {
    vm.runInContext(readFileSync(join(__dirname, "../src/frontend", file), "utf8"), context);
  }
  // Traversal is separately tested with real TreeWalker in the browser fixtures.
  context.SuperReader.walkDOM = function* walk(root, skip) {
    if (root.nodeType === 1 && skip(root)) return;
    yield root;
    if (root.shadowRoot) yield* walk(root.shadowRoot, skip);
    for (const child of root.childNodes || []) yield* walk(child, skip);
  };
  const changes = context.SuperReader.createContentChanges(document);
  const stop = changes.watch(() => { notifications += 1; });
  const mutation = (target, addedNodes = [], removedNodes = []) => ({ type: "childList", target, addedNodes, removedNodes });
  return { changes, stop, observer, html, element, text, mutation, count: () => notifications };
}

test("text replacement and direct edits notify once per delivery; numeric and excluded changes do not", () => {
  const p = setup();
  const title = p.element("h1", p.html);
  p.observer.queue(p.mutation(title, [p.text("同样的标题文字")], [p.text("同样的标题文字")]));
  p.observer.queue({ type: "characterData", target: p.text("更新后的中文", title) });
  p.observer.deliver();
  assert.equal(p.count(), 1);
  p.observer.queue({ type: "characterData", target: p.text("00:03", title) });
  p.observer.queue(p.mutation(title, [p.text("123")]));
  const script = p.element("script", p.html);
  script.childNodes.push(p.text("脚本里面的中文", script));
  p.observer.queue(p.mutation(p.html, [script]));
  p.observer.queue({ type: "characterData", target: script.childNodes[0] });
  p.observer.queue(p.mutation(p.element("span", p.element("pre", p.html)), [p.text("代码文字")]));
  p.observer.deliver();
  assert.equal(p.count(), 1);
  p.stop();
});

test("queued page changes survive suppression, marker writes do not notify, and later page edits still do", () => {
  const p = setup();
  const change = p.mutation(p.html, [p.text("页面更新的中文")]);
  p.observer.queue(change);
  assert.equal(p.changes.withoutObservation(() => {
    assert.equal(p.count(), 1, "queued page changes must be handled before disconnecting");
    p.observer.queue(change); // Simulate splitText while disconnected.
    return "written";
  }), "written");
  p.observer.deliver();
  assert.equal(p.count(), 1);
  p.observer.queue(change);
  p.observer.deliver();
  assert.equal(p.count(), 2);
  assert.throws(() => p.changes.withoutObservation(() => { throw new Error("write failed"); }), /write failed/);
  p.observer.queue(change);
  p.observer.deliver();
  assert.equal(p.count(), 3, "observation must resume even if writing throws");
  p.stop();
});

test("new shadow hosts request a read; discovered roots are observed and detached roots are released", () => {
  const p = setup();
  const host = p.element("div", p.html);
  const root = { nodeType: 11, host, isConnected: true, childNodes: [] };
  host.shadowRoot = root;
  p.observer.queue(p.mutation(p.html, [host]));
  p.observer.deliver();
  assert.equal(p.count(), 1, "even an empty new root must request discovery");
  p.changes.observeRoot(root);
  p.changes.observeRoot(root);
  assert.equal(p.observer.targets.size, 2);
  p.observer.queue({ type: "characterData", target: { nodeType: 3, parentNode: root, nodeValue: "评论内容更新" } });
  p.observer.deliver();
  assert.equal(p.count(), 2);
  host.parentElement = null;
  root.isConnected = false;
  p.observer.queue(p.mutation(p.html, [], [host]));
  p.observer.deliver();
  assert.equal(p.observer.targets.has(root), false);
  assert.equal(p.observer.targets.has(p.html), true);
  p.stop();
  assert.equal(p.observer.targets.size, 0);
  p.changes.observeRoot(root);
  assert.equal(p.observer.targets.size, 0);
  assert.equal(p.changes.withoutObservation(() => "off"), "off");
});

test("detached anchor and area URL hosts never become DOM ancestors", () => {
  const p = setup();
  for (const tag of ["a", "area"]) {
    const detached = p.element(tag);
    detached.host = "news.baidu.com";
    const node = p.text("移除的链接文字", detached);
    p.observer.queue({ type: "characterData", target: node });
    p.observer.queue(p.mutation(detached, [], [node]));
    assert.doesNotThrow(() => p.observer.deliver());
  }
  assert.equal(p.count(), 2);
  p.stop();
});
