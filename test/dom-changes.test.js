const test = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const vm = require("node:vm");

function harness() {
  let observer;
  let changes = 0;
  const roots = new Set();
  class MutationObserver {
    constructor(callback) { this.callback = callback; this.records = []; observer = this; }
    observe(root) { roots.add(root); }
    disconnect() { roots.clear(); this.records = []; }
    takeRecords() { const records = this.records; this.records = []; return records; }
  }
  const element = (ignored = false) => ({ nodeType: 1, isConnected: true, matches: () => ignored, children: [] });
  const document = { documentElement: element(), defaultView: { MutationObserver } };
  const context = vm.createContext({ document });
  for (const name of ["dom-tree.js", "changes.js"]) {
    vm.runInContext(readFileSync(join(__dirname, "../src/frontend", name), "utf8"), context);
  }
  // Discovery is tested with native DOM separately; here test subscription behavior.
  context.SuperReader.walkDOM = function* walk(node, skip) {
    if (node.nodeType === 1 && skip(node)) return;
    yield node;
    if (node.shadowRoot) yield* walk(node.shadowRoot, skip);
    for (const child of node.children || []) yield* walk(child, skip);
  };
  const monitor = context.SuperReader.createDOMChanges(document);
  const stop = monitor.watch(() => { changes += 1; });
  return {
    monitor, stop, roots, element, document, changes: () => changes,
    emit: (records) => observer.callback(records),
    queue: (record) => observer.records.push(record),
    shadow(host = element()) {
      const root = Object.assign(new EventTarget(), { nodeType: 11, host, children: [] });
      host.shadowRoot = root;
      return root;
    },
  };
}
const text = (nodeValue = "新评论内容", parentNode = null) => ({ nodeType: 3, nodeValue, parentNode, parentElement: parentNode });
const added = (target, ...nodes) => ({ type: "childList", target, addedNodes: nodes, removedNodes: [] });

test("new nested shadow roots are observed and their scroll events stop after unsubscribe", () => {
  const page = harness();
  const outer = page.shadow();
  const inner = page.shadow();
  outer.children.push(inner.host);
  inner.children.push(text());
  page.emit([added(page.document.documentElement, outer.host)]);
  assert.equal(page.changes(), 1);
  assert.ok(page.roots.has(outer) && page.roots.has(inner));
  inner.dispatchEvent(new Event("scroll"));
  inner.dispatchEvent(new Event("slotchange"));
  assert.equal(page.changes(), 3);
  page.stop();
  inner.dispatchEvent(new Event("scroll"));
  assert.equal(page.changes(), 3);
  assert.equal(page.roots.size, 0);
});

test("text and visibility changes notify, while scripts and irrelevant text do not", () => {
  const page = harness();
  page.emit([added(page.document.documentElement, text("12345"))]);
  page.emit([{ type: "characterData", target: text("中文脚本内容", page.element(true)) }]);
  assert.equal(page.changes(), 0);
  page.emit([{ type: "characterData", target: text() }]);
  const host = page.element();
  host.children.push(text());
  page.emit([{ type: "attributes", target: host }]);
  assert.equal(page.changes(), 2);
  page.stop();
});

test("marker writes pause observation but preserve already queued page changes", () => {
  const page = harness();
  const root = page.shadow();
  page.monitor.observeRoot(root);
  page.queue(added(page.document.documentElement, text()));
  const value = page.monitor.mutate(() => {
    assert.equal(page.changes(), 1);
    assert.equal(page.roots.size, 0);
    return "written nodes";
  });
  assert.equal(value, "written nodes");
  assert.equal(page.roots.size, 2);
  page.monitor.mutate(() => {});
  assert.equal(page.changes(), 1);
  assert.throws(() => page.monitor.mutate(() => { throw new Error("write failed"); }));
  assert.equal(page.roots.size, 2);
  page.stop();
  assert.equal(page.monitor.mutate(() => 42), 42);
});

test("removing a shadow host releases its observation and event listeners", () => {
  const page = harness();
  const root = page.shadow();
  page.monitor.observeRoot(root);
  root.host.isConnected = false;
  page.emit([{ type: "childList", target: page.document.documentElement, addedNodes: [], removedNodes: [root.host] }]);
  assert.equal(page.roots.size, 1);
  assert.equal(page.changes(), 1);
  root.dispatchEvent(new Event("scroll"));
  assert.equal(page.changes(), 1);
  page.stop();
});
