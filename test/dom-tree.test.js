const test = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const vm = require("node:vm");

const context = vm.createContext({});
vm.runInContext(readFileSync(join(__dirname, "../src/frontend/dom-tree.js"), "utf8"), context);
const { parentElementAcrossRoots } = context.SuperReader;

test("parent lookup follows slots, ordinary parents, and shadow roots", () => {
  const parent = { nodeType: 1 };
  const slot = { nodeType: 1 };
  const host = { nodeType: 1 };
  const shadowRoot = { nodeType: 11, host };
  assert.equal(parentElementAcrossRoots({ nodeType: 3, parentElement: parent }), parent);
  assert.equal(parentElementAcrossRoots({ nodeType: 1, parentElement: parent, assignedSlot: slot }), slot);
  assert.equal(parentElementAcrossRoots({ nodeType: 3, parentNode: shadowRoot }), host);
  assert.equal(parentElementAcrossRoots(shadowRoot), host);
});

test("non-shadow nodes cannot supply a host, and ordinary fragments have no parent element", () => {
  const anchor = { nodeType: 1, host: "news.example.test", parentElement: null, parentNode: null };
  const fragment = { nodeType: 11 };
  assert.equal(parentElementAcrossRoots(anchor), null);
  assert.equal(parentElementAcrossRoots({ nodeType: 3, parentElement: anchor, parentNode: anchor }), anchor);
  assert.equal(parentElementAcrossRoots(fragment), null);
  assert.equal(parentElementAcrossRoots({ nodeType: 3, parentNode: fragment }), null);
  assert.equal(parentElementAcrossRoots({ nodeType: 3, parentNode: null }), null);
  assert.equal(parentElementAcrossRoots({ nodeType: 1, parentNode: { nodeType: 9, host: "not a shadow host" } }), null);
});
