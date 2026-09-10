const test = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const vm = require("node:vm");

const source = readFileSync(join(__dirname, "../src/frontend/processed-text.js"), "utf8");

test("processed text is identified by node identity, current value, and parent", () => {
  const context = vm.createContext({});
  vm.runInContext(source, context);
  const { isProcessedText, rememberProcessedText } = context.SuperReader;
  const parent = {};
  const node = { nodeValue: "相同文字", parentNode: parent };
  assert.equal(isProcessedText(node), false);
  rememberProcessedText([node]);
  assert.equal(isProcessedText(node), true);
  assert.equal(isProcessedText({ ...node }), false);
  node.nodeValue = "更新文字";
  assert.equal(isProcessedText(node), false);
  node.nodeValue = "相同文字";
  node.parentNode = {};
  assert.equal(isProcessedText(node), false);
});

test("script reinjection preserves processed text, and clearing resets it", () => {
  const context = vm.createContext({});
  vm.runInContext(source, context);
  const node = { nodeValue: "已处理", parentNode: {} };
  context.SuperReader.rememberProcessedText([node]);
  vm.runInContext(source, context);
  assert.equal(context.SuperReader.isProcessedText(node), true);
  context.SuperReader.clearProcessedText();
  assert.equal(context.SuperReader.isProcessedText(node), false);
});
