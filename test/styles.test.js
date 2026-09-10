const test = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");

const read = (path) => readFileSync(join(__dirname, "..", path), "utf8");

test("the toolbar is the only extension control", () => {
  const manifest = JSON.parse(read("manifest.json"));
  assert.equal(manifest.action.default_popup, undefined);
  assert.equal(manifest.commands, undefined);
  assert.equal(manifest.content_scripts, undefined);
  assert.equal(manifest.permissions.includes("storage"), false);
});

test("divider appearance lives in CSS without runtime style configuration", () => {
  const manifest = JSON.parse(read("manifest.json"));
  assert.ok(manifest.web_accessible_resources.some((entry) =>
    entry.resources.includes("src/content.css") && entry.matches.includes("<all_urls>")),
  "shadow roots must be able to load the shared extension stylesheet");
  const css = read("src/content.css");
  assert.match(css, /\.super-reader-divider::before/u);
  assert.match(css, /content:\s*""/u);
  assert.match(css, /border-inline-start:\s*3px solid #ff1744/u);
  for (const path of [
    "src/app/reader.js", "src/frontend/read.js", "src/frontend/write.js", "src/platform/chrome/background.js",
  ]) {
    assert.doesNotMatch(read(path), /dividerWidth|dividerColor|style\.setProperty/u);
  }
});

test("the sample article has no second reader implementation or settings UI", () => {
  assert.doesNotMatch(read("demo.html"), /<script|<input|<select/u);
});
