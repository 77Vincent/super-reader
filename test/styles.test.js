const test = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");

const read = (path) => readFileSync(join(__dirname, "..", path), "utf8");

test("the keyboard shortcut invokes the toolbar action without a popup", () => {
  const manifest = JSON.parse(read("manifest.json"));
  assert.equal(manifest.action.default_popup, undefined);
  assert.deepEqual(manifest.commands, {
    _execute_action: { suggested_key: "Alt+Shift+R" },
  });
  assert.equal(manifest.content_scripts, undefined);
  assert.equal(manifest.permissions.includes("storage"), true);
  assert.equal(manifest.permissions.includes("activeTab"), false);
  assert.deepEqual(manifest.host_permissions, ["http://*/*", "https://*/*", "file:///*"]);
});

test("divider appearance lives in CSS without runtime style configuration", () => {
  const manifest = JSON.parse(read("manifest.json"));
  assert.ok(manifest.web_accessible_resources.some((entry) =>
    entry.resources.includes("src/content.css") && entry.matches.includes("<all_urls>")),
  "shadow roots must be able to load the shared extension stylesheet");
  const css = read("src/content.css");
  assert.match(css, /\.super-reader-divider::before/u);
  assert.match(css, /content:\s*""/u);
  assert.match(css, /width:\s*calc\(1em \/ 6\)/u);
  assert.match(css, /height:\s*1em/u);
  assert.match(css, /background-color:\s*#ff1744/u);
  for (const path of [
    "src/app/reader.js", "src/frontend/read.js", "src/frontend/write.js", "src/platform/chrome/background.js",
  ]) {
    assert.doesNotMatch(read(path), /dividerWidth|dividerColor|style\.setProperty/u);
  }
});
