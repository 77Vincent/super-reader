const test = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");

const projectRoot = join(__dirname, "..");

test("extension alternates between underlined and plain chunks", () => {
  const css = readFileSync(join(projectRoot, "src/content.css"), "utf8");

  assert.match(css, /text-decoration-thickness:\s*3px/u);
  assert.match(css, /\.super-reader-chunk--a\s*\{[^}]*text-decoration-line:\s*underline/su);
  assert.doesNotMatch(css, /\.super-reader-chunk--a\s*\{[^}]*font-weight/su);
  assert.match(css, /\.super-reader-chunk--a\s*\{[^}]*text-decoration-color:\s*currentColor/su);
  assert.match(css, /\.super-reader-chunk--b\s*\{[^}]*text-decoration-line:\s*none/su);

  const contentScript = readFileSync(join(projectRoot, "src/content.js"), "utf8");
  assert.match(contentScript, /splitUnderlineRuns\(chunk\.text\)/u);
});

test("HTML demo uses the same underlined and plain rhythm", () => {
  const html = readFileSync(join(projectRoot, "demo.html"), "utf8");

  assert.match(html, /text-decoration-thickness:\s*3px/u);
  assert.match(html, /\.reader-chunk--b\s*\{[^}]*text-decoration-line:\s*none/su);
  assert.doesNotMatch(html, /\.reader-chunk--a\s*\{[^}]*font-weight/su);
  assert.match(html, /\.reader-chunk--a\s*\{[^}]*text-decoration-color:\s*currentColor/su);
  assert.match(html, /splitUnderlineRuns\(chunk\.text\)/u);
  assert.doesNotMatch(html, /data-palette|palette-button|下划线样式/u);
  assert.ok(html.indexOf("src/boundary-model-data.js") < html.indexOf("src/model-backend.js"));
  assert.ok(html.indexOf("src/model-backend.js") < html.indexOf("src/chunker.js"));
});

test("extension and demo load the same model backend", () => {
  const manifest = JSON.parse(readFileSync(join(projectRoot, "manifest.json"), "utf8"));

  assert.deepEqual(manifest.content_scripts[0].js, [
    "src/boundary-model-data.js",
    "src/model-backend.js",
    "src/chunker.js",
    "src/content.js",
  ]);
});

test("target chunk length is no longer configurable", () => {
  const demo = readFileSync(join(projectRoot, "demo.html"), "utf8");
  const popup = readFileSync(join(projectRoot, "popup/popup.html"), "utf8");
  const popupScript = readFileSync(join(projectRoot, "popup/popup.js"), "utf8");
  const contentScript = readFileSync(join(projectRoot, "src/content.js"), "utf8");

  for (const source of [demo, popup, popupScript, contentScript]) {
    assert.doesNotMatch(source, /targetLength|chunk-length|分块长度/u);
  }
});
