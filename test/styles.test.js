const test = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");

const projectRoot = join(__dirname, "..");

test("extension renders thin vertical separators without underlines", () => {
  const css = readFileSync(join(projectRoot, "src/content.css"), "utf8");
  const contentScript = readFileSync(join(projectRoot, "src/content.js"), "utf8");

  assert.match(css, /\.super-reader-chunk--separated::before/u);
  assert.match(css, /height:\s*1em/u);
  assert.match(
    css,
    /border-inline-start:\s*var\(--super-reader-divider-width, 2px\) solid currentColor/u,
  );
  assert.match(css, /margin:\s*0 0\.08em/u);
  assert.match(css, /content:\s*""/u);
  assert.doesNotMatch(css, /opacity:\s*0\.[0-9]+/u);
  assert.doesNotMatch(css, /text-decoration|underline|chunk--a|chunk--b/u);
  assert.match(contentScript, /chunk\.separated/u);
  assert.match(contentScript, /super-reader-chunk--separated/u);
  assert.doesNotMatch(contentScript, /splitUnderlineRuns|underlined|chunk--a|chunk--b/u);
});

test("HTML demo uses the same vertical separator treatment", () => {
  const html = readFileSync(join(projectRoot, "demo.html"), "utf8");

  assert.match(html, /\.reader-chunk--separated::before/u);
  assert.match(html, /height:\s*1em/u);
  assert.match(
    html,
    /border-inline-start:\s*var\(--reader-divider-width, 2px\) solid currentColor/u,
  );
  assert.match(html, /margin:\s*0 0\.08em/u);
  assert.match(html, /reader-chunk--separated/u);
  assert.doesNotMatch(html, /\.reader-chunk--separated::before\s*\{[^}]*opacity:\s*0\.[0-9]+/su);
  assert.doesNotMatch(
    html,
    /text-decoration-(?:line|thickness|color|style)|splitUnderlineRuns|reader-chunk--a|reader-chunk--b/u,
  );
  assert.ok(html.indexOf("src/boundary-model-data.js") < html.indexOf("src/model-backend.js"));
  assert.ok(html.indexOf("src/model-backend.js") < html.indexOf("src/chunker.js"));
});

test("demo and popup expose the same divider-width range", () => {
  const demo = readFileSync(join(projectRoot, "demo.html"), "utf8");
  const popup = readFileSync(join(projectRoot, "popup/popup.html"), "utf8");
  const contentScript = readFileSync(join(projectRoot, "src/content.js"), "utf8");

  for (const source of [demo, popup]) {
    assert.match(
      source,
      /id="divider-width"[^>]*min="1"[^>]*max="4"[^>]*step="0\.5"/u,
    );
    assert.match(source, /id="divider-width"[^>]*value="2"/u);
  }
  assert.match(contentScript, /dividerWidth:\s*2/u);
  assert.match(contentScript, /--super-reader-divider-width/u);
  assert.match(contentScript, /Math\.min\(4, Math\.max\(1, width\)\)/u);
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
