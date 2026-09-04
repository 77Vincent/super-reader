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
});
