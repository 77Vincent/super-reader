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
    /border-inline-start:\s*var\(--super-reader-divider-width, 3px\) solid var\(--super-reader-divider-color, #ff1744\)/u,
  );
  assert.match(css, /margin:\s*0 0\.08em/u);
  assert.match(css, /vertical-align:\s*-0\.15em/u);
  assert.match(css, /filter:\s*var\(--super-reader-divider-filter, none\)/u);
  assert.match(css, /content:\s*""/u);
  assert.doesNotMatch(css, /opacity:\s*0\.[0-9]+/u);
  assert.doesNotMatch(css, /text-decoration|underline|chunk--a|chunk--b/u);
  assert.match(contentScript, /offsetsByText/u);
  assert.match(contentScript, /super-reader-chunk--separated/u);
  assert.doesNotMatch(contentScript, /splitUnderlineRuns|underlined|chunk--a|chunk--b/u);
});

test("extension chunks inline-formatted paragraph text as one model input", () => {
  const contentScript = readFileSync(join(projectRoot, "src/content.js"), "utf8");
  const fixture = readFileSync(join(projectRoot, "test/browser-fixture.html"), "utf8");

  assert.match(contentScript, /function collectTextRuns\(scope\)/u);
  assert.match(contentScript, /textNodes\.map\(\(textNode\) => textNode\.nodeValue\)\.join\(""\)/u);
  assert.match(contentScript, /requestDividerOffsets\(/u);
  assert.match(contentScript, /markerPlacements\(textNodes, offsets\)/u);
  assert.match(contentScript, /textNode\.splitText\(localOffset\)/u);
  assert.match(contentScript, /marker\.dataset\.superReaderDivider = "true"/u);
  assert.match(fixture, /id="inline-sample"[^>]*>[^<]*<strong>[^<]+<\/strong>/u);
});

test("extension schedules DOM reprocessing during browser idle time", () => {
  const contentScript = readFileSync(join(projectRoot, "src/content.js"), "utf8");

  assert.match(contentScript, /requestIdleCallback\(\(\) =>/u);
  assert.match(contentScript, /state\.processingJob/u);
  assert.match(contentScript, /MARKER_INSERT_BATCH_SIZE = 32/u);
  assert.match(contentScript, /cancelScheduledFlush\(\)/u);
  assert.doesNotMatch(contentScript, /queueMicrotask\(flushPendingRoots\)/u);
});

test("extension runs bounded model inference outside the page main thread", () => {
  const contentScript = readFileSync(join(projectRoot, "src/content.js"), "utf8");
  const backgroundScript = readFileSync(join(projectRoot, "src/background.js"), "utf8");
  const inferenceService = readFileSync(
    join(projectRoot, "src/inference-service.js"),
    "utf8",
  );
  const inferenceWorker = readFileSync(
    join(projectRoot, "src/inference-worker.js"),
    "utf8",
  );
  const chunker = readFileSync(join(projectRoot, "src/chunker.js"), "utf8");

  assert.match(contentScript, /chrome\.runtime\.sendMessage\(\{/u);
  assert.match(contentScript, /await requestDividerOffsets/u);
  assert.match(contentScript, /await waitForDomIdle\(\)/u);
  assert.doesNotMatch(contentScript, /SuperReaderChunker/u);
  assert.match(backgroundScript, /chrome\.offscreen\.createDocument/u);
  assert.match(backgroundScript, /reasons:\s*\["WORKERS"\]/u);
  assert.match(backgroundScript, /SUPER_READER_RUN_INFERENCE/u);
  assert.doesNotMatch(backgroundScript, /SuperReaderChunker|importScripts/u);
  assert.match(
    inferenceService,
    /new Worker\(\s*chrome\.runtime\.getURL\("src\/inference-worker\.js"\)/su,
  );
  assert.match(
    inferenceWorker,
    /importScripts\(\s*"boundary-model-data\.js",\s*"model-backend\.js",\s*"chunker\.js"/su,
  );
  assert.match(inferenceWorker, /SuperReaderChunker\.buildVisualChunks/u);
  assert.match(backgroundScript, /message\?\.type !== "SUPER_READER_SPLIT_TEXTS"/u);
  assert.match(chunker, /MAX_MODEL_WINDOW_TOKENS = 256/u);
  assert.match(chunker, /tokens\.slice\(scoreStart, scoreEnd\)/u);
});

test("extension defers text blocks until they enter the viewport", () => {
  const contentScript = readFileSync(join(projectRoot, "src/content.js"), "utf8");

  assert.match(contentScript, /new IntersectionObserver/u);
  assert.match(contentScript, /rootMargin:\s*"0px"/u);
  assert.match(contentScript, /if \(!entry\.isIntersecting\) return/u);
  assert.match(contentScript, /observeTextScopes\(document\.body \|\| document\.documentElement\)/u);
  assert.match(contentScript, /mutation\.addedNodes\.forEach\(observeTextScopes\)/u);
  assert.doesNotMatch(
    contentScript,
    /queueRoot\(document\.body \|\| document\.documentElement\)/u,
  );
});

test("extension observes and cleans up nested open Shadow DOM", () => {
  const contentScript = readFileSync(join(projectRoot, "src/content.js"), "utf8");
  const fixture = readFileSync(join(projectRoot, "test/browser-fixture.html"), "utf8");

  assert.match(contentScript, /shadowRoots: new Set\(\)/u);
  assert.match(contentScript, /root\.host\.shadowRoot === root/u);
  assert.match(contentScript, /observer\.observe\(target, OBSERVER_OPTIONS\)/u);
  assert.match(contentScript, /if \(!ignored && node\.shadowRoot\) visit\(node\.shadowRoot, true\)/u);
  assert.match(contentScript, /applyShadowMarkerLayout\(marker\)/u);
  assert.match(contentScript, /state\.shadowRoots\.clear\(\)/u);
  assert.equal((contentScript.match(/new MutationObserver/gu) || []).length, 1);
  assert.equal((fixture.match(/attachShadow\(\{ mode: "open" \}\)/gu) || []).length, 3);
  assert.match(fixture, /paragraph\.id = "dynamic-shadow-sample"/u);
});

test("HTML demo uses the same vertical separator treatment", () => {
  const html = readFileSync(join(projectRoot, "demo.html"), "utf8");

  assert.match(html, /\.reader-chunk--separated::before/u);
  assert.match(html, /height:\s*1em/u);
  assert.match(
    html,
    /border-inline-start:\s*var\(--reader-divider-width, 3px\) solid var\(--reader-divider-color, #ff1744\)/u,
  );
  assert.match(html, /margin:\s*0 0\.08em/u);
  assert.match(html, /vertical-align:\s*-0\.15em/u);
  assert.match(html, /filter:\s*var\(--reader-divider-filter, none\)/u);
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
  const popupScript = readFileSync(join(projectRoot, "popup/popup.js"), "utf8");
  const contentScript = readFileSync(join(projectRoot, "src/content.js"), "utf8");
  const backgroundScript = readFileSync(join(projectRoot, "src/background.js"), "utf8");

  for (const source of [demo, popup]) {
    assert.match(
      source,
      /id="divider-width"[^>]*min="1"[^>]*max="5"[^>]*step="1"/u,
    );
    assert.match(source, /id="divider-width"[^>]*value="3"/u);
  }
  for (const script of [popupScript, contentScript, backgroundScript]) {
    assert.match(script, /dividerWidth:\s*3/u);
  }
  assert.match(contentScript, /--super-reader-divider-width/u);
  assert.match(contentScript, /Math\.min\(5, Math\.max\(1, Math\.round\(width\)\)\)/u);
  assert.match(popup, /class="preview" aria-label="效果预览"/u);
  assert.match(popupScript, /--super-reader-divider-width/u);
  assert.match(popupScript, /--super-reader-divider-color/u);
});

test("popup preview is rendered by the shared model backend", () => {
  const popup = readFileSync(join(projectRoot, "popup/popup.html"), "utf8");
  const popupScript = readFileSync(join(projectRoot, "popup/popup.js"), "utf8");

  assert.match(
    popup,
    /class="preview"[^>]*>将长句切分成短句加速阅读理解<\/div>/u,
  );
  assert.doesNotMatch(popup, /<span[^>]*preview-chunk--separated/u);
  assert.doesNotMatch(popup, /boundary-model-data|model-backend|chunker\.js/u);
  assert.match(
    popupScript,
    /type:\s*"SUPER_READER_SPLIT_TEXTS"/u,
  );
  assert.match(popupScript, /releaseWhenDisabled:\s*true/u);
  assert.match(popupScript, /response\?\.offsetsByText/u);
  assert.doesNotMatch(popupScript, /SuperReaderChunker/u);
  assert.match(
    popupScript,
    /\(async function initializePopup\(\) \{\s*const previewPromise = renderModelPreview\(\);\s*const settings = await chrome\.storage\.sync\.get\(DEFAULTS\)/u,
  );
});

test("demo and extension expose five matching divider colors with red as default", () => {
  const demo = readFileSync(join(projectRoot, "demo.html"), "utf8");
  const popup = readFileSync(join(projectRoot, "popup/popup.html"), "utf8");
  const popupScript = readFileSync(join(projectRoot, "popup/popup.js"), "utf8");
  const contentScript = readFileSync(join(projectRoot, "src/content.js"), "utf8");
  const backgroundScript = readFileSync(join(projectRoot, "src/background.js"), "utf8");
  const colors = {
    red: "#ff1744",
    yellow: "#ffd600",
    blue: "#2979ff",
    green: "#00e676",
    text: "currentColor",
  };

  for (const markup of [demo, popup]) {
    const options = markup.match(/name="divider-color"/gu) || [];
    assert.equal(options.length, 5);
    for (const name of Object.keys(colors)) {
      assert.match(markup, new RegExp(`name="divider-color" value="${name}"`, "u"));
    }
    assert.match(markup, /name="divider-color" value="red" checked/u);
  }

  for (const script of [demo, contentScript]) {
    for (const [name, value] of Object.entries(colors)) {
      assert.match(script, new RegExp(`${name}: "${value}"`, "u"));
    }
    assert.match(script, /drop-shadow/u);
  }
  for (const [name, value] of Object.entries(colors)) {
    assert.match(popupScript, new RegExp(`${name}: "${value}"`, "u"));
  }
  for (const script of [popupScript, contentScript, backgroundScript]) {
    assert.match(script, /dividerColor:\s*"red"/u);
  }
  assert.match(contentScript, /changes\.dividerColor/u);
  assert.match(popupScript, /chrome\.storage\.sync\.set\(\{ dividerColor:/u);
});

test("extension loads its small page runtime without dynamic extension imports", () => {
  const manifest = JSON.parse(readFileSync(join(projectRoot, "manifest.json"), "utf8"));
  const popupScript = readFileSync(join(projectRoot, "popup/popup.js"), "utf8");

  assert.deepEqual(manifest.content_scripts[0].js, ["src/content.js"]);
  assert.equal(manifest.web_accessible_resources, undefined);
  assert.equal(manifest.minimum_chrome_version, "109");
  assert.ok(manifest.permissions.includes("offscreen"));
  assert.match(popupScript, /files:\s*\["src\/content\.js"\]/u);
  assert.doesNotMatch(popupScript, /src\/loader\.js/u);
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
