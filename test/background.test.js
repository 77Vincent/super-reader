const test = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const vm = require("node:vm");

const runScript = (path, context) => vm.runInContext(
  readFileSync(join(__dirname, "..", path), "utf8"), context, { filename: path },
);
const drain = () => new Promise((resolve) => setImmediate(resolve));

function createBackground() {
  let click, message, navigate;
  let offscreen = false;
  let creations = 0;
  const pages = new Map();
  const badges = new Map();
  const disabled = new Set();
  const injected = [];
  const injectedFiles = [];
  const requests = [];
  const context = vm.createContext({ console });
  context.chrome = {
    action: {
      onClicked: { addListener: (callback) => { click = callback; } },
      async setBadgeText({ tabId, text }) { badges.set(tabId, text); },
      async setTitle() {},
      async disable(id) { disabled.add(id); },
      async enable(id) { disabled.delete(id); },
    },
    tabs: {
      onUpdated: { addListener: (callback) => { navigate = callback; } },
      async sendMessage(id, request) {
        if (!pages.has(id)) throw new Error("No receiver");
        if (request.type === "SUPER_READER_TOGGLE") {
          pages.set(id, !pages.get(id));
          message({ type: "SUPER_READER_STATE", enabled: pages.get(id), busy: false },
            { id: "test-extension", tab: { id } }, () => {});
          // A command response may arrive after a newer state publication.
          return { enabled: pages.get(id), busy: true };
        }
        return { enabled: pages.get(id), busy: false };
      },
    },
    scripting: {
      async insertCSS() {},
      async executeScript({ target, files }) {
        injected.push(target.tabId);
        injectedFiles.push(Array.from(files));
        pages.set(target.tabId, false);
      },
    },
    offscreen: { async createDocument() { creations += 1; offscreen = true; } },
    runtime: {
      id: "test-extension",
      getURL: (path) => `chrome-extension://test-extension/${path}`,
      async getContexts() { return offscreen ? [{}] : []; },
      async sendMessage(request) {
        requests.push(request);
        return { offsetsByText: request.texts.map(() => [4]) };
      },
      onMessage: { addListener: (callback) => { message = callback; } },
    },
  };
  const manifest = JSON.parse(readFileSync(join(__dirname, "../manifest.json"), "utf8"));
  runScript(manifest.background.service_worker, context);
  return {
    pages, badges, disabled, injected, injectedFiles, requests,
    creations: () => creations,
    async click(id, url = "https://example.com/") { click({ id, url }); await drain(); },
    async navigate(id) { navigate(id, { status: "loading" }); await drain(); },
    message(request, tabId = 1) {
      return new Promise((resolve) => message(request, { id: "test-extension", tab: { id: tabId } }, resolve));
    },
  };
}

test("toolbar clicks inject only the current page and toggle each page independently", async () => {
  const background = createBackground();
  await background.click(1);
  assert.deepEqual(background.injected, [1]);
  assert.deepEqual(background.injectedFiles[0], [
    "src/frontend/viewport.js", "src/frontend/visibility.js", "src/frontend/processed-text.js",
    "src/frontend/read.js", "src/frontend/write.js", "src/app/reader.js",
    "src/platform/chrome/content.js", "src/start-reader.js",
  ]);
  assert.equal(background.pages.get(1), true);
  await background.click(2);
  await background.click(1);
  assert.equal(background.pages.get(1), false);
  assert.equal(background.pages.get(2), true);
  assert.deepEqual(background.injected, [1, 2]);
  assert.equal(background.badges.get(1), "");
  assert.equal(background.badges.get(2), "ON");
  assert.equal(background.disabled.size, 0, "stale toggle responses must not relock completed pages");
  await background.navigate(2);
  assert.equal(background.badges.get(2), "");
});

test("viewport status locks only its own toolbar button and errors unlock it", async () => {
  const background = createBackground();
  await background.message({ type: "SUPER_READER_STATE", enabled: true, busy: true });
  assert.equal(background.disabled.has(1), true);
  assert.equal(background.disabled.has(2), false);
  await background.message({ type: "SUPER_READER_STATE", enabled: false, busy: false, error: "failed" });
  assert.equal(background.disabled.has(1), false);
  assert.equal(background.badges.get(1), "ERR");
  await background.click(3, "chrome://extensions/");
  assert.equal(background.pages.has(3), false);
});

test("whole viewports are forwarded intact and concurrent pages share one lazy offscreen context", async () => {
  const background = createBackground();
  assert.equal(background.creations(), 0);
  const request = { type: "SUPER_READER_PROCESS", texts: ["这里是一段需要切分的中文文本", "甲".repeat(500)] };
  const results = await Promise.all([background.message(request, 1), background.message(request, 2)]);
  assert.equal(background.creations(), 1);
  results.forEach((result) => assert.deepEqual(result.offsetsByText, [[4], [4]]));
  assert.equal(background.requests.length, 2);
  assert.ok(background.requests.every(({ texts }) => texts === request.texts));
});

test("offscreen requests match their results; timeouts terminate the worker and allow retry", async () => {
  let listener;
  const workers = [];
  const timers = new Map();
  let nextTimer = 1;
  class FakeWorker {
    constructor() { workers.push(this); this.messages = []; }
    postMessage(message) { this.messages.push(message); }
    terminate() { this.terminated = true; }
  }
  const context = vm.createContext({
    Worker: FakeWorker,
    setTimeout(callback) { const id = nextTimer++; timers.set(id, callback); return id; },
    clearTimeout: (id) => timers.delete(id),
    chrome: { runtime: {
      id: "test-extension", getURL: (path) => path,
      onMessage: { addListener: (callback) => { listener = callback; } },
    } },
  });
  runScript("src/inference-service.js", context);
  runScript("src/platform/chrome/inference.js", context);
  runScript("src/start-inference.js", context);
  const request = () => new Promise((resolve) => listener({
    target: "offscreen", type: "SUPER_READER_RUN_INFERENCE", texts: [sampleText()],
  }, { id: "test-extension" }, resolve));
  const first = request();
  const second = request();
  assert.equal(workers.length, 1);
  const [one, two] = workers[0].messages;
  workers[0].onmessage({ data: { id: two.id, offsetsByText: [[8]] } });
  assert.equal((await second).offsetsByText[0][0], 8);
  workers[0].onmessage({ data: { id: one.id, offsetsByText: [[4]] } });
  assert.equal((await first).offsetsByText[0][0], 4);
  assert.equal(timers.size, 0);
  const timeout = request();
  timers.values().next().value();
  assert.match((await timeout).error, /timed out/u);
  assert.equal(workers[0].terminated, true);
  const retry = request();
  assert.equal(workers.length, 2);
  workers[1].onmessage({ data: { id: workers[1].messages[0].id, offsetsByText: [[]] } });
  assert.equal((await retry).error, undefined);
  assert.equal(timers.size, 0);
});

function sampleText() { return "将长句切分成短句加速阅读理解"; }

test("the inference service sends a whole viewport once and validates the data interface", async () => {
  const urls = [];
  const requests = [];
  class FakeWorker {
    constructor(url) { urls.push(url); }
    postMessage({ id, texts }) {
      requests.push(texts);
      this.onmessage({ data: { id, offsetsByText: texts.map(() => [4]) } });
    }
  }
  const context = vm.createContext({ Worker: FakeWorker, setTimeout, clearTimeout });
  runScript("src/inference-service.js", context);
  let service;
  context.SuperReader.createInferenceAdapter = () => ({
    workerUrl: "./inference-worker.js",
    connect: (instance) => { service = instance; },
  });
  runScript("src/start-inference.js", context);
  await assert.rejects(service.runInference([42]), /invalid inference input/u);
  await assert.rejects(service.runInference("text"), /invalid inference input/u);
  assert.equal((await service.runInference([])).length, 0);
  assert.equal(urls.length, 0);
  const texts = [sampleText(), "甲".repeat(500)];
  assert.deepEqual(await service.runInference(texts), [[4], [4]]);
  assert.equal(requests.length, 1);
  assert.equal(requests[0], texts);
  assert.deepEqual(await service.runInference([sampleText()]), [[4]]);
  assert.deepEqual(urls, ["./inference-worker.js"]);
});

test("the actual worker returns ordered model results for a whole viewport including long strings", () => {
  let response;
  let replies = 0;
  const context = vm.createContext({ atob, btoa, console, Intl });
  context.importScripts = (...paths) => paths.forEach((path) => runScript(`src/${path}`, context));
  context.postMessage = (message) => { response = message; replies += 1; };
  context.self = context;
  runScript("src/inference-worker.js", context);
  const texts = [sampleText(), `${sampleText()}。`.repeat(12), "", "English only", "𠀀甲乙丙丁戊己庚辛壬癸子丑"];
  context.onmessage({ data: { id: 1, texts } });
  assert.equal(response.error, undefined);
  assert.equal(replies, 1);
  assert.equal(response.offsetsByText.length, texts.length);
  assert.deepEqual(Array.from(response.offsetsByText[0]), [8]);
  assert.deepEqual(Array.from(response.offsetsByText[2]), []);
  assert.deepEqual(Array.from(response.offsetsByText[3]), []);
  response.offsetsByText.forEach((offsets, index) => {
    let previous = 0;
    for (const offset of offsets) {
      assert.ok(Number.isInteger(offset) && offset > previous && offset < texts[index].length);
      assert.doesNotMatch(texts[index][offset], /[\uDC00-\uDFFF]/u);
      previous = offset;
    }
  });
  context.onmessage({ data: { id: 2, texts: null } });
  assert.ok(response.error);
});
