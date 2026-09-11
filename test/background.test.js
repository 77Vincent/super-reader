const test = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const vm = require("node:vm");

const runScript = (path, context) => vm.runInContext(
  readFileSync(join(__dirname, "..", path), "utf8"), context, { filename: path },
);
const drain = () => new Promise((resolve) => setImmediate(resolve));

function createBackground(saved = {}) {
  let click, message, navigate, activate, focusWindow, startup, install;
  let activeId = null;
  let offscreen = false;
  let creations = 0;
  const tabs = new Map();
  const pages = new Map();
  const badges = new Map();
  const icons = new Map();
  const titles = new Map();
  const disabled = new Set();
  const commands = [];
  const queries = [];
  const writes = [];
  const injected = [];
  const injectedFiles = [];
  const requests = [];
  const injectionDelays = new Map();
  const statusDelays = new Map();
  const settingDelays = new Map();

  function publish(request, tabId = 1) {
    return new Promise((resolve) => message(request,
      { id: "test-extension", tab: { id: tabId }, frameId: 0 }, resolve));
  }

  // Use the real reader and Chrome page adapter. Only DOM work and inference
  // are stubbed, so focus/toggle handling exercises both sides of the protocol.
  function injectPage(id) {
    let listener;
    const page = { reads: 0, clears: 0, marker: null, process: async () => [[]] };
    const context = vm.createContext({ setTimeout, clearTimeout, chrome: { runtime: {
      getURL: (path) => path,
      onMessage: { addListener: (callback) => { listener = callback; } },
      sendMessage: (request) => publish(request, id),
    } } });
    runScript("src/app/reader.js", context);
    runScript("src/platform/chrome/content.js", context);
    const adapter = context.SuperReader.createReaderAdapter();
    page.reader = context.SuperReader.createReader({
      read() { page.reads += 1; return { texts: ["需要处理的中文"] }; },
      process: (texts) => page.process(texts),
      write() { page.marker = {}; },
      clear() { page.clears += 1; page.marker = null; },
      remember() {}, reset() {}, watch: () => () => {},
      publishState: adapter.publishState,
    });
    adapter.connect(page.reader);
    page.message = (request) => {
      let response;
      listener(request, {}, (value) => { response = value; });
      return response;
    };
    pages.set(id, page);
  }

  const chrome = {
    action: {
      onClicked: { addListener: (callback) => { click = callback; } },
      async setBadgeText({ tabId, text }) { badges.set(tabId, text); },
      async setIcon({ tabId, path }) {
        const resolved = {};
        // Chrome's service-worker binding fetches paths relative to the worker URL.
        for (const [size, imagePath] of Object.entries(path)) {
          const url = new URL(imagePath, chrome.runtime.getURL(manifest.background.service_worker));
          assert.equal(url.protocol, "chrome-extension:");
          assert.equal(url.hostname, chrome.runtime.id);
          const png = readFileSync(join(__dirname, "..", url.pathname.slice(1)));
          assert.equal(png.subarray(1, 4).toString(), "PNG");
          assert.equal(png.readUInt32BE(16), Number(size));
          assert.equal(png.readUInt32BE(20), Number(size));
          resolved[size] = url.href;
        }
        icons.set(tabId, resolved);
      },
      async setTitle({ tabId, title }) { titles.set(tabId, title); },
      async disable(id) { disabled.add(id); },
      async enable(id) { disabled.delete(id); },
    },
    tabs: {
      onUpdated: { addListener: (callback) => { navigate = callback; } },
      onActivated: { addListener: (callback) => { activate = callback; } },
      async query(query) {
        queries.push(query);
        assert.equal(query.active, true, "never enumerate inactive tabs");
        assert.equal(query.lastFocusedWindow, true);
        return activeId === null ? [] : [{ ...tabs.get(activeId) }];
      },
      async sendMessage(id, request, options) {
        assert.equal(options.frameId, 0);
        commands.push({ id, ...request });
        if (!pages.has(id)) throw new Error("No receiver");
        const response = pages.get(id).message(request);
        if (request.type === "SUPER_READER_PING" && statusDelays.has(id)) {
          const delay = statusDelays.get(id);
          statusDelays.delete(id);
          await delay;
        }
        if (request.type === "SUPER_READER_APPLY_SETTING" && settingDelays.has(id)) {
          const delay = settingDelays.get(id);
          settingDelays.delete(id);
          await delay;
        }
        return response;
      },
    },
    windows: {
      WINDOW_ID_NONE: -1,
      onFocusChanged: { addListener: (callback) => { focusWindow = callback; } },
    },
    storage: { local: {
      async get(defaults) { return { ...defaults, ...saved }; },
      async set(value) { Object.assign(saved, value); writes.push({ ...value }); },
    } },
    scripting: {
      async insertCSS() {},
      async executeScript({ target, files }) {
        injected.push(target.tabId);
        injectedFiles.push(Array.from(files));
        await injectionDelays.get(target.tabId);
        injectPage(target.tabId);
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
      onStartup: { addListener: (callback) => { startup = callback; } },
      onInstalled: { addListener: (callback) => { install = callback; } },
    },
  };
  const manifest = JSON.parse(readFileSync(join(__dirname, "../manifest.json"), "utf8"));
  const boot = () => runScript(manifest.background.service_worker, vm.createContext({ console, chrome }));
  boot();

  function addTab(id, url = "https://example.com/", status = "complete") {
    tabs.set(id, { id, url, status, active: false });
  }

  function select(id) {
    if (!tabs.has(id)) addTab(id);
    for (const tab of tabs.values()) tab.active = tab.id === id;
    activeId = id;
  }

  return {
    pages, badges, icons, titles, disabled, commands, queries, writes, saved, tabs, injected, injectedFiles, requests,
    addTab,
    pauseInjection(id) {
      let resume;
      injectionDelays.set(id, new Promise((resolve) => { resume = resolve; }));
      return resume;
    },
    pauseStatus(id) {
      let resume;
      statusDelays.set(id, new Promise((resolve) => { resume = resolve; }));
      return resume;
    },
    pauseSettingReply(id) {
      let resume;
      settingDelays.set(id, new Promise((resolve) => { resume = resolve; }));
      return resume;
    },
    creations: () => creations,
    async click(id) { select(id); click(tabs.get(id)); await drain(); },
    async focus(id) { select(id); activate({ tabId: id }); await drain(); },
    async windowFocus(id) { select(id); focusWindow(2); await drain(); },
    async navigate(id) {
      pages.delete(id);
      const tab = tabs.get(id);
      tab.status = "loading";
      navigate(id, { status: "loading" }, { ...tab });
      await drain();
      tab.status = "complete";
      navigate(id, { status: "complete" }, { ...tab });
      await drain();
    },
    async restartWorker() { boot(); await drain(); },
    async startup() { startup(); await drain(); },
    async install() { install(); await drain(); },
    message: publish,
  };
}

test("the saved switch is global but toggles and focus leave every inactive tab untouched", async () => {
  const background = createBackground();
  background.addTab(2);
  await background.click(1);
  assert.deepEqual(background.injected, [1]);
  assert.deepEqual(background.injectedFiles[0], [
    "src/frontend/dom-tree.js", "src/frontend/viewport.js", "src/frontend/visibility.js", "src/frontend/processed-text.js",
    "src/frontend/read.js", "src/frontend/write.js", "src/frontend/content-changes.js", "src/app/reader.js",
    "src/platform/chrome/content.js", "src/start-reader.js",
  ]);
  assert.equal(background.saved.enabled, true);
  assert.equal(background.pages.get(1).reader.status().enabled, true);
  assert.equal(background.pages.has(2), false);
  await background.focus(2);
  assert.equal(background.pages.get(2).reader.status().enabled, true);
  const first = background.pages.get(1);
  const marker = first.marker;
  await background.focus(1);
  await background.focus(2);
  assert.equal(first.marker, marker);
  assert.equal(first.reads, 1);
  const beforeToggle = background.commands.length;
  await background.click(2); // OFF only touches the focused tab.
  assert.equal(background.saved.enabled, false);
  assert.equal(background.pages.get(2).reader.status().enabled, false);
  assert.equal(first.reader.status().enabled, true);
  assert.equal(first.marker, marker);
  assert.ok(background.commands.slice(beforeToggle).every(({ id }) => id === 2));
  await background.focus(1);
  assert.equal(first.reader.status().enabled, false);
  assert.equal(first.marker, null);
  assert.deepEqual(background.injected, [1, 2]);
  assert.equal(background.badges.get(1), "");
  assert.equal(background.badges.get(2), "");
  assert.deepEqual(background.icons.get(1), {
    16: "chrome-extension://test-extension/icons/off-16.png",
    32: "chrome-extension://test-extension/icons/off-32.png",
  });
  assert.deepEqual(background.icons.get(2), background.icons.get(1));
});

test("active reloads and new tabs use the saved switch; background reloads wait for focus", async () => {
  const background = createBackground();
  await background.click(1);
  const oldPage = background.pages.get(1);
  await background.navigate(1);
  assert.notEqual(background.pages.get(1), oldPage);
  assert.equal(background.pages.get(1).reads, 1);
  assert.equal(background.badges.get(1), "");
  assert.deepEqual(background.icons.get(1), {
    16: "chrome-extension://test-extension/icons/on-16.png",
    32: "chrome-extension://test-extension/icons/on-32.png",
  });
  await background.focus(2);
  assert.equal(background.pages.get(2).reads, 1);
  await background.navigate(1);
  assert.equal(background.pages.has(1), false, "an inactive page reload must not initialize a reader");
  await background.focus(1);
  assert.equal(background.pages.get(1).reads, 1);
  await background.click(1);
  await background.navigate(1);
  await background.focus(3);
  assert.equal(background.pages.has(1), false);
  assert.equal(background.pages.has(3), false);
});

test("a tab that missed OFF and ON keeps its existing results when focused again", async () => {
  const background = createBackground();
  await background.click(1);
  const first = background.pages.get(1);
  const marker = first.marker;
  await background.focus(2);
  const beforeToggle = background.commands.length;
  await background.click(2);
  await background.click(2);
  assert.ok(background.commands.slice(beforeToggle).every(({ id }) => id === 2));
  assert.equal(first.reads, 1);
  assert.equal(first.marker, marker);
  assert.equal(first.clears, 0);
  await background.focus(1);
  assert.equal(first.reads, 1);
  assert.equal(first.clears, 0);
  assert.equal(first.marker, marker);
  await background.focus(1);
  assert.equal(first.reads, 1);
  assert.deepEqual(background.saved, { enabled: true });
});

test("worker and browser restarts retain the setting and window focus only checks its active tab", async () => {
  const background = createBackground();
  await background.click(1);
  const marker = background.pages.get(1).marker;
  await background.restartWorker();
  await background.focus(1);
  assert.equal(background.pages.get(1).marker, marker);
  await background.windowFocus(2);
  assert.equal(background.pages.get(2).reads, 1);
  assert.equal(background.saved.enabled, true);
  const restarted = createBackground(background.saved);
  await restarted.focus(1);
  await restarted.startup();
  await restarted.install();
  assert.equal(restarted.pages.get(1).reads, 1);
  assert.equal(restarted.badges.get(1), "");
  assert.deepEqual(restarted.icons.get(1), {
    16: "chrome-extension://test-extension/icons/on-16.png",
    32: "chrome-extension://test-extension/icons/on-32.png",
  });
  await restarted.click(1);
  const stopped = createBackground(background.saved);
  await stopped.focus(1);
  assert.equal(stopped.pages.size, 0);
});

test("busy clicks are ignored without flicker, and inactive pages are neither queried nor stopped", async () => {
  const background = createBackground();
  await background.click(1);
  const title = background.titles.get(1);
  const icon = background.icons.get(1);
  let complete;
  const first = background.pages.get(1);
  first.reader.toggle(false);
  first.process = () => new Promise((resolve) => { complete = resolve; });
  first.reader.toggle(true);
  await drain();
  const writes = background.writes.length;
  await background.click(1);
  await background.click(1);
  assert.equal(background.writes.length, writes, "busy clicks are discarded");
  assert.equal(background.badges.get(1), "");
  assert.deepEqual(background.icons.get(1), icon);
  assert.equal(background.titles.get(1), title);
  assert.equal(background.disabled.size, 0);
  await background.focus(2);
  const beforeToggle = background.commands.length;
  await background.click(2);
  assert.equal(background.saved.enabled, false);
  assert.equal(first.reader.status().busy, true);
  assert.ok(background.commands.slice(beforeToggle).every(({ id }) => id === 2));
  complete([[]]);
  await drain();
  assert.equal(first.reader.status().enabled, true, "inactive completion does not apply a new toggle");
  await background.focus(1);
  assert.equal(first.reader.status().enabled, false);
});

test("a busy tab ignores global OFF and synchronizes on its next idle focus", async () => {
  const background = createBackground();
  await background.click(1);
  const first = background.pages.get(1);
  let finish;
  first.reader.toggle(false);
  first.process = () => new Promise((resolve) => { finish = resolve; });
  first.reader.toggle(true);
  await background.focus(2);
  await background.click(2); // Global OFF.
  await background.focus(1); // The running reader ignores OFF.
  assert.equal(first.reader.status().busy, true);
  assert.equal(first.reader.status().enabled, true);
  finish([[]]);
  await drain();
  assert.equal(first.reader.status().enabled, true, "completion must not apply a discarded command");
  assert.equal(first.reader.status().busy, false);
  assert.equal(first.reads, 2);
  assert.equal(first.clears, 1);
  assert.equal(background.saved.enabled, false);
  await background.focus(2);
  await background.focus(1);
  assert.equal(first.reader.status().enabled, false);
  assert.equal(first.marker, null);
  assert.equal(first.clears, 2);
  assert.equal(background.badges.get(1), "");
});

test("page failures stay local and focusing again doesn't retry or flip the global switch", async () => {
  const background = createBackground();
  await background.click(1);
  const first = background.pages.get(1);
  first.reader.toggle(false);
  first.process = async () => { throw new Error("failed"); };
  first.reader.toggle(true);
  await drain();
  const reads = first.reads;
  assert.equal(background.saved.enabled, true);
  assert.equal(background.badges.get(1), "ERR");
  await background.focus(2);
  assert.equal(background.pages.get(2).reader.status().enabled, true);
  await background.click(2);
  await background.click(2);
  await background.focus(1);
  assert.equal(first.reads, reads);
  assert.equal(background.badges.get(1), "ERR");
  first.process = async () => [[]];
  await background.click(1);
  await background.click(1);
  assert.equal(first.reader.status().enabled, true);
  assert.equal(background.badges.get(1), "");
  assert.deepEqual(background.icons.get(1), {
    16: "chrome-extension://test-extension/icons/on-16.png",
    32: "chrome-extension://test-extension/icons/on-32.png",
  });
});

test("a delayed busy command reply cannot erase a newer inference error", async () => {
  const background = createBackground();
  await background.click(1);
  const first = background.pages.get(1);
  first.reader.toggle(false);
  first.process = async () => { throw new Error("model failed"); };
  const resume = background.pauseSettingReply(1);
  await background.focus(1);
  assert.match(first.reader.status().error, /model failed/u);
  assert.equal(background.badges.get(1), "ERR");
  resume();
  await drain();
  assert.equal(background.badges.get(1), "ERR");
  assert.match(background.titles.get(1), /model failed/u);
});

test("restricted pages allow switching the global preference without script injection", async () => {
  const background = createBackground();
  background.addTab(1, "chrome://extensions/");
  await background.click(1);
  assert.equal(background.saved.enabled, true);
  assert.equal(background.injected.length, 0);
  await background.focus(2);
  assert.equal(background.pages.get(2).reader.status().enabled, true);
});

test("switching away during script loading leaves that tab inert until it is focused again", async () => {
  const background = createBackground({ enabled: true });
  const resume = background.pauseInjection(1);
  await background.focus(1);
  await background.focus(2);
  resume();
  await drain();
  assert.equal(background.pages.get(1).reads, 0);
  assert.equal(background.pages.get(2).reads, 1);
  assert.ok(!background.commands.some(({ id, type }) => id === 1 && type === "SUPER_READER_APPLY_SETTING"));
  await background.focus(1);
  assert.equal(background.pages.get(1).reads, 1);
  assert.equal(background.injected.filter((id) => id === 1).length, 1);
});

test("a reload finishing during an old status check gets a fresh check without another focus event", async () => {
  const background = createBackground();
  await background.click(1);
  const oldPage = background.pages.get(1);
  const resume = background.pauseStatus(1);
  await background.focus(1);
  await background.navigate(1);
  resume();
  await drain();
  assert.ok(background.pages.has(1), "the replacement document must receive its own reader");
  assert.notEqual(background.pages.get(1), oldPage);
  assert.equal(background.pages.get(1).reads, 1);
  assert.deepEqual(background.injected, [1, 1]);
});

test("overlapping focus events recheck only the latest active tab", async () => {
  const background = createBackground();
  await background.click(1);
  const resume = background.pauseStatus(1);
  await background.focus(1);
  await background.focus(2);
  await background.focus(3);
  resume();
  await drain();
  assert.equal(background.pages.has(2), false, "intermediate tabs must not be queued for processing");
  assert.equal(background.pages.get(3).reads, 1);
  assert.equal(background.pages.get(1).reads, 1);
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
  const texts = [
    sampleText(), `${sampleText()}。`.repeat(12), "", "English only", "𠀀甲乙丙丁戊己庚辛壬癸子丑",
    "我一直在思考明天早上的早餐吃什么".repeat(33),
  ];
  context.onmessage({ data: { id: 1, texts } });
  assert.equal(response.error, undefined);
  assert.equal(replies, 1);
  assert.equal(response.offsetsByText.length, texts.length);
  assert.deepEqual(Array.from(response.offsetsByText[0]), [8]);
  assert.deepEqual(Array.from(response.offsetsByText[2]), []);
  assert.deepEqual(Array.from(response.offsetsByText[3]), []);
  assert.ok(response.offsetsByText[5].length > 1);
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
