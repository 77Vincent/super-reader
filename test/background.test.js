const test = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const vm = require("node:vm");

const projectRoot = join(__dirname, "..");

function runScript(path, context) {
  vm.runInContext(readFileSync(join(projectRoot, path), "utf8"), context, {
    filename: path,
  });
}

test("background lazily creates one offscreen inference context", async () => {
  let inferenceListener = null;
  let offscreenOpen = false;
  let createParameters = null;
  let createCount = 0;
  let forwardedMessage = null;
  let storageChangeListener = null;
  const noopEvent = { addListener() {} };
  const context = vm.createContext({ console });
  context.chrome = {
    action: {
      setBadgeBackgroundColor() {},
      setBadgeText() {},
      setTitle() {},
    },
    commands: { onCommand: noopEvent },
    offscreen: {
      async closeDocument() {
        offscreenOpen = false;
      },
      async createDocument(parameters) {
        createParameters = parameters;
        createCount += 1;
        offscreenOpen = true;
      },
    },
    runtime: {
      async getContexts() {
        return offscreenOpen ? [{}] : [];
      },
      getURL: (path) => `chrome-extension://test-extension/${path}`,
      id: "test-extension",
      onInstalled: noopEvent,
      onMessage: {
        addListener(listener) {
          inferenceListener = listener;
        },
      },
      onStartup: noopEvent,
      async sendMessage(message) {
        forwardedMessage = message;
        return { offsetsByText: [[8]] };
      },
    },
    storage: {
      onChanged: {
        addListener(listener) {
          storageChangeListener = listener;
        },
      },
      sync: {
        async get(defaults) {
          return defaults;
        },
        async remove() {},
        async set() {},
      },
    },
  };

  runScript("src/background.js", context);
  assert.equal(typeof inferenceListener, "function");

  const requestInference = () => new Promise((resolve) => {
    const listenerResult = inferenceListener({
      type: "SUPER_READER_SPLIT_TEXTS",
      texts: ["将长句切分成短句加速阅读理解"],
    }, { id: "test-extension" }, resolve);
    assert.equal(listenerResult, true);
  });
  const [response] = await Promise.all([requestInference(), requestInference()]);

  assert.deepEqual(JSON.parse(JSON.stringify(createParameters)), {
    url: "src/inference.html",
    reasons: ["WORKERS"],
    justification: "Run the bundled Chinese boundary model outside web pages",
  });
  assert.equal(forwardedMessage.target, "offscreen");
  assert.equal(forwardedMessage.type, "SUPER_READER_RUN_INFERENCE");
  assert.equal(createCount, 1);
  assert.equal(response.error, undefined);
  assert.deepEqual(Array.from(response.offsetsByText[0]), [8]);

  storageChangeListener({ enabled: { newValue: false } }, "sync");
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(offscreenOpen, false);
});

test("offscreen service correlates background requests with worker results", async () => {
  let messageListener = null;
  let workerPath = null;
  class FakeWorker {
    constructor(path) {
      workerPath = path;
    }

    postMessage(message) {
      queueMicrotask(() => {
        this.onmessage({ data: { id: message.id, offsetsByText: [[8]] } });
      });
    }

    terminate() {}
  }

  const context = vm.createContext({ console, queueMicrotask, Worker: FakeWorker });
  context.chrome = {
    runtime: {
      id: "test-extension",
      onMessage: {
        addListener(listener) {
          messageListener = listener;
        },
      },
    },
  };

  runScript("src/inference-service.js", context);
  const response = await new Promise((resolve) => {
    const listenerResult = messageListener(
      {
        target: "offscreen",
        type: "SUPER_READER_RUN_INFERENCE",
        texts: ["将长句切分成短句加速阅读理解"],
      },
      { id: "test-extension" },
      resolve,
    );
    assert.equal(listenerResult, true);
  });

  assert.equal(workerPath, "inference-worker.js");
  assert.deepEqual(Array.from(response.offsetsByText[0]), [8]);
});

test("dedicated worker performs model inference", () => {
  let workerResponse = null;
  const context = vm.createContext({ atob, btoa, console, Intl });
  context.importScripts = (...paths) => {
    paths.forEach((path) => runScript(`src/${path}`, context));
  };
  context.postMessage = (message) => {
    workerResponse = message;
  };
  context.self = context;

  runScript("src/inference-worker.js", context);
  context.self.onmessage({
    data: {
      id: 1,
      texts: ["将长句切分成短句加速阅读理解"],
    },
  });

  assert.equal(workerResponse.error, undefined);
  assert.equal(workerResponse.id, 1);
  assert.deepEqual(Array.from(workerResponse.offsetsByText[0]), [8]);
});
