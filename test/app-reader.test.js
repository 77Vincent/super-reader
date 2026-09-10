const test = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const vm = require("node:vm");

const drain = () => new Promise((resolve) => setImmediate(resolve));
const source = readFileSync(join(__dirname, "../src/app/reader.js"), "utf8");

// Load one factory without DOM, timers, model, or extension APIs. Instances created
// from this same factory must keep their lifecycle state independent.
const context = vm.createContext({ Error });
vm.runInContext(source, context, { filename: "src/app/reader.js" });

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function createReader({
  read = () => ({ texts: [] }),
  process = async () => [],
  write = () => {},
  clear = () => {},
  publishState = () => {},
} = {}) {
  const states = [];
  let clears = 0;
  const reader = context.SuperReader.createReader({
    read,
    process,
    write,
    clear() { clears += 1; clear(); },
    publishState(state) { states.push({ ...state }); publishState(state); },
  });
  return { reader, states, clears: () => clears };
}

test("construction starts disabled without invoking dependencies", async () => {
  const calls = [];
  const record = () => calls.push("called");
  const { reader, states, clears } = createReader({
    read: record, process: record, write: record, clear: record, publishState: record,
  });

  await drain();
  assert.deepEqual({ ...reader.status() }, { enabled: false, busy: false, error: null });
  assert.deepEqual(calls, []);
  assert.deepEqual(states, []);
  assert.equal(clears(), 0);
});

test("status, toggle results, and notifications are snapshots of private state", async () => {
  const notifications = [];
  const { reader } = createReader({
    publishState(state) { notifications.push(state); },
  });
  const initial = reader.status();
  const editable = reader.status();
  Object.assign(editable, { enabled: true, busy: true, error: "external change" });
  assert.deepEqual({ ...reader.status() }, { enabled: false, busy: false, error: null });

  const started = reader.toggle();
  Object.assign(started, { enabled: false, busy: false, error: "external change" });
  assert.deepEqual({ ...reader.status() }, { enabled: true, busy: true, error: null });
  await drain();

  assert.deepEqual({ ...initial }, { enabled: false, busy: false, error: null });
  assert.deepEqual({ ...notifications[0] }, { enabled: true, busy: true, error: null });
  Object.assign(notifications.at(-1), { enabled: false, busy: true, error: "external change" });
  assert.deepEqual({ ...reader.status() }, { enabled: true, busy: false, error: null });
});

test("reader stays locked through the complete processing and writing, then allows cleanup", async () => {
  const processing = deferred();
  const writing = deferred();
  const snapshot = { texts: ["第一段文字", "第二段文字"], mapping: {} };
  const results = [[2], [3]];
  const calls = [];
  const { reader, states, clears } = createReader({
    read() {
      assert.equal(reader.status().busy, true);
      calls.push("read");
      return snapshot;
    },
    async process(texts) {
      assert.equal(texts, snapshot.texts);
      calls.push("process");
      await processing.promise;
      return results;
    },
    write(sourceSnapshot, offsetsByText) {
      assert.equal(sourceSnapshot, snapshot);
      assert.equal(offsetsByText, results);
      calls.push("write");
      return writing.promise;
    },
  });

  assert.deepEqual({ ...reader.status() }, { enabled: false, busy: false, error: null });
  assert.deepEqual(calls, []);
  assert.deepEqual({ ...reader.toggle() }, { enabled: true, busy: true, error: null });
  assert.deepEqual(calls, ["read", "process"]);

  for (const phase of ["processing", "writing"]) {
    await drain();
    for (let click = 0; click < 3; click += 1) {
      assert.deepEqual({ ...reader.toggle() }, { enabled: true, busy: true, error: null });
    }
    assert.deepEqual(calls, phase === "processing" ? ["read", "process"] : ["read", "process", "write"]);
    assert.equal(clears(), 0);
    assert.deepEqual(states, [{ enabled: true, busy: true, error: null }]);
    if (phase === "processing") processing.resolve();
  }

  assert.deepEqual(calls, ["read", "process", "write"]);
  writing.resolve();
  await drain();
  assert.deepEqual({ ...reader.status() }, { enabled: true, busy: false, error: null });
  assert.equal(states.length, 2);
  assert.deepEqual({ ...reader.toggle() }, { enabled: false, busy: false, error: null });
  assert.equal(clears(), 1);
  assert.deepEqual(states.at(-1), { enabled: false, busy: false, error: null });
});

test("the lock also blocks toggles made during the initial state notification", async () => {
  let reads = 0;
  const nestedToggles = [];
  const { reader, states, clears } = createReader({
    read() { reads += 1; return { texts: [] }; },
    publishState(state) {
      if (state.busy) nestedToggles.push({ ...reader.toggle() });
    },
  });

  reader.toggle();
  await drain();
  assert.equal(reads, 1);
  assert.equal(clears(), 0);
  assert.deepEqual(nestedToggles, [{ enabled: true, busy: true, error: null }]);
  assert.deepEqual(states, [
    { enabled: true, busy: true, error: null },
    { enabled: true, busy: false, error: null },
  ]);
});

test("disable clears before notifying, and re-enabling reads a fresh snapshot", async () => {
  const snapshots = [{ texts: ["原来的文字"] }, { texts: ["更新的文字"] }];
  const written = [];
  const events = [];
  let reads = 0;
  const { reader, clears } = createReader({
    read() { return snapshots[reads++]; },
    async process(texts) { return texts.map(() => []); },
    write(snapshot, results) { written.push({ snapshot, results }); },
    clear() { events.push("clear"); },
    publishState(state) { if (!state.enabled) events.push("disabled"); },
  });

  reader.toggle();
  await drain();
  assert.equal(written[0].snapshot, snapshots[0]);
  assert.deepEqual(written[0].results, [[]]);
  assert.equal(clears(), 0);

  assert.deepEqual({ ...reader.toggle() }, { enabled: false, busy: false, error: null });
  assert.deepEqual(events, ["clear", "disabled"]);
  await drain();
  assert.equal(reads, 1);
  assert.equal(written.length, 1);

  reader.toggle();
  await drain();
  assert.equal(reads, 2);
  assert.equal(written.length, 2);
  assert.equal(written[1].snapshot, snapshots[1]);
  assert.equal(clears(), 1);
  assert.deepEqual({ ...reader.status() }, { enabled: true, busy: false, error: null });
});

const stages = ["read", "process", "write"];
for (const [failedStage, mode] of [
  ["read", "throws"],
  ["process", "throws"],
  ["process", "rejects"],
  ["write", "throws"],
  ["write", "rejects"],
]) {
  test(`reader stops, retains the error, and allows retry after ${failedStage} ${mode}`, async () => {
    let shouldFail = true;
    const calls = [];
    const cleanupStates = [];
    const failure = deferred();
    function visit(stage) {
      calls.push(stage);
      if (shouldFail && stage === failedStage) {
        if (mode === "throws") throw new Error(`${stage} failed`);
        return failure.promise;
      }
    }
    const { reader, states, clears } = createReader({
      read() { visit("read"); return { texts: ["一段文字"] }; },
      process() { return visit("process") || Promise.resolve([[2]]); },
      write() { return visit("write"); },
      clear() { cleanupStates.push({ ...reader.status() }); },
    });

    reader.toggle();
    if (mode === "rejects") {
      await drain();
      assert.deepEqual({ ...reader.toggle() }, { enabled: true, busy: true, error: null });
      assert.equal(clears(), 0);
      failure.reject(new Error(`${failedStage} failed`));
    }
    await drain();
    assert.deepEqual(calls, stages.slice(0, stages.indexOf(failedStage) + 1));
    assert.equal(clears(), 1);
    const failedState = { enabled: false, busy: false, error: `${failedStage} failed` };
    assert.deepEqual(states.at(-1), failedState);
    assert.deepEqual({ ...reader.status() }, failedState);
    assert.deepEqual(cleanupStates, [{ ...failedState, busy: true }]);
    assert.deepEqual(states, [
      { enabled: true, busy: true, error: null },
      failedState,
    ]);

    // Failure does not schedule an automatic retry or discard the error on read.
    await drain();
    assert.deepEqual(calls, stages.slice(0, stages.indexOf(failedStage) + 1));
    assert.deepEqual({ ...reader.status() }, failedState);

    shouldFail = false;
    calls.length = 0;
    assert.deepEqual({ ...reader.toggle() }, { enabled: true, busy: true, error: null });
    await drain();
    assert.deepEqual(calls, stages);
    assert.equal(clears(), 1);
    assert.deepEqual({ ...reader.status() }, { enabled: true, busy: false, error: null });
    assert.deepEqual(states.at(-1), { enabled: true, busy: false, error: null });
  });
}

test("non-Error rejections are retained as a readable error message", async () => {
  const failure = deferred();
  const { reader, states, clears } = createReader({ process: () => failure.promise });

  reader.toggle();
  failure.reject("worker disconnected");
  await drain();
  const failedState = { enabled: false, busy: false, error: "worker disconnected" };
  assert.deepEqual({ ...reader.status() }, failedState);
  assert.deepEqual(states.at(-1), failedState);
  assert.equal(clears(), 1);
});

test("reader instances isolate enabled state, busy locks, failures, and cleanup", async () => {
  const processing = deferred();
  const first = createReader({ process: () => processing.promise });
  const second = createReader();

  first.reader.toggle();
  assert.deepEqual({ ...second.reader.status() }, { enabled: false, busy: false, error: null });
  second.reader.toggle();
  await drain();
  assert.deepEqual({ ...first.reader.status() }, { enabled: true, busy: true, error: null });
  assert.deepEqual({ ...second.reader.status() }, { enabled: true, busy: false, error: null });

  second.reader.toggle();
  assert.equal(first.clears(), 0);
  assert.equal(second.clears(), 1);
  processing.reject(new Error("first reader failed"));
  await drain();
  assert.equal(first.clears(), 1);
  assert.deepEqual({ ...second.reader.status() }, { enabled: false, busy: false, error: null });

  second.reader.toggle();
  await drain();
  assert.deepEqual({ ...second.reader.status() }, { enabled: true, busy: false, error: null });
  assert.deepEqual({ ...first.reader.status() }, {
    enabled: false, busy: false, error: "first reader failed",
  });
});
