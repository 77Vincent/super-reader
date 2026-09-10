const test = require("node:test");
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");
const vm = require("node:vm");

const drain = () => new Promise((resolve) => setImmediate(resolve));
const source = readFileSync(join(__dirname, "../src/app/reader.js"), "utf8");

// Load one factory without DOM, model, or extension APIs. Instances created
// from this same factory must keep their lifecycle state independent.
const context = vm.createContext({ Error, setTimeout, clearTimeout });
vm.runInContext(source, context, { filename: "src/app/reader.js" });

function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

function createClock() {
  let now = 0;
  let nextId = 0;
  const timers = new Map();
  return {
    setTimeout(callback, delay) {
      const id = ++nextId;
      timers.set(id, { at: now + delay, callback });
      return id;
    },
    clearTimeout(id) { timers.delete(id); },
    tick(duration) {
      const end = now + duration;
      while (true) {
        const next = [...timers].sort((a, b) => a[1].at - b[1].at)[0];
        if (!next || next[1].at > end) break;
        const [id, timer] = next;
        now = timer.at;
        timers.delete(id);
        timer.callback();
      }
      now = end;
    },
    pending: () => timers.size,
  };
}

function createWatchedReader(dependencies = {}) {
  const clock = createClock();
  let onChange;
  let starts = 0;
  let stops = 0;
  const page = createReader({
    ...dependencies,
    clock,
    watch(callback) {
      starts += 1;
      onChange = callback;
      return () => { stops += 1; };
    },
  });
  return { ...page, clock, change: () => onChange(), starts: () => starts, stops: () => stops };
}

function createReader({
  read = () => ({ texts: ["测试文字"] }),
  process = async () => [],
  write = () => {},
  clear = () => {},
  watch = () => () => {},
  clock,
  publishState = () => {},
} = {}) {
  const states = [];
  let clears = 0;
  let factory = context.SuperReader.createReader;
  if (clock) {
    const timedContext = vm.createContext({ Error, setTimeout: clock.setTimeout, clearTimeout: clock.clearTimeout });
    vm.runInContext(source, timedContext);
    factory = timedContext.SuperReader.createReader;
  }
  const reader = factory({
    read,
    process,
    write,
    watch,
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

test("rapid input changes produce one refresh after 200 ms, reading only the latest input", async () => {
  let currentText = "初始画面";
  const inputs = [];
  const page = createWatchedReader({
    read() { inputs.push(currentText); return { texts: [currentText] }; },
  });
  assert.equal(page.starts(), 0);
  page.reader.toggle();
  await drain();
  assert.equal(page.starts(), 1);

  currentText = "中间画面";
  page.change();
  page.clock.tick(100);
  currentText = "最后画面";
  page.change();
  page.clock.tick(199);
  assert.deepEqual(inputs, ["初始画面"]);
  assert.equal(page.reader.status().busy, false);
  assert.equal(page.clock.pending(), 1);
  page.clock.tick(1);
  await drain();
  assert.deepEqual(inputs, ["初始画面", "最后画面"]);
  page.clock.tick(1000);
  assert.equal(inputs.length, 2);
  assert.equal(page.clock.pending(), 0);
});

test("a settled refresh waits for both processing and writing, then captures the latest input", async () => {
  const processing = deferred();
  const writing = deferred();
  let currentText = "旧画面";
  const inputs = [];
  let writes = 0;
  const page = createWatchedReader({
    read() { inputs.push(currentText); return { texts: [currentText] }; },
    process: () => inputs.length === 1 ? processing.promise : Promise.resolve([[]]),
    write: () => ++writes === 1 ? writing.promise : undefined,
  });
  page.reader.toggle();
  currentText = "变化后的画面";
  page.change();
  page.change();
  page.clock.tick(200);
  assert.deepEqual(inputs, ["旧画面"]);
  processing.resolve([[]]);
  await drain();
  assert.equal(page.reader.status().busy, true);
  assert.equal(inputs.length, 1);
  currentText = "实际开始时的画面";
  writing.resolve();
  await drain();
  assert.deepEqual(inputs, ["旧画面", "实际开始时的画面"]);
  assert.equal(writes, 2);
  assert.equal(page.reader.status().busy, false);
});

test("finishing a task does not bypass a debounce timer that is still running", async () => {
  const processing = deferred();
  let reads = 0;
  const page = createWatchedReader({
    read() { reads += 1; return { texts: ["文字"] }; },
    process: () => reads === 1 ? processing.promise : Promise.resolve([[]]),
  });
  page.reader.toggle();
  page.change();
  page.clock.tick(50);
  processing.resolve([[]]);
  await drain();
  assert.equal(page.reader.status().busy, false);
  assert.equal(reads, 1);
  page.clock.tick(149);
  assert.equal(reads, 1);
  page.clock.tick(1);
  await drain();
  assert.equal(reads, 2);
});

test("a new change restarts the settling period even after an earlier timer expired while busy", async () => {
  const processing = deferred();
  let reads = 0;
  const page = createWatchedReader({
    read() { reads += 1; return { texts: ["文字"] }; },
    process: () => reads === 1 ? processing.promise : Promise.resolve([[]]),
  });
  page.reader.toggle();
  page.change();
  page.clock.tick(200);
  page.change();
  processing.resolve([[]]);
  await drain();
  assert.equal(reads, 1);
  page.clock.tick(199);
  assert.equal(reads, 1);
  page.clock.tick(1);
  await drain();
  assert.equal(reads, 2);
});

test("changes during the next operation can request one further run without overlapping work", async () => {
  const second = deferred();
  let reads = 0;
  let active = 0;
  const page = createWatchedReader({
    read() { reads += 1; return { texts: ["文字"] }; },
    async process() {
      active += 1;
      assert.equal(active, 1);
      if (reads === 2) await second.promise;
      active -= 1;
      return [[]];
    },
  });
  page.reader.toggle();
  await drain();
  page.change();
  page.clock.tick(200);
  assert.equal(reads, 2);
  page.change();
  page.change();
  page.clock.tick(200);
  assert.equal(reads, 2);
  second.resolve();
  await drain();
  assert.equal(reads, 3);
  assert.equal(active, 0);
  page.clock.tick(1000);
  assert.equal(reads, 3);
});

test("disable removes the subscription and pending timer; re-enable starts clean", async () => {
  let reads = 0;
  const page = createWatchedReader({ read() { reads += 1; return { texts: ["文字"] }; } });
  page.reader.toggle();
  await drain();
  page.change();
  page.clock.tick(100);
  page.reader.toggle();
  assert.equal(page.stops(), 1);
  assert.equal(page.clock.pending(), 0);
  page.change(); // A late notification after unsubscribe must also be harmless.
  page.clock.tick(1000);
  assert.equal(reads, 1);
  page.reader.toggle();
  await drain();
  page.clock.tick(1000);
  assert.equal(page.starts(), 2);
  assert.equal(reads, 2);
});

for (const elapsed of [100, 200]) {
  test(`failure discards a pending refresh after ${elapsed} ms and permits an explicit retry`, async () => {
    const processing = deferred();
    let reads = 0;
    const page = createWatchedReader({
      read() { reads += 1; return { texts: ["文字"] }; },
      process: () => reads === 1 ? processing.promise : Promise.resolve([[]]),
    });
    page.reader.toggle();
    page.change();
    page.clock.tick(elapsed);
    processing.reject(new Error("计算失败"));
    await drain();
    assert.deepEqual({ ...page.reader.status() }, { enabled: false, busy: false, error: "计算失败" });
    assert.equal(page.stops(), 1);
    assert.equal(page.clock.pending(), 0);
    page.change();
    page.clock.tick(1000);
    assert.equal(reads, 1);
    page.reader.toggle();
    await drain();
    page.clock.tick(1000);
    assert.equal(reads, 2);
  });
}

test("empty snapshots complete without inference or writing and remain subscribed", async () => {
  let calls = 0;
  const page = createWatchedReader({
    read: () => ({ texts: [] }),
    process: () => { calls += 1; },
    write: () => { calls += 1; },
  });
  page.reader.toggle();
  page.change();
  page.clock.tick(200);
  await drain();
  assert.equal(calls, 0);
  assert.equal(page.reader.status().enabled, true);
  assert.equal(page.reader.status().busy, false);
  assert.equal(page.starts(), 1);
  assert.equal(page.stops(), 0);
});
