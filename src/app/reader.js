"use strict";

globalThis.SuperReader ??= {};

/**
 * A fixed viewport snapshot. The DOM layer retains its mapping in this object;
 * only texts are passed to segmentation.
 * @typedef {Object} ViewportSnapshot
 * @property {string[]} texts Text entries in result order.
 */

/**
 * Utilities supplied by startup when the individual layers are ready.
 * @typedef {Object} ReaderDependencies
 * @property {() => ViewportSnapshot} read DOM layer: capture visible text
 * and its mapping synchronously.
 * @property {(texts: string[]) => Promise<number[][]>} process Segmentation layer:
 * return offsets for every text in one operation; reject on failure.
 * @property {(snapshot: ViewportSnapshot, results: number[][]) => (unknown | Promise<unknown>)} write
 * DOM layer: check source validity and apply offsets. Settle only after all writes
 * have stopped. Both asynchronous operations reject on failure.
 * @property {() => void} clear DOM layer: synchronously remove markers.
 * @property {(written: unknown) => void} remember Record successfully handled data.
 * @property {() => void} reset Forget previously handled data.
 * @property {(onChange: () => void) => (() => void)} watch Subscribe to input
 * changes; return a function that removes the subscription.
 * @property {() => boolean} [isAvailable] Whether the host context still exists.
 * @property {(state: ReaderState) => void} publishState Adapter: deliver state;
 * handle delivery errors inside the adapter.
 */

/**
 * Owns one viewport's read/process/write lifecycle and the toggle lock.
 * DOM handling and inference live in their own layers.
 * @param {ReaderDependencies} dependencies
 * @returns {Reader}
 */
globalThis.SuperReader.createReader = function createReader({
  read,
  process,
  write,
  clear,
  remember,
  reset,
  watch,
  publishState,
  isAvailable = () => true,
}) {
  let enabled = false;
  let busy = false;
  let error = null;
  let refreshPending = false;
  let refreshTimer = null;
  let stopWatching = null;
  let disposed = false;
  const DEBOUNCE_MS = 200;

  function status() {
    return { enabled, busy, error };
  }

  function isActive() {
    if (disposed) return false;
    if (isAvailable()) return true;
    dispose();
    return false;
  }

  function refreshWhenReady() {
    if (!enabled || busy || !refreshPending || refreshTimer !== null) return;
    // Clear before starting, so changes during this operation can request another.
    refreshPending = false;
    void runTask();
  }

  function requestRefresh() {
    if (!isActive() || !enabled) return;
    refreshPending = true;
    clearTimeout(refreshTimer);
    refreshTimer = setTimeout(() => {
      refreshTimer = null;
      refreshWhenReady();
    }, DEBOUNCE_MS);
  }

  function stopRefreshes() {
    stopWatching?.();
    stopWatching = null;
    clearTimeout(refreshTimer);
    refreshTimer = null;
    refreshPending = false;
  }

  // Host teardown is permanent, unlike the user toggle. It stops callbacks and
  // discards any late result; it does not attempt to cancel model computation.
  function dispose() {
    if (disposed) return;
    disposed = true;
    enabled = false;
    stopRefreshes();
    clear();
    reset();
  }

  async function runTask() {
    if (!isActive()) return;
    // One lock covers the complete snapshot, including reading and writing.
    busy = true;
    publishState(status());

    try {
      const snapshot = read();
      if (snapshot.texts.length > 0) {
        const results = await process(snapshot.texts);
        if (!isActive()) return;
        const written = write(snapshot, results);
        // Keep synchronous writes and recording together, before page observers
        // can change these nodes. Asynchronous writers still settle first.
        const completed = written && typeof written.then === "function" ? await written : written;
        if (isActive()) remember(completed);
      }
    } catch (failure) {
      if (!isActive()) return;
      error = failure instanceof Error ? failure.message : String(failure);
      enabled = false;
      stopRefreshes();
      clear();
      reset();
    } finally {
      busy = false;
      if (isActive()) {
        publishState({ ...status() });
        refreshWhenReady();
      }
    }
  }

  function toggle() {
    if (!isActive() || busy) return status();

    error = null;
    enabled = !enabled;
    if (enabled) {
      stopWatching = watch(requestRefresh);
      // Keep toggle synchronous for the adapter; completion is published separately.
      void runTask();
    } else {
      stopRefreshes();
      clear();
      reset();
      publishState(status());
    }
    return status();
  }

  return { status, toggle, dispose };
};
