"use strict";

globalThis.SuperReader ??= {};

/**
 * Application state, independent of the host's message format.
 * @typedef {Object} ReaderState
 * @property {boolean} enabled
 * @property {boolean} busy
 * @property {string | null} error
 */

/**
 * @typedef {Object} Reader
 * @property {() => ReaderState} status
 * @property {(enabled: boolean) => ReaderState} toggle
 * Apply a desired state while idle. Busy requests and matching states do
 * nothing. Repeated ON preserves a failure until OFF then ON.
 */

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
 * @property {(state: ReaderState) => void} publishState Report state only;
 * the adapter handles delivery errors and never changes reader state here.
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
}) {
  let enabled = false;
  let busy = false;
  let error = null;
  let refreshPending = false;
  let refreshTimer = null;
  let stopWatching = null;
  const DEBOUNCE_MS = 200;

  function status() {
    return { enabled, busy, error };
  }

  function refreshWhenReady() {
    if (!enabled || busy || !refreshPending || refreshTimer !== null) return;
    // Clear before starting, so changes during this operation can request another.
    refreshPending = false;
    void runTask();
  }

  function requestRefresh() {
    if (!enabled) return;
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

  async function runTask() {
    // One lock covers the complete snapshot, including reading and writing.
    busy = true;
    publishState(status());

    try {
      const snapshot = read();
      if (snapshot.texts.length > 0) {
        const results = await process(snapshot.texts);
        const written = write(snapshot, results);
        // Keep synchronous writes and recording together, before page observers
        // can change these nodes. Asynchronous writers still settle first.
        remember(written && typeof written.then === "function" ? await written : written);
      }
    } catch (failure) {
      error = failure instanceof Error ? failure.message : String(failure);
      enabled = false;
      stopRefreshes();
      clear();
      reset();
    } finally {
      busy = false;
      publishState(status());
      refreshWhenReady();
    }
  }

  function toggle(value) {
    if (busy) return status();
    if (value && error) return status();
    if (value === enabled && !error) return status();

    error = null;
    if (value) {
      enabled = true;
      stopWatching = watch(requestRefresh);
      void runTask();
    } else {
      if (enabled) {
        enabled = false;
        stopRefreshes();
        clear();
        reset();
      }
      publishState(status());
    }
    return status();
  }

  return { status, toggle };
};
