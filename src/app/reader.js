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
 * @property {(snapshot: ViewportSnapshot, results: number[][]) => (void | Promise<void>)} write
 * DOM layer: check source validity and apply offsets. Settle only after all writes
 * have stopped. Both asynchronous operations reject on failure.
 * @property {() => void} clear DOM layer: synchronously remove markers.
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
  publishState,
}) {
  let enabled = false;
  let busy = false;
  let error = null;

  function status() {
    return { enabled, busy, error };
  }

  async function runTask() {
    // One lock covers the complete snapshot, including reading and writing.
    busy = true;
    publishState(status());

    try {
      const snapshot = read();
      const results = await process(snapshot.texts);
      await write(snapshot, results);
    } catch (failure) {
      error = failure instanceof Error ? failure.message : String(failure);
      enabled = false;
      clear();
    } finally {
      busy = false;
      publishState({ ...status() });
    }
  }

  function toggle() {
    if (busy) return status();

    error = null;
    enabled = !enabled;
    if (enabled) {
      // Keep toggle synchronous for the adapter; completion is published separately.
      void runTask();
    } else {
      clear();
      publishState(status());
    }
    return status();
  }

  return { status, toggle };
};
