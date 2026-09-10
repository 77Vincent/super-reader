// Shared contracts, expressed as JSDoc so the project stays plain JavaScript.
// This file contains no runtime code and does not need script injection.

/**
 * @typedef {Object} ReaderState
 * @property {boolean} enabled
 * @property {boolean} busy
 * @property {string | null} [error]
 */

/**
 * @typedef {Object} Reader
 * @property {() => ReaderState} status
 * @property {() => ReaderState} toggle
 * @property {() => void} dispose Permanently stop this instance on host teardown.
 */

/**
 * Host services used by the reader. No browser message types cross this boundary.
 * @typedef {Object} ReaderServices
 * @property {(texts: string[]) => Promise<number[][]>} process One viewport; rejects on failure.
 * @property {(state: ReaderState) => void} publishState Handles delivery errors itself.
 * @property {() => boolean} [isAvailable] False when the host context is gone.
 */

/**
 * connect() attaches the host's controls to the reader; startup calls it once.
 * @typedef {ReaderServices & {markerStyleUrl: string, connect: (reader: Reader) => void}} ReaderAdapter
 * markerStyleUrl supplies the stylesheet URL for markers inside shadow roots.
 */

/**
 * @typedef {Object} InferenceService
 * @property {(texts: string[]) => Promise<number[][]>} runInference Rejects on failure.
 */

/**
 * @typedef {Object} InferenceAdapter
 * @property {string} workerUrl A URL loadable by the standard Worker constructor.
 * @property {(service: InferenceService) => void} connect Routes host requests and responses.
 */
