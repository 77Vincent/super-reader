(function startInference() {
  "use strict";
  // Shared inference-host startup; the loaded adapter supplies the implementation.
  const adapter = globalThis.SuperReader.createInferenceAdapter();
  const service = globalThis.SuperReader.createInferenceService({ workerUrl: adapter.workerUrl });
  adapter.connect(service);
})();
