"use strict";

const DEFAULTS = {
  enabled: false,
  dividerWidth: 3,
  dividerColor: "red",
};
const OFFSCREEN_DOCUMENT_PATH = "src/inference.html";
const OFFSCREEN_DOCUMENT_URL = chrome.runtime.getURL(OFFSCREEN_DOCUMENT_PATH);
let offscreenLifecycle = Promise.resolve();

function runOffscreenLifecycleOperation(operation) {
  const result = offscreenLifecycle.then(operation, operation);
  offscreenLifecycle = result.catch(() => undefined);
  return result;
}

async function hasOffscreenDocument() {
  if (typeof chrome.runtime.getContexts === "function") {
    const contexts = await chrome.runtime.getContexts({
      contextTypes: ["OFFSCREEN_DOCUMENT"],
      documentUrls: [OFFSCREEN_DOCUMENT_URL],
    });
    return contexts.length > 0;
  }

  const matchedClients = await clients.matchAll();
  return matchedClients.some((client) => client.url === OFFSCREEN_DOCUMENT_URL);
}

function ensureOffscreenDocument() {
  return runOffscreenLifecycleOperation(async () => {
    if (await hasOffscreenDocument()) return;
    await chrome.offscreen.createDocument({
      url: OFFSCREEN_DOCUMENT_PATH,
      reasons: ["WORKERS"],
      justification: "Run the bundled Chinese boundary model outside web pages",
    });
  });
}

function closeInferenceDocument() {
  return runOffscreenLifecycleOperation(async () => {
    if (await hasOffscreenDocument()) await chrome.offscreen.closeDocument();
  });
}

async function runSharedInference(texts, releaseWhenDisabled = false) {
  try {
    await ensureOffscreenDocument();
    const response = await chrome.runtime.sendMessage({
      target: "offscreen",
      type: "SUPER_READER_RUN_INFERENCE",
      texts,
    });
    if (response?.error) throw new Error(response.error);
    if (!Array.isArray(response?.offsetsByText)) {
      throw new Error("Super Reader inference service returned an invalid result");
    }
    return response.offsetsByText;
  } finally {
    if (releaseWhenDisabled) {
      const { enabled } = await chrome.storage.sync.get({ enabled: false });
      if (!enabled) await closeInferenceDocument();
    }
  }
}

function updateBadge(enabled) {
  chrome.action.setBadgeText({ text: enabled ? "ON" : "" });
  chrome.action.setBadgeBackgroundColor({ color: "#4f46e5" });
  chrome.action.setTitle({
    title: enabled ? "Super Reader（已开启）" : "Super Reader（已关闭）",
  });
}

chrome.runtime.onInstalled.addListener(async () => {
  const current = await chrome.storage.sync.get(DEFAULTS);
  await chrome.storage.sync.set(current);
  await chrome.storage.sync.remove(["palette", "targetLength"]);
  updateBadge(current.enabled);
});

chrome.runtime.onStartup.addListener(async () => {
  const { enabled } = await chrome.storage.sync.get(DEFAULTS);
  updateBadge(enabled);
  if (!enabled) await closeInferenceDocument();
});

chrome.storage.onChanged.addListener((changes, areaName) => {
  if (areaName === "sync" && changes.enabled) {
    const enabled = Boolean(changes.enabled.newValue);
    updateBadge(enabled);
    if (!enabled) {
      void closeInferenceDocument().catch((error) => {
        console.error("Super Reader failed to stop inference", error);
      });
    }
  }
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type !== "SUPER_READER_SPLIT_TEXTS") return undefined;

  try {
    const texts = message.texts;
    if (
      sender.id !== chrome.runtime.id ||
      !Array.isArray(texts) ||
      texts.some((text) => typeof text !== "string")
    ) {
      throw new TypeError("Super Reader received invalid inference input");
    }
    void runSharedInference(texts, message.releaseWhenDisabled === true).then(
      (offsetsByText) => sendResponse({ offsetsByText }),
      (error) => sendResponse({
        error: error instanceof Error ? error.message : String(error),
      }),
    );
  } catch (error) {
    sendResponse({
      error: error instanceof Error ? error.message : String(error),
    });
    return false;
  }
  return true;
});

chrome.commands.onCommand.addListener(async (command) => {
  if (command !== "toggle-reader") return;
  const { enabled } = await chrome.storage.sync.get(DEFAULTS);
  await chrome.storage.sync.set({ enabled: !enabled });
});
