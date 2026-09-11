"use strict";

// Chrome entry point: toolbar, script loading, routing, and offscreen host.
const INFERENCE_PAGE = "src/inference.html";
let creatingInferencePage = null;
let enabled = false;
let toggling = false;
let checkingActiveTab = false;
let checkPending = false;
const settingsReady = chrome.storage.local.get({ enabled: false }).then((saved) => {
  enabled = saved.enabled;
  return updateAction();
});

function updateAction(tabId, error) {
  // Busy is a click lock, not a visual state.
  const title = error || (enabled ? "Super Reader（点击关闭）" : "Super Reader（点击开启）");
  const icon = enabled ? "on" : "off";
  return Promise.all([
    chrome.action.setIcon({ tabId, path: {
      // setIcon fetches relative to this worker, which lives in a subdirectory.
      16: chrome.runtime.getURL(`icons/${icon}-16.png`),
      32: chrome.runtime.getURL(`icons/${icon}-32.png`),
    } }),
    chrome.action.setBadgeText({ tabId, text: error ? "ERR" : "" }),
    chrome.action.setTitle({ tabId, title }),
  ]);
}

async function ensureInferencePage() {
  if (creatingInferencePage) return creatingInferencePage;
  creatingInferencePage = (async () => {
    const url = chrome.runtime.getURL(INFERENCE_PAGE);
    const contexts = typeof chrome.runtime.getContexts === "function"
      ? await chrome.runtime.getContexts({ contextTypes: ["OFFSCREEN_DOCUMENT"], documentUrls: [url] })
      : (await clients.matchAll()).filter((client) => client.url === url);
    if (!contexts.length) {
      await chrome.offscreen.createDocument({
        url: INFERENCE_PAGE, reasons: ["WORKERS"],
        justification: "Run the bundled boundary model outside web pages",
      });
    }
  })();
  try {
    await creatingInferencePage;
  } finally {
    creatingInferencePage = null;
  }
}

async function runInference(texts) {
  await ensureInferencePage();
  return chrome.runtime.sendMessage({
    target: "offscreen", type: "SUPER_READER_RUN_INFERENCE", texts,
  });
}

async function readerStatus(tabId) {
  try {
    return await chrome.tabs.sendMessage(tabId, { type: "SUPER_READER_PING" }, { frameId: 0 });
  } catch (_) { /* The document may not have a reader yet. */ }
}

function supportsReader(tab) {
  return tab?.id != null && !tab.discarded && tab.status !== "loading" && /^(https?|file):/u.test(tab.url || "");
}

async function activeTab() {
  const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  return tab;
}

async function applyToTab(tab) {
  if (!supportsReader(tab)) return;
  try {
    const ready = await readerStatus(tab.id);
    if (!ready) {
      if (!enabled) return;
      await chrome.scripting.insertCSS({ target: { tabId: tab.id }, files: ["src/content.css"] });
      await chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files: [
          "src/frontend/dom-tree.js", "src/frontend/viewport.js", "src/frontend/visibility.js", "src/frontend/processed-text.js",
          "src/frontend/read.js", "src/frontend/write.js", "src/app/reader.js",
          "src/platform/chrome/content.js", "src/start-reader.js",
        ],
      });
    }
    // Injection can outlast a tab switch. Leave that page inert until visited.
    if ((await activeTab())?.id !== tab.id) return;
    const state = await chrome.tabs.sendMessage(tab.id, {
      type: "SUPER_READER_APPLY_SETTING", enabled,
    }, { frameId: 0 });
    // A successful reply can predate an error already published by the reader.
    // Only restore a retained error here; transitions publish their own state.
    if (state?.error) await updateAction(tab.id, enabled ? state.error : null);
  } catch (error) {
    await updateAction(tab.id, error.message).catch(() => {});
  }
}

async function applyToActiveTab() {
  checkPending = true;
  if (checkingActiveTab) return;
  checkingActiveTab = true;
  try {
    await settingsReady;
    while (checkPending) {
      checkPending = false;
      // A reload or focus event during a check requests another check of the
      // latest document in the focused tab. Intermediate tabs aren't queued.
      const tab = await activeTab();
      if (!tab) continue;
      await updateAction(tab.id);
      await applyToTab(tab);
    }
  } finally {
    checkingActiveTab = false;
  }
}

async function toggleGlobal(tab) {
  if (toggling) return;
  toggling = true;
  try {
    await settingsReady;
    // Only the clicked page's task locks this click. Other tabs apply the new
    // setting when visited, without interrupting their current operation.
    if ((await readerStatus(tab.id))?.busy) return;
    const next = !enabled;
    await chrome.storage.local.set({ enabled: next });
    enabled = next;
    await updateAction();
    await applyToActiveTab();
  } catch (error) {
    await updateAction(tab?.id, error.message).catch(() => {});
  } finally {
    toggling = false;
  }
}

chrome.action.onClicked.addListener((tab) => { void toggleGlobal(tab); });
chrome.tabs.onActivated.addListener(() => { void applyToActiveTab().catch(() => {}); });
chrome.windows.onFocusChanged.addListener((windowId) => {
  if (windowId !== chrome.windows.WINDOW_ID_NONE) void applyToActiveTab().catch(() => {});
});
chrome.tabs.onUpdated.addListener((tabId, change, tab) => {
  if (change.status === "loading") void settingsReady.then(() => updateAction(tabId)).catch(() => {});
  if (change.status === "complete" && tab.active) void applyToActiveTab().catch(() => {});
});
chrome.runtime.onInstalled.addListener(() => { void applyToActiveTab().catch(() => {}); });
chrome.runtime.onStartup.addListener(() => { void applyToActiveTab().catch(() => {}); });
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (sender.id !== chrome.runtime.id || sender.tab?.id == null || sender.frameId !== 0) return;
  if (message?.type === "SUPER_READER_STATE") {
    // Page failures don't change the saved global preference or other pages.
    void settingsReady.then(() => updateAction(sender.tab.id, enabled ? message.error : null)).catch(() => {});
    sendResponse({});
  }
  if (message?.type !== "SUPER_READER_PROCESS") return;
  void runInference(message.texts).then(sendResponse, (error) => sendResponse({ error: error.message }));
  return true;
});
