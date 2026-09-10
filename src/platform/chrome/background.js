"use strict";

// Chrome entry point: toolbar, script loading, routing, and offscreen host.
const INFERENCE_PAGE = "src/inference.html";
let creatingInferencePage = null;
const openingTabs = new Set();

function updateAction(tabId, { enabled = false, busy = false, error } = {}) {
  const title = error || (busy
    ? "Super Reader（正在处理）"
    : enabled ? "Super Reader（点击关闭）" : "Super Reader（点击开启）");
  return Promise.all([
    chrome.action.setBadgeText({ tabId, text: error ? "ERR" : busy ? "…" : enabled ? "ON" : "" }),
    chrome.action.setTitle({ tabId, title }),
    busy ? chrome.action.disable(tabId) : chrome.action.enable(tabId),
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

async function toggleTab(tab) {
  if (!tab?.id || openingTabs.has(tab.id)) return;
  openingTabs.add(tab.id);
  try {
    if (!/^(https?|file):/u.test(tab.url || "")) throw new Error("当前页面不支持阅读辅助");
    let ready;
    try {
      ready = await chrome.tabs.sendMessage(tab.id, { type: "SUPER_READER_PING" });
    } catch (_) { /* The page may predate extension installation. */ }
    if (!ready) {
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
    // The reader publishes every transition. A command response can already be
    // stale when it arrives, so it must not overwrite a newer published state.
    await chrome.tabs.sendMessage(tab.id, { type: "SUPER_READER_TOGGLE" });
  } catch (error) {
    await updateAction(tab.id, { error: error.message }).catch(() => {});
  } finally {
    openingTabs.delete(tab.id);
  }
}

chrome.action.onClicked.addListener((tab) => { void toggleTab(tab); });
chrome.tabs.onUpdated.addListener((tabId, change) => {
  if (change.status === "loading") void updateAction(tabId).catch(() => {});
});
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (sender.id !== chrome.runtime.id || !sender.tab?.id) return;
  if (message?.type === "SUPER_READER_STATE") {
    void updateAction(sender.tab.id, message).catch(() => {});
    sendResponse({});
  }
  if (message?.type !== "SUPER_READER_PROCESS") return;
  void runInference(message.texts).then(sendResponse, (error) => sendResponse({ error: error.message }));
  return true;
});
