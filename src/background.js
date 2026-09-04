"use strict";

const DEFAULTS = {
  enabled: false,
  dividerWidth: 3,
  dividerColor: "red",
};

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
});

chrome.storage.onChanged.addListener((changes, areaName) => {
  if (areaName === "sync" && changes.enabled) {
    updateBadge(Boolean(changes.enabled.newValue));
  }
});

chrome.commands.onCommand.addListener(async (command) => {
  if (command !== "toggle-reader") return;
  const { enabled } = await chrome.storage.sync.get(DEFAULTS);
  await chrome.storage.sync.set({ enabled: !enabled });
});
