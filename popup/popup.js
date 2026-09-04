"use strict";

const DEFAULTS = {
  enabled: false,
  dividerWidth: 2,
};

const enabledInput = document.querySelector("#enabled");
const dividerInput = document.querySelector("#divider-width");
const dividerOutput = document.querySelector("#divider-output");
const switchState = document.querySelector("#switch-state");
const pageStatus = document.querySelector("#page-status");
const settingsPanel = document.querySelector(".settings");
const preview = document.querySelector(".preview");

function renderDividerWidth(value) {
  const width = Number(value);
  dividerInput.value = width;
  dividerOutput.textContent = `${width} px`;
  preview.style.setProperty("--super-reader-divider-width", `${width}px`);
}

function render(settings) {
  enabledInput.checked = settings.enabled;
  renderDividerWidth(settings.dividerWidth);
  switchState.textContent = settings.enabled ? "已开启" : "已关闭";
  settingsPanel.setAttribute("aria-disabled", String(!settings.enabled));
}

function isInjectableUrl(url = "") {
  return /^(https?|file):/u.test(url);
}

async function getActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return tab;
}

async function pingTab(tabId) {
  try {
    return await chrome.tabs.sendMessage(tabId, { type: "SUPER_READER_PING" });
  } catch (_error) {
    return null;
  }
}

async function ensureCurrentPageReady() {
  const tab = await getActiveTab();
  if (!tab?.id || !isInjectableUrl(tab.url)) {
    pageStatus.textContent = "当前是 Chrome 内部页面，浏览器不允许扩展修改";
    pageStatus.classList.add("error");
    return false;
  }

  if (await pingTab(tab.id)) return true;

  try {
    await chrome.scripting.insertCSS({
      target: { tabId: tab.id },
      files: ["src/content.css"],
    });
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files: [
        "src/boundary-model-data.js",
        "src/model-backend.js",
        "src/chunker.js",
        "src/content.js",
      ],
    });
    pageStatus.textContent = "已连接当前网页，设置会应用到所有普通网页";
    pageStatus.classList.remove("error");
    return true;
  } catch (_error) {
    pageStatus.textContent = tab.url?.startsWith("file:")
      ? "请先在扩展详情中开启“允许访问文件网址”"
      : "当前网页不允许扩展修改，请刷新页面后重试";
    pageStatus.classList.add("error");
    return false;
  }
}

enabledInput.addEventListener("change", async () => {
  const enabled = enabledInput.checked;
  await chrome.storage.sync.set({ enabled });
  switchState.textContent = enabled ? "已开启" : "已关闭";
  settingsPanel.setAttribute("aria-disabled", String(!enabled));
  await ensureCurrentPageReady();
});

dividerInput.addEventListener("input", () => {
  renderDividerWidth(dividerInput.value);
});

dividerInput.addEventListener("change", async () => {
  await chrome.storage.sync.set({ dividerWidth: Number(dividerInput.value) });
  await ensureCurrentPageReady();
});

chrome.storage.onChanged.addListener(async (_changes, areaName) => {
  if (areaName !== "sync") return;
  const settings = await chrome.storage.sync.get(DEFAULTS);
  render(settings);
});

(async function initializePopup() {
  const settings = await chrome.storage.sync.get(DEFAULTS);
  render(settings);
  await ensureCurrentPageReady();
})();
