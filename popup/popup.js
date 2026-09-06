"use strict";

const DEFAULTS = {
  enabled: false,
  dividerWidth: 3,
  dividerColor: "red",
};
const PREVIEW_TEXT = "将长句切分成短句加速阅读理解";

const DIVIDER_COLORS = Object.freeze({
  red: "#ff1744",
  yellow: "#ffd600",
  blue: "#2979ff",
  green: "#00e676",
  text: "currentColor",
});
const COLORED_DIVIDER_FILTER = "drop-shadow(0.5px 0 0 rgba(255, 255, 255, 0.7))";

const enabledInput = document.querySelector("#enabled");
const dividerInput = document.querySelector("#divider-width");
const dividerOutput = document.querySelector("#divider-output");
const dividerColorInputs = Array.from(document.querySelectorAll("[name='divider-color']"));
const pageStatus = document.querySelector("#page-status");
const preview = document.querySelector(".preview");

async function renderModelPreview() {
  preview.setAttribute("aria-busy", "true");
  try {
    const response = await chrome.runtime.sendMessage({
      type: "SUPER_READER_SPLIT_TEXTS",
      texts: [PREVIEW_TEXT],
      releaseWhenDisabled: true,
    });
    if (response?.error) throw new Error(response.error);
    const offsets = Array.isArray(response?.offsetsByText?.[0])
      ? response.offsetsByText[0]
      : [];

    preview.replaceChildren();
    let start = 0;
    [...offsets, PREVIEW_TEXT.length].forEach((end) => {
      const span = document.createElement("span");
      if (start > 0) span.className = "preview-chunk--separated";
      span.textContent = PREVIEW_TEXT.slice(start, end);
      preview.append(span);
      start = end;
    });
  } catch (error) {
    console.error("Super Reader failed to render its model preview", error);
  } finally {
    preview.removeAttribute("aria-busy");
  }
}

function renderDividerWidth(value) {
  const numericWidth = Number(value);
  const width = Number.isFinite(numericWidth)
    ? Math.min(5, Math.max(1, Math.round(numericWidth)))
    : DEFAULTS.dividerWidth;
  dividerInput.value = width;
  dividerOutput.textContent = `${width} px`;
  dividerInput.setAttribute("aria-valuetext", `${width} 像素`);
  preview.style.setProperty("--super-reader-divider-width", `${width}px`);
}

function renderDividerColor(value) {
  const colorName = Object.hasOwn(DIVIDER_COLORS, value) ? value : DEFAULTS.dividerColor;
  dividerColorInputs.forEach((input) => {
    input.checked = input.value === colorName;
  });
  preview.style.setProperty("--super-reader-divider-color", DIVIDER_COLORS[colorName]);
  preview.style.setProperty(
    "--super-reader-divider-filter",
    colorName === "text" ? "none" : COLORED_DIVIDER_FILTER,
  );
}

function render(settings) {
  enabledInput.checked = settings.enabled;
  renderDividerWidth(settings.dividerWidth);
  renderDividerColor(settings.dividerColor);
}

function showPageError(message = "") {
  pageStatus.textContent = message;
  pageStatus.hidden = !message;
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
    showPageError("当前页面不支持视觉分块");
    return false;
  }

  if (await pingTab(tab.id)) {
    showPageError();
    return true;
  }

  try {
    await chrome.scripting.insertCSS({
      target: { tabId: tab.id },
      files: ["src/content.css"],
    });
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files: ["src/content.js"],
    });
    showPageError();
    return true;
  } catch (_error) {
    showPageError(tab.url?.startsWith("file:")
      ? "请先在扩展详情中开启“允许访问文件网址”"
      : "当前页面不支持视觉分块");
    return false;
  }
}

enabledInput.addEventListener("change", async () => {
  const enabled = enabledInput.checked;
  await chrome.storage.sync.set({ enabled });
  if (enabled) {
    await ensureCurrentPageReady();
  } else {
    showPageError();
  }
});

dividerInput.addEventListener("input", () => {
  renderDividerWidth(dividerInput.value);
});

dividerInput.addEventListener("change", async () => {
  await chrome.storage.sync.set({ dividerWidth: Number(dividerInput.value) });
  await ensureCurrentPageReady();
});

dividerColorInputs.forEach((input) => {
  input.addEventListener("change", async () => {
    if (!input.checked) return;
    renderDividerColor(input.value);
    await chrome.storage.sync.set({ dividerColor: input.value });
    await ensureCurrentPageReady();
  });
});

chrome.storage.onChanged.addListener(async (_changes, areaName) => {
  if (areaName !== "sync") return;
  const settings = await chrome.storage.sync.get(DEFAULTS);
  render(settings);
});

(async function initializePopup() {
  const previewPromise = renderModelPreview();
  const settings = await chrome.storage.sync.get(DEFAULTS);
  render(settings);
  if (settings.enabled) await ensureCurrentPageReady();
  await previewPromise;
})();
