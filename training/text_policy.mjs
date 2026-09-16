import { readFileSync } from "node:fs";

export const DATA_POLICY = Object.freeze(JSON.parse(readFileSync(new URL("./text-policy.json", import.meta.url), "utf8")));
export const PROXY_PUNCTUATION = new Set(DATA_POLICY.proxy_punctuation);
export function normalizeContext(text) {
  // Preserve the distinction between the Chinese ellipsis proxy and ASCII dots.
  return text.split("…").map((part) => part.normalize("NFKC")).join("…");
}
export function validProxyLabel(label) {
  return typeof label === "string" && label.length > 0 && Array.from(label).every((c) => PROXY_PUNCTUATION.has(c));
}
export function requireDataPolicy(summary) {
  if (summary.standard !== DATA_POLICY.standard || summary.tokenization !== "character" ||
      summary.input_representation !== DATA_POLICY.input_representation ||
      !DATA_POLICY.excluded_proxy_punctuation.every((c) => summary.excluded_proxy_punctuation?.includes(c))) {
    throw new Error(`Data requires ${DATA_POLICY.standard}; regenerate original documents in a fresh directory`);
  }
}
