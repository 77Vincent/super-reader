import { readFileSync } from "node:fs";

export const DATA_POLICY = Object.freeze(JSON.parse(readFileSync(new URL("./text-policy.json", import.meta.url), "utf8")));
export const PROXY_PUNCTUATION = new Set(DATA_POLICY.proxy_punctuation);
const LINE_BOUNDARIES = new RegExp(DATA_POLICY.line_boundaries, "u");
const SURFACE_PATTERNS = Object.entries(DATA_POLICY.sample_filter.fragment_patterns)
  .map(([name, pattern]) => [name, new RegExp(pattern, "u")]);
const WHOLE_FRAGMENT = new RegExp(DATA_POLICY.sample_filter.whole_fragment_pattern, "u");
const EMPTY_TEMPLATE = new RegExp(DATA_POLICY.sample_filter.empty_template_pattern, "gu");
const SIGN = String.raw`[+\-−﹢﹣＋－]`;
const NUMBER = `${SIGN}? *(?:\\p{Nd}+(?:[.．]\\p{Nd}*)?|[.．]\\p{Nd}+)(?:[eEｅＥ]${SIGN}?\\p{Nd}+)?`;
const NUMERIC_COMMAS = new RegExp(`(?<![\\p{Nd}.．])${NUMBER} *(，) *(?=${NUMBER})`, "gu");
const HAN = /^[\u3007\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u{20000}-\u{2fa1f}\u{30000}-\u{323af}]$/u;
const normalizeFragment = (text) => text.normalize("NFKC").replace(/\s+/gu, " ").trim();

export function validBoundaryContext(left, right) {
  return HAN.test(Array.from(left).at(-1) || "") || HAN.test(Array.from(right)[0] || "");
}

export function* trainingFragmentLines(text) {
  for (const rawLine of String(text || "").split(LINE_BOUNDARIES)) {
    const line = rawLine.replace(/\s+/gu, " ").trim();
    const emptySpans = Array.from(line.matchAll(EMPTY_TEMPLATE), (m) => [m.index, m.index + m[0].length]);
    const numericCommas = new Set(line.includes("，") ? Array.from(line.matchAll(NUMERIC_COMMAS),
      (m) => m.index + m[0].indexOf("，")) : []);
    const fragments = [];
    let start = 0;
    const add = (end, punctuation) => {
      const raw = line.slice(start, end), value = normalizeFragment(raw);
      if (!value) {
        if (fragments.length) fragments.at(-1).punctuation += punctuation;
        return;
      }
      const reasons = SURFACE_PATTERNS.filter(([, pattern]) => pattern.test(raw) || pattern.test(value)).map(([name]) => name);
      if (WHOLE_FRAGMENT.test(raw.trim()) || WHOLE_FRAGMENT.test(value)) reasons.push("web_control");
      if (emptySpans.some(([left, right]) => left < end && right > start)) reasons.push("empty_template");
      fragments.push({ text: value, punctuation, reasons });
    };
    const characters = Array.from(line);
    let offset = 0;
    for (let i = 0; i < characters.length; i += 1) {
      const character = characters[i];
      if (PROXY_PUNCTUATION.has(character) && !numericCommas.has(offset)) {
        add(offset, character);
        start = offset + character.length;
      }
      offset += character.length;
    }
    add(line.length, "");
    yield fragments;
  }
}

export function* trainingPairs(text, rejectionCounts) {
  let lineIndex = 0;
  for (const fragments of trainingFragmentLines(text)) {
    for (let index = 0; index < fragments.length-1; index += 1) {
      const left = fragments[index], right = fragments[index+1];
      if (!/[\p{L}\p{N}]/u.test(left.text) || !/[\p{L}\p{N}]/u.test(right.text)) continue;
      const reasons = new Set([...left.reasons, ...right.reasons]);
      if (!validBoundaryContext(left.text, right.text)) reasons.add("both_neighbors_non_han");
      if (reasons.size) {
        if (rejectionCounts) {
          rejectionCounts.pairs = (rejectionCounts.pairs || 0) + 1;
          for (const reason of reasons) rejectionCounts[reason] = (rejectionCounts[reason] || 0) + 1;
        }
        continue;
      }
      yield { left: left.text, right: right.text, punctuation: left.punctuation, lineIndex, index };
    }
    lineIndex += 1;
  }
}
export function validProxyLabel(label) {
  return typeof label === "string" && label.length > 0 && Array.from(label).every((c) => PROXY_PUNCTUATION.has(c));
}
export function requireDataPolicy(summary) {
  const proxies = summary.proxy_punctuation;
  if (summary.standard !== DATA_POLICY.standard || summary.tokenization !== "character" ||
      summary.input_representation !== DATA_POLICY.input_representation ||
      !Array.isArray(proxies) || new Set(proxies).size !== PROXY_PUNCTUATION.size ||
      !proxies.every((c) => PROXY_PUNCTUATION.has(c)) ||
      summary.normalization !== DATA_POLICY.normalization ||
      summary.numeric_punctuation !== DATA_POLICY.numeric_punctuation ||
      summary.boundary_context !== DATA_POLICY.boundary_context ||
      summary.line_boundaries !== DATA_POLICY.line_boundaries ||
      JSON.stringify(summary.sample_filter) !== JSON.stringify(DATA_POLICY.sample_filter)) {
    throw new Error(`Data requires ${DATA_POLICY.standard}; regenerate original documents in a fresh directory`);
  }
}
