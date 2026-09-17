import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";

export const DATA_POLICY = Object.freeze(JSON.parse(readFileSync(new URL("./text-policy.json", import.meta.url), "utf8")));
export const PROXY_PUNCTUATION = new Set(DATA_POLICY.proxy_punctuation);
const LINE_BOUNDARIES = new RegExp(DATA_POLICY.line_boundaries, "u");
const SURFACE_PATTERNS = Object.entries(DATA_POLICY.sample_filter.fragment_patterns)
  .map(([name, pattern]) => [name, new RegExp(pattern, "u")]);
const WHOLE_FRAGMENT = new RegExp(DATA_POLICY.sample_filter.whole_fragment_pattern, "u");
const EMPTY_TEMPLATE = new RegExp(DATA_POLICY.sample_filter.empty_template_pattern, "gu");
export const DISCARD_BARRIER = DATA_POLICY.source_filter.barrier;
const SOURCE_PATTERNS = Object.entries(DATA_POLICY.source_filter.span_patterns)
  .map(([name, pattern]) => [new RegExp(pattern, name.startsWith("html_") ? "giu" : "gu"),
    DATA_POLICY.source_filter.span_exemptions[name] ? new RegExp(DATA_POLICY.source_filter.span_exemptions[name], "u") : null]);
const symbolBytes = readFileSync(new URL(DATA_POLICY.symbol_window.symbols_file, import.meta.url));
if (createHash("sha256").update(symbolBytes).digest("hex") !== DATA_POLICY.symbol_window.symbols_sha256) {
  throw new Error("Symbol table does not match data policy");
}
const SYMBOLS = new Set(JSON.parse(symbolBytes).ranges.flatMap(([a, b]) =>
  Array.from({ length: b - a + 1 }, (_, i) => String.fromCodePoint(a + i))));
for (const character of PROXY_PUNCTUATION) SYMBOLS.delete(character);
const SIGN = String.raw`[+\-−﹢﹣＋－]`;
const NUMBER = `${SIGN}? *(?:\\p{Nd}+(?:[.．]\\p{Nd}*)?|[.．]\\p{Nd}+)(?:[eEｅＥ]${SIGN}?\\p{Nd}+)?`;
const NUMERIC_COMMAS = new RegExp(`(?<![\\p{Nd}.．])${NUMBER} *(，) *(?=${NUMBER})`, "gu");
const HAN = /[\u3007\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u{20000}-\u{2fa1f}\u{30000}-\u{323af}]/u;
const normalizeFragment = (text) => text.normalize("NFKC").replace(/\s+/gu, " ").trim();

export function validBoundaryContext(left, right) {
  return HAN.test(Array.from(left).at(-1) || "") || HAN.test(Array.from(right)[0] || "");
}

export function discardedSpan(text) {
  const breaks = String(text).match(new RegExp(DATA_POLICY.line_boundaries, "gu")) || [];
  return DISCARD_BARRIER + breaks.map((newline) => newline + DISCARD_BARRIER).join("");
}

export function discardBalanced(text, opening, closing) {
  const pieces = [];
  let start = 0, depth = 0, i = 0;
  while (i < text.length) {
    if (text.startsWith(opening, i)) {
      if (depth === 0) { pieces.push(text.slice(start, i)); start = i; }
      depth += 1; i += opening.length;
    } else if (depth && text.startsWith(closing, i)) {
      depth -= 1; i += closing.length;
      if (depth === 0) { pieces.push(discardedSpan(text.slice(start, i))); start = i; }
    } else i += 1;
  }
  pieces.push(depth ? discardedSpan(text.slice(start)) : text.slice(start));
  return pieces.join("");
}

export function prepareSourceText(text) {
  let prepared = String(text || "");
  for (const [pattern, exemption] of SOURCE_PATTERNS) prepared = prepared.replace(pattern,
    (match) => exemption?.test(match) ? match : discardedSpan(match));
  for (const [opening, closing] of DATA_POLICY.source_filter.balanced_spans) {
    if (prepared.includes(opening)) prepared = discardBalanced(prepared, opening, closing);
  }
  return prepared;
}

export function symbolEnclosures(line, maximum = DATA_POLICY.symbol_window.max_distance) {
  const characters = Array.from(line), result = new Map();
  let left = null;
  for (let right = 0; right < characters.length; right += 1) {
    if (characters[right] === DISCARD_BARRIER) left = null;
    else if (SYMBOLS.has(characters[right])) {
      if (left !== null && right - left <= maximum) {
        for (let index = left + 1; index < right; index += 1) {
          if (PROXY_PUNCTUATION.has(characters[index])) result.set(index, right - left);
        }
      }
      left = right;
    }
  }
  return result;
}

export function* trainingFragmentLines(text, { symbolWindow = DATA_POLICY.symbol_window.max_distance } = {}) {
  for (const rawLine of prepareSourceText(text).split(LINE_BOUNDARIES)) {
    // Preserve code-point distances until after the window check.
    const line = rawLine.replace(/\s/gu, " ");
    const emptySpans = Array.from(line.matchAll(EMPTY_TEMPLATE), (m) => [m.index, m.index + m[0].length]);
    const numericCommas = new Set(line.includes("，") ? Array.from(line.matchAll(NUMERIC_COMMAS),
      (m) => m.index + m[0].indexOf("，")) : []);
    const surrounded = symbolEnclosures(line, symbolWindow);
    const fragments = [];
    let start = 0;
    const add = (end, punctuation, boundaryReasons = []) => {
      const raw = line.slice(start, end), value = normalizeFragment(raw);
      // In particular JS treats U+FEFF as whitespace: inspect it before folding.
      const original = rawLine.slice(start, end);
      const reasons = SURFACE_PATTERNS.filter(([, pattern]) => pattern.test(original) || pattern.test(value)).map(([name]) => name);
      if (!value && !reasons.length) {
        if (fragments.length) {
          fragments.at(-1).punctuation += punctuation;
          fragments.at(-1).boundary_reasons.push(...boundaryReasons);
        }
        return;
      }
      if (DATA_POLICY.sample_filter.require_han && !HAN.test(value)) reasons.push("fragment_without_han");
      if (WHOLE_FRAGMENT.test(raw.trim()) || WHOLE_FRAGMENT.test(value)) reasons.push("web_control");
      if (emptySpans.some(([left, right]) => left < end && right > start)) reasons.push("empty_template");
      fragments.push({ text: value, punctuation, reasons, boundary_reasons: boundaryReasons });
    };
    const characters = Array.from(line);
    let offset = 0;
    for (let i = 0; i < characters.length; i += 1) {
      const character = characters[i];
      if (PROXY_PUNCTUATION.has(character) && !numericCommas.has(offset)) {
        // A rejected target remains a delimiter, never fusing its two sides.
        add(offset, character, surrounded.has(i) ? ["symbol_window"] : []);
        start = offset + character.length;
      }
      offset += character.length;
    }
    add(line.length, "");
    if (fragments.length) {
      const edges = DATA_POLICY.sample_filter.line_edges;
      if (edges.discard_first_fragment) fragments[0].reasons.push("line_first_fragment");
      if (edges.discard_unterminated_last_fragment && !fragments.at(-1).punctuation) {
        fragments.at(-1).reasons.push("line_unterminated_tail");
      }
    }
    yield fragments;
  }
}

export function* trainingPairs(text, rejectionCounts, options) {
  let lineIndex = 0;
  for (const fragments of trainingFragmentLines(text, options)) {
    for (let index = 0; index < fragments.length-1; index += 1) {
      const left = fragments[index], right = fragments[index+1];
      if (!/[\p{L}\p{N}]/u.test(left.text) || !/[\p{L}\p{N}]/u.test(right.text)) continue;
      const reasons = new Set([...left.reasons, ...right.reasons, ...left.boundary_reasons]);
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
      JSON.stringify(summary.source_filter) !== JSON.stringify(DATA_POLICY.source_filter) ||
      JSON.stringify(summary.symbol_window) !== JSON.stringify(DATA_POLICY.symbol_window) ||
      JSON.stringify(summary.sample_filter) !== JSON.stringify(DATA_POLICY.sample_filter)) {
    throw new Error(`Data requires ${DATA_POLICY.standard}; regenerate original documents in a fresh directory`);
  }
}
