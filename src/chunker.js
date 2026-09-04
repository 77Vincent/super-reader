(function exposeChunker(root, factory) {
  const modelBackend = typeof module === "object" && module.exports
    ? require("./model-backend.js")
    : root.SuperReaderModelBackend;
  const api = factory(modelBackend);

  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }

  root.SuperReaderChunker = api;
})(globalThis, function createChunker(modelBackend) {
  "use strict";

  const HAN_CHARACTER = /\p{Script=Han}/u;
  const CLAUSE_END_CHARACTER = /[，,、。.！？!?；;：:\n…]/u;
  const OPENING_DOUBLE_QUOTE = /[“]/u;
  const CLOSING_DOUBLE_QUOTE = /[”]/u;
  const TRAILING_CLOSER = /[”’」』）》】〉〕〗〙〛"'）)\]]/u;
  const CONFIDENCE_THRESHOLD = 0.6;
  const CONFIDENCE_EPSILON = 1e-12;

  function visualLength(text) {
    return Array.from(text).reduce(
      (length, character) => length + (HAN_CHARACTER.test(character) ? 1 : 0),
      0,
    );
  }

  function splitUnderlineRuns(text) {
    const runs = [];

    for (const character of Array.from(text)) {
      const underlinable = HAN_CHARACTER.test(character);
      const previous = runs[runs.length - 1];

      if (previous?.underlinable === underlinable) {
        previous.text += character;
      } else {
        runs.push({ text: character, underlinable });
      }
    }

    return runs;
  }

  function tokenizeHanCharacters(text) {
    const tokens = [];
    let index = 0;

    for (const character of text) {
      if (HAN_CHARACTER.test(character)) {
        tokens.push({ segment: character, index });
      }
      index += character.length;
    }

    return tokens;
  }

  function createSegmenter(locale) {
    if (typeof Intl !== "undefined" && typeof Intl.Segmenter === "function") {
      return new Intl.Segmenter(locale || "zh-CN", { granularity: "word" });
    }
    return null;
  }

  function splitClauses(text) {
    if (!text) return [];

    const characters = Array.from(text);
    const clauses = [];
    let buffer = "";
    let isInsideStraightDoubleQuote = false;

    const flush = () => {
      if (!buffer) return;
      clauses.push(buffer);
      buffer = "";
    };

    for (let index = 0; index < characters.length; index += 1) {
      const character = characters[index];
      const isOpeningDoubleQuote =
        OPENING_DOUBLE_QUOTE.test(character) ||
        (character === '"' && !isInsideStraightDoubleQuote);

      if (isOpeningDoubleQuote) {
        flush();
        buffer = character;
        if (character === '"') isInsideStraightDoubleQuote = true;
        continue;
      }

      buffer += character;

      const isClosingDoubleQuote =
        CLOSING_DOUBLE_QUOTE.test(character) ||
        (character === '"' && isInsideStraightDoubleQuote);

      if (isClosingDoubleQuote) {
        if (character === '"') isInsideStraightDoubleQuote = false;
        flush();
        continue;
      }

      if (!CLAUSE_END_CHARACTER.test(character)) continue;

      while (index + 1 < characters.length) {
        const nextCharacter = characters[index + 1];
        const isTrailingPunctuation = CLAUSE_END_CHARACTER.test(nextCharacter);
        const isTrailingCloser =
          TRAILING_CLOSER.test(nextCharacter) &&
          (nextCharacter !== '"' || isInsideStraightDoubleQuote);

        if (!isTrailingPunctuation && !isTrailingCloser) break;

        index += 1;
        buffer += nextCharacter;
        if (nextCharacter === '"') isInsideStraightDoubleQuote = false;
      }

      flush();
    }

    flush();
    return clauses;
  }

  function selectConfidentBoundary(scores) {
    if (!Array.isArray(scores) || scores.length === 0) return null;

    let bestIndex = 0;
    for (let index = 1; index < scores.length; index += 1) {
      if (scores[index] > scores[bestIndex]) bestIndex = index;
    }

    const maximum = scores[bestIndex];
    if (!Number.isFinite(maximum)) return null;
    let denominator = 0;
    for (const score of scores) {
      if (!Number.isFinite(score)) return null;
      denominator += Math.exp(score - maximum);
    }
    const confidence = 1 / denominator;

    return confidence > CONFIDENCE_THRESHOLD + CONFIDENCE_EPSILON
      ? { index: bestIndex, confidence }
      : null;
  }

  function boundaryFallsInsideWord(text, boundary, segmenter) {
    if (!segmenter) return false;

    let fallbackIndex = 0;
    for (const item of segmenter.segment(text)) {
      const start = Number.isInteger(item.index) ? item.index : fallbackIndex;
      const end = start + item.segment.length;
      fallbackIndex = end;
      if (item.isWordLike && boundary > start && boundary < end) return true;
    }
    return false;
  }

  function chunkByModel(text, segmenter) {
    const tokens = tokenizeHanCharacters(text);
    if (tokens.length < 2) return [text];
    if (!modelBackend || typeof modelBackend.scoreTokens !== "function") {
      throw new Error("Super Reader model backend must load before the chunker");
    }

    const ranges = [];

    function visit(start, end) {
      if (end - start < 2) {
        ranges.push({ start, end });
        return;
      }

      const scores = modelBackend.scoreTokens(
        tokens.slice(start, end).map((token) => token.segment),
      );
      const prediction = selectConfidentBoundary(scores);
      if (!prediction) {
        ranges.push({ start, end });
        return;
      }

      const boundaryAfter = start + prediction.index;
      const sourceBoundary = tokens[boundaryAfter + 1].index;
      if (boundaryFallsInsideWord(text, sourceBoundary, segmenter)) {
        ranges.push({ start, end });
        return;
      }

      visit(start, boundaryAfter + 1);
      visit(boundaryAfter + 1, end);
    }

    visit(0, tokens.length);

    return ranges.map((range, index) => {
      const start = index === 0 ? 0 : tokens[range.start].index;
      const end = index === ranges.length - 1
        ? text.length
        : tokens[ranges[index + 1].start].index;
      return text.slice(start, end);
    });
  }

  function chunkTextByClause(text, options = {}) {
    if (!text) return [];
    const segmenter = options.segmenter === undefined
      ? createSegmenter(options.locale)
      : options.segmenter;
    return splitClauses(text).map((clause) => chunkByModel(clause, segmenter));
  }

  function chunkText(text, options = {}) {
    return chunkTextByClause(text, options).flat();
  }

  function buildVisualChunks(text, options = {}) {
    let visualIndex = 0;
    return chunkText(text, options).map((chunk) => {
      const processed = HAN_CHARACTER.test(chunk);
      const underlined = processed && visualIndex % 2 === 0;
      if (processed) visualIndex += 1;
      return { text: chunk, underlined, processed };
    });
  }

  return Object.freeze({
    buildVisualChunks,
    boundaryFallsInsideWord,
    chunkText,
    chunkTextByClause,
    createSegmenter,
    selectConfidentBoundary,
    splitClauses,
    tokenizeHanCharacters,
    splitUnderlineRuns,
    visualLength,
  });
});
