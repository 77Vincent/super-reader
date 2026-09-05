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
  const TRAILING_CLOSER = /[”’」』）》】〉〕〗〙〛"'）)\]]/u;
  const SPLIT_LENGTH_THRESHOLD = 10;
  const MODEL_CANDIDATE_MIN_RELATIVE_SCORE = 0.05;
  const MODEL_CANDIDATE_LOGIT_MARGIN = -Math.log(
    MODEL_CANDIDATE_MIN_RELATIVE_SCORE,
  );

  function visualLength(text) {
    return Array.from(text).reduce(
      (length, character) => length + (HAN_CHARACTER.test(character) ? 1 : 0),
      0,
    );
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
      buffer += character;
      if (character === '"') {
        isInsideStraightDoubleQuote = !isInsideStraightDoubleQuote;
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

  function selectBestBoundary(scores, isAllowed = () => true) {
    if (!Array.isArray(scores) || scores.length === 0) return null;

    let bestIndex = null;
    for (let index = 0; index < scores.length; index += 1) {
      if (!Number.isFinite(scores[index]) || !isAllowed(index)) continue;
      if (bestIndex === null || scores[index] > scores[bestIndex]) {
        bestIndex = index;
      }
    }

    return bestIndex;
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

  function selectCenteredModelBoundary(scores, isAllowed = () => true) {
    const bestIndex = selectBestBoundary(scores, isAllowed);
    if (bestIndex === null) return null;

    const minimumCandidateScore =
      scores[bestIndex] - MODEL_CANDIDATE_LOGIT_MARGIN;
    const center = (scores.length + 1) / 2;
    let selectedIndex = null;

    for (let index = 0; index < scores.length; index += 1) {
      if (
        !Number.isFinite(scores[index]) ||
        !isAllowed(index) ||
        scores[index] < minimumCandidateScore
      ) {
        continue;
      }

      if (selectedIndex === null) {
        selectedIndex = index;
        continue;
      }

      const distance = Math.abs(index + 1 - center);
      const selectedDistance = Math.abs(selectedIndex + 1 - center);
      if (
        distance < selectedDistance ||
        (distance === selectedDistance && scores[index] > scores[selectedIndex])
      ) {
        selectedIndex = index;
      }
    }

    return selectedIndex;
  }

  function chunkByModel(text, segmenter) {
    const tokens = tokenizeHanCharacters(text);
    if (tokens.length < 2) return [text];
    if (!modelBackend || typeof modelBackend.scoreTokens !== "function") {
      throw new Error("Super Reader model backend must load before the chunker");
    }
    const ranges = [];

    function visit(start, end) {
      if (end - start <= SPLIT_LENGTH_THRESHOLD) {
        ranges.push({ start, end });
        return;
      }

      const scores = modelBackend.scoreTokens(
        tokens.slice(start, end).map((token) => token.segment),
      );
      const boundaryIndex = selectCenteredModelBoundary(
        scores,
        (candidateIndex) => {
          const boundaryAfter = start + candidateIndex;
          const sourceBoundary = tokens[boundaryAfter + 1].index;
          return !boundaryFallsInsideWord(text, sourceBoundary, segmenter);
        },
      );
      if (boundaryIndex === null) {
        ranges.push({ start, end });
        return;
      }

      const boundaryAfter = start + boundaryIndex;
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
    return chunkTextByClause(text, options).flatMap((clauseChunks) => (
      clauseChunks.map((chunk, index) => {
        const processed = HAN_CHARACTER.test(chunk);
        const previousProcessed = index > 0 && HAN_CHARACTER.test(clauseChunks[index - 1]);
        return {
          text: chunk,
          processed,
          separated: processed && previousProcessed,
        };
      })
    ));
  }

  return Object.freeze({
    buildVisualChunks,
    boundaryFallsInsideWord,
    chunkText,
    chunkTextByClause,
    createSegmenter,
    selectCenteredModelBoundary,
    selectBestBoundary,
    splitClauses,
    tokenizeHanCharacters,
    visualLength,
  });
});
