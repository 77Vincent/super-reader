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

  const CLAUSE_END_CHARACTER = /[，,、。！？!?；;：:\n…]/u;
  const OPENING_DOUBLE_QUOTE = /[“]/u;
  const CLOSING_DOUBLE_QUOTE = /[”]/u;
  const TRAILING_CLOSER = /[”’」』）》】〉〕〗〙〛"'）)\]]/u;
  const HAN_CHARACTER = /\p{Script=Han}/u;

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

  function createSegmenter(locale) {
    if (typeof Intl !== "undefined" && typeof Intl.Segmenter === "function") {
      return new Intl.Segmenter(locale || "zh-CN", { granularity: "word" });
    }

    return null;
  }

  function tokenize(text, segmenter) {
    if (!segmenter) return [];

    let fallbackIndex = 0;
    return Array.from(segmenter.segment(text), ({ segment, index, isWordLike }) => {
      const tokenIndex = Number.isInteger(index) ? index : fallbackIndex;
      fallbackIndex = tokenIndex + segment.length;
      return {
        segment,
        index: tokenIndex,
        isWordLike: Boolean(isWordLike),
      };
    });
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

      const isOpeningDoubleQuote = (
        OPENING_DOUBLE_QUOTE.test(character) ||
        (character === '"' && !isInsideStraightDoubleQuote)
      );

      if (isOpeningDoubleQuote) {
        flush();
        buffer = character;
        if (character === '"') isInsideStraightDoubleQuote = true;
        continue;
      }

      buffer += character;

      const isClosingDoubleQuote = (
        CLOSING_DOUBLE_QUOTE.test(character) ||
        (character === '"' && isInsideStraightDoubleQuote)
      );

      if (isClosingDoubleQuote) {
        if (character === '"') isInsideStraightDoubleQuote = false;
        flush();
        continue;
      }

      if (!CLAUSE_END_CHARACTER.test(character)) continue;

      while (
        index + 1 < characters.length &&
        (
          CLAUSE_END_CHARACTER.test(characters[index + 1]) ||
          (
            TRAILING_CLOSER.test(characters[index + 1]) &&
            (
              characters[index + 1] !== '"' ||
              isInsideStraightDoubleQuote
            )
          )
        )
      ) {
        index += 1;
        buffer += characters[index];
        if (characters[index] === '"') isInsideStraightDoubleQuote = false;
      }

      flush();
    }

    flush();
    return clauses;
  }

  function createBoundaryTree(wordTokens, scores) {
    const lengthPrefix = [0];
    for (const token of wordTokens) {
      lengthPrefix.push(lengthPrefix.at(-1) + visualLength(token.segment));
    }

    function createNode(start, end) {
      const node = {
        start,
        end,
        length: lengthPrefix[end] - lengthPrefix[start],
        text: wordTokens.slice(start, end).map((token) => token.segment).join(""),
      };
      if (end - start <= 1) return node;

      let boundary = start;
      for (let gap = start + 1; gap < end - 1; gap += 1) {
        if (scores[gap] > scores[boundary]) boundary = gap;
      }
      node.boundaryAfter = boundary;
      node.boundaryScore = scores[boundary];
      node.left = createNode(start, boundary + 1);
      node.right = createNode(boundary + 1, end);
      return node;
    }

    return createNode(0, wordTokens.length);
  }

  function collectTargetRanges(node, targetLength, ranges) {
    if (!node.left || !node.right || node.length <= targetLength) {
      ranges.push({ start: node.start, end: node.end });
      return;
    }
    collectTargetRanges(node.left, targetLength, ranges);
    collectTargetRanges(node.right, targetLength, ranges);
  }

  function chunkClause(text, targetLength, segmenter) {
    const wordTokens = tokenize(text, segmenter).filter(
      (token) => token.isWordLike && HAN_CHARACTER.test(token.segment),
    );
    if (wordTokens.length < 2) return [text];
    if (!modelBackend || typeof modelBackend.scoreTokens !== "function") {
      throw new Error("Super Reader model backend must load before the chunker");
    }

    const scores = modelBackend.scoreTokens(wordTokens.map((token) => token.segment));
    const tree = createBoundaryTree(wordTokens, scores);
    const ranges = [];
    collectTargetRanges(tree, targetLength, ranges);

    return ranges.map((range, index) => {
      const start = index === 0 ? 0 : wordTokens[range.start].index;
      const end = index === ranges.length - 1
        ? text.length
        : wordTokens[ranges[index + 1].start].index;
      return text.slice(start, end);
    });
  }

  function chunkTextByClause(text, options = {}) {
    if (!text) return [];

    const targetLength = Math.min(12, Math.max(2, Number(options.targetLength) || 7));
    const segmenter = options.segmenter === undefined
      ? createSegmenter(options.locale)
      : options.segmenter;

    return splitClauses(text).map((clause) => (
      chunkClause(clause, targetLength, segmenter)
    ));
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
    createBoundaryTree,
    chunkText,
    chunkTextByClause,
    createSegmenter,
    splitUnderlineRuns,
    splitClauses,
    visualLength,
  });
});
