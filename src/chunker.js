(function exposeChunker(root, factory) {
  const api = factory();

  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }

  root.SuperReaderChunker = api;
})(globalThis, function createChunker() {
  "use strict";

  const CLAUSE_END_CHARACTER = /[，,、。！？!?；;：:\n…]/u;
  const OPENING_DOUBLE_QUOTE = /[“]/u;
  const CLOSING_DOUBLE_QUOTE = /[”]/u;
  const TRAILING_CLOSER = /[”’」』）》】〉〕〗〙〛"'）)\]]/u;
  const COUNTABLE_CHARACTER = /[\p{L}\p{N}]/u;

  function visualLength(text) {
    return Array.from(text).reduce(
      (length, character) => length + (COUNTABLE_CHARACTER.test(character) ? 1 : 0),
      0,
    );
  }

  function splitUnderlineRuns(text) {
    const runs = [];

    for (const character of Array.from(text)) {
      const underlinable = COUNTABLE_CHARACTER.test(character);
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
    if (!segmenter) {
      return Array.from(text, (segment) => ({
        segment,
        isWordLike: COUNTABLE_CHARACTER.test(segment),
      }));
    }

    return Array.from(segmenter.segment(text), ({ segment, isWordLike }) => ({
      segment,
      isWordLike: Boolean(isWordLike),
    }));
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

  function buildStableUnits(tokens) {
    const units = [];

    const isSingleCharacterWord = (token) => (
      token?.isWordLike && visualLength(token.segment) === 1
    );
    const isStableWord = (token) => (
      token?.isWordLike && visualLength(token.segment) >= 2
    );
    const pushUnit = (unitTokens) => {
      units.push({
        text: unitTokens.map((token) => token.segment).join(""),
        length: unitTokens.reduce(
          (length, token) => length + visualLength(token.segment),
          0,
        ),
        wordCount: unitTokens.filter((token) => token.isWordLike).length,
      });
    };

    for (let index = 0; index < tokens.length; index += 1) {
      const token = tokens[index];

      if (!token.isWordLike) {
        if (units.length === 0) {
          pushUnit([token]);
        } else {
          units[units.length - 1].text += token.segment;
        }
        continue;
      }

      if (!isSingleCharacterWord(token)) {
        pushUnit([token]);
        continue;
      }

      let runEnd = index;
      while (
        runEnd < tokens.length &&
        isSingleCharacterWord(tokens[runEnd])
      ) {
        runEnd += 1;
      }

      const runLength = runEnd - index;
      const hasStableWordBefore = isStableWord(tokens[index - 1]);
      const hasStableWordAfter = isStableWord(tokens[runEnd]);
      const shouldMergeRun = (
        runLength >= 2 &&
        runLength < 4 &&
        hasStableWordBefore &&
        hasStableWordAfter
      );

      if (shouldMergeRun) {
        pushUnit(tokens.slice(index, runEnd));
      } else {
        tokens.slice(index, runEnd).forEach((singleToken) => {
          pushUnit([singleToken]);
        });
      }

      index = runEnd - 1;
    }

    return units;
  }

  function chunkClause(text, targetLength, segmenter) {
    const preferredMinimum = Math.max(2, targetLength - 1);
    const units = buildStableUnits(tokenize(text, segmenter));
    const chunks = [];
    let buffer = "";
    let bufferLength = 0;
    let bufferWordCount = 0;

    const flush = () => {
      if (!buffer) return;
      chunks.push({
        text: buffer,
        length: bufferLength,
        wordCount: bufferWordCount,
      });
      buffer = "";
      bufferLength = 0;
      bufferWordCount = 0;
    };

    for (const unit of units) {
      const projectedLength = bufferLength + unit.length;
      const flexibleMaximum = targetLength + Math.ceil(targetLength / 3);

      if (
        unit.wordCount > 0 &&
        bufferLength > 0 &&
        (
          bufferLength >= targetLength ||
          projectedLength > flexibleMaximum ||
          (
            bufferLength >= preferredMinimum &&
            projectedLength > targetLength
          )
        )
      ) {
        flush();
      }

      buffer += unit.text;
      bufferLength += unit.length;
      bufferWordCount += unit.wordCount;
    }

    flush();

    if (chunks.length > 1) {
      const tail = chunks[chunks.length - 1];
      const previous = chunks[chunks.length - 2];
      const minimumTailLength = Math.max(2, Math.floor(targetLength / 2));
      const minimumSingleWordLength = Math.max(2, Math.ceil(targetLength / 2));
      const isOrphan = (
        tail.length < minimumTailLength ||
        (
          tail.wordCount === 1 &&
          tail.length < minimumSingleWordLength
        )
      );
      const mergedLength = previous.length + tail.length;

      if (isOrphan && mergedLength <= targetLength + preferredMinimum) {
        previous.text += tail.text;
        previous.length = mergedLength;
        previous.wordCount += tail.wordCount;
        chunks.pop();
      }
    }

    return chunks.map((chunk) => chunk.text);
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
    return chunkText(text, options).map((chunk, index) => ({
      text: chunk,
      underlined: index % 2 === 0,
    }));
  }

  return Object.freeze({
    buildVisualChunks,
    chunkText,
    chunkTextByClause,
    createSegmenter,
    splitUnderlineRuns,
    splitClauses,
    visualLength,
  });
});
