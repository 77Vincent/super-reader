(function exposeChunker(root, factory) {
  const modelBackend = typeof module === "object" && module.exports
    ? require("./inference.js")
    : root.SuperReaderModelBackend;
  const api = factory(modelBackend);

  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }

  root.SuperReaderChunker = api;
})(globalThis, function createChunker(modelBackend) {
  "use strict";

  const HAN_CHARACTER = /\p{Script=Han}/u;
  const CLAUSE_END_CHARACTER = /[，,、。.！？!?；;：:\n…（）()《》〈〉]/u;
  const TRAILING_CLOSER = /[”’」』）》】〉〕〗〙〛"'）)\]]/u;
  const NUMERIC_TOKEN = /^\p{Number}+(?:[.,]\p{Number}+)*$/u;
  const NUMERIC_EXPRESSION = /\p{Number}+(?:[.,]\p{Number}+)?(?:\s+\p{Number}+[\/／]\p{Number}+|[\/／]\p{Number}+)?/gu;
  const SINGLE_HAN_WORD = /^\p{Script=Han}$/u;
  const WHITESPACE = /^\s+$/u;
  const SPLIT_LENGTH_THRESHOLD = 12;
  const MAX_MODEL_WINDOW_TOKENS = 256;
  const USES_CONTEXT = modelBackend?.getModelInfo?.().inputRepresentation === "unicode-context-v1";
  // Keep in sync with training/text-policy.json; whitespace is context, not a proxy.
  const CONTEXT_PROXY = /[，,。.！？!?；;…]/u;

  function isContextProxy(characters, index) {
    const c = characters[index];
    return CONTEXT_PROXY.test(c || "") && !(
      /[.,，．]/u.test(c) && /^\p{Nd}$/u.test(characters[index - 1] || "") &&
      /^\p{Nd}$/u.test(characters[index + 1] || "")
    );
  }

  // Keep source UTF-16 offsets even when NFKC expands a grapheme (e.g. ﬃ).
  function normalizeContextTokens(text) {
    const graphemes = typeof Intl?.Segmenter === "function"
      ? Array.from(new Intl.Segmenter("und", { granularity: "grapheme" }).segment(text))
      : Array.from(text).reduce((items, segment) => {
        const previous = items[items.length - 1];
        items.push({ segment, index: previous ? previous.index + previous.segment.length : 0 });
        return items;
      }, []);
    return graphemes.flatMap(({ segment, index }) => (
      Array.from(segment.normalize("NFKC"), (c) => ({ segment: /\s/u.test(c) ? " " : c, index }))
    ));
  }

  function tokenizeContext(text) {
    const expanded = normalizeContextTokens(text);
    const characters = expanded.map((token) => token.segment);
    const tokens = [];
    for (let i = 0; i < expanded.length; i += 1) {
      const token = expanded[i];
      if (isContextProxy(characters, i)) continue;
      if (token.segment === " " && (!tokens.length || tokens[tokens.length - 1].segment === " ")) continue;
      tokens.push(token);
    }
    if (tokens[tokens.length - 1]?.segment === " ") tokens.pop();
    return tokens;
  }

  function visualLength(text) {
    const hanLength = Array.from(text).reduce(
      (length, character) => length + (HAN_CHARACTER.test(character) ? 1 : 0),
      0,
    );
    const numericExpressionLength = Array.from(text.matchAll(NUMERIC_EXPRESSION)).length;
    const latinWords = USES_CONTEXT ? Array.from(text.matchAll(/[A-Za-zＡ-Ｚａ-ｚ]+/gu)).length : 0;
    return hanLength + numericExpressionLength + latinWords;
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
    const normalized = USES_CONTEXT ? normalizeContextTokens(text) : [];
    const normalizedCharacters = normalized.map((token) => token.segment);
    const boundaryOffsets = new Set(normalized.filter((token, index) => (
      isContextProxy(normalizedCharacters, index) || token.segment === "、"
    )).map((token) => token.index));
    let sourceOffset = 0;
    const sourceOffsets = characters.map((character) => {
      const start = sourceOffset;
      sourceOffset += character.length;
      return start;
    });
    const isEnd = (index) => USES_CONTEXT
      ? boundaryOffsets.has(sourceOffsets[index])
      : CLAUSE_END_CHARACTER.test(characters[index]);

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

      if (!isEnd(index)) continue;

      while (index + 1 < characters.length) {
        const nextCharacter = characters[index + 1];
        const isTrailingPunctuation = isEnd(index + 1);
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

  function boundaryFallsInsideWord(text, boundary, segmenter) {
    return normalizedSegments(text, segmenter).some(
      (item) => item.isWordLike && boundary > item.start && boundary < item.end,
    );
  }

  function normalizedSegments(text, segmenter) {
    if (!segmenter) return [];

    let fallbackIndex = 0;
    return Array.from(segmenter.segment(text), (item) => {
      const start = Number.isInteger(item.index) ? item.index : fallbackIndex;
      const end = start + item.segment.length;
      fallbackIndex = end;
      return { ...item, start, end };
    });
  }

  function nextWordLikeSegment(segments, startIndex) {
    for (let index = startIndex; index < segments.length; index += 1) {
      const item = segments[index];
      if (item.isWordLike) return { item, index };
      if (!WHITESPACE.test(item.segment)) return null;
    }
    return null;
  }

  function quantityPhraseRangesFromSegments(segments) {
    const ranges = [];

    for (let index = 0; index < segments.length; index += 1) {
      const number = segments[index];
      if (!number.isWordLike || !NUMERIC_TOKEN.test(number.segment)) continue;

      const classifier = nextWordLikeSegment(segments, index + 1);
      if (!classifier || !SINGLE_HAN_WORD.test(classifier.item.segment)) continue;

      const noun = nextWordLikeSegment(segments, classifier.index + 1);
      if (!noun || !HAN_CHARACTER.test(noun.item.segment)) continue;

      ranges.push({ start: number.start, end: noun.item.end });
    }

    return ranges;
  }

  function quantityPhraseRanges(text, segmenter) {
    return quantityPhraseRangesFromSegments(normalizedSegments(text, segmenter));
  }

  function numericAttachmentRanges(text, segments) {
    const ranges = [];

    for (const match of text.matchAll(NUMERIC_EXPRESSION)) {
      const start = match.index;
      const expressionEnd = start + match[0].length;
      const attachment = segments.find((item) => (
        item.start >= expressionEnd &&
        item.isWordLike &&
        HAN_CHARACTER.test(item.segment) &&
        (
          item.start === expressionEnd ||
          WHITESPACE.test(text.slice(expressionEnd, item.start))
        )
      ));
      if (!attachment) continue;
      ranges.push({ start, end: attachment.end });
    }

    return ranges;
  }

  function boundaryFallsInsideQuantityPhrase(text, boundary, segmenter) {
    return quantityPhraseRanges(text, segmenter).some(
      (range) => boundary > range.start && boundary < range.end,
    );
  }

  // Cumulative visual lengths at token boundaries, including numeric expressions.
  function visualOffsetsForTokens(text, tokens) {
    if (USES_CONTEXT) {
      const ends = [];
      let index = 0;
      for (const c of text) {
        if (HAN_CHARACTER.test(c)) ends.push(index + c.length);
        index += c.length;
      }
      for (const pattern of [NUMERIC_EXPRESSION, /[A-Za-zＡ-Ｚａ-ｚ]+/gu]) {
        for (const match of text.matchAll(pattern)) ends.push(match.index + match[0].length);
      }
      ends.sort((a, b) => a - b);
      let units = 0;
      const result = tokens.map((token) => {
        while (units < ends.length && ends[units] <= token.index) units += 1;
        return units;
      });
      result[0] = 0;
      result.push(ends.length);
      return result;
    }
    const numbers = Array.from(text.matchAll(NUMERIC_EXPRESSION), (match) => match.index);
    let numberCount = 0;
    const offsets = tokens.map((token, index) => {
      while (numberCount < numbers.length && numbers[numberCount] < token.index) numberCount += 1;
      return index + numberCount;
    });
    offsets[0] = 0; // The first fragment also includes any leading numbers.
    offsets.push(tokens.length + numbers.length);
    return offsets;
  }

  function selectBestBoundary(scores, isAllowed = () => true, start = 0, end, visualOffsets) {
    if (!Array.isArray(scores) || scores.length === 0) return null;
    end ??= scores.length;
    const sourceStart = visualOffsets?.[start] ?? start;
    const totalLength = (visualOffsets?.[end + 1] ?? end + 1) - sourceStart;

    let bestIndex = null;
    let bestScore = -Infinity;
    for (let index = start; index < end; index += 1) {
      if (!Number.isFinite(scores[index]) || !isAllowed(index)) continue;
      const leftLength = (visualOffsets?.[index + 1] ?? index + 1) - sourceStart;
      const rightLength = totalLength - leftLength;
      const balance = (leftLength / totalLength) * (rightLength / totalLength); // p * (1 - p)
      // Same ranking as softmax(scores)[index] * balance, without normalization.
      // Equal weighted scores keep the first allowed gap.
      const score = scores[index] + Math.log(balance);
      if (score > bestScore) {
        bestIndex = index;
        bestScore = score;
      }
    }

    return bestIndex;
  }

  function chunkByModel(text, segmenter) {
    const tokens = USES_CONTEXT ? tokenizeContext(text) : tokenizeHanCharacters(text);
    if (tokens.length < 2 || visualLength(text) <= SPLIT_LENGTH_THRESHOLD) return [text];
    if (!modelBackend || typeof modelBackend.scoreTokens !== "function") {
      throw new Error("Super Reader model backend must load before the chunker");
    }
    const visualOffsets = visualOffsetsForTokens(text, tokens);
    const ranges = [];
    const segments = normalizedSegments(text, segmenter);
    const protectedBoundaryRanges = segments
      .filter((item) => item.isWordLike)
      .map((item) => ({ start: item.start, end: item.end }));
    protectedBoundaryRanges.push(...quantityPhraseRangesFromSegments(segments));
    protectedBoundaryRanges.push(...numericAttachmentRanges(text, segments));
    if (USES_CONTEXT) {
      for (const match of text.matchAll(/\p{Nd}+(?:[:：.,，．/／]\p{Nd}+)+/gu)) {
        protectedBoundaryRanges.push({ start: match.index, end: match.index + match[0].length });
      }
    }
    const protectedBoundaryOffsets = new Set(
      tokens.slice(1)
        .map((token) => token.index)
        .filter((boundary) => protectedBoundaryRanges.some(
          (range) => boundary > range.start && boundary < range.end,
        )),
    );
    if (USES_CONTEXT) {
      // Context punctuation has its own tokens. Keep opening marks with what
      // follows and closing marks with what precedes, including straight quotes.
      const opening = new Set(Array.from("“‘「『（(《〈【〔〖〘〚[{"));
      const closing = new Set(Array.from("”’」』）)》〉】〕〗〙〛]}"));
      const insideQuote = new Set();
      tokens.forEach((token, index) => {
        const mark = token.segment;
        const straight = mark === '"' || mark === "'";
        const isClosing = closing.has(mark) || (straight && insideQuote.has(mark));
        const isOpening = opening.has(mark) || (straight && !insideQuote.has(mark));
        if (isClosing) protectedBoundaryOffsets.add(token.index);
        if (isOpening) {
          for (let next = index + 1; next < tokens.length; next += 1) {
            protectedBoundaryOffsets.add(tokens[next].index);
            if (tokens[next].segment !== " ") break;
          }
        }
        if (straight) {
          if (insideQuote.has(mark)) insideQuote.delete(mark);
          else insideQuote.add(mark);
        }
      });
    }

    // Score each gap once, before choosing any cuts. Adjacent windows share one
    // token so the gap between windows is scored too; window edges do not force cuts.
    const scores = [];
    for (let start = 0; start < tokens.length - 1; start += MAX_MODEL_WINDOW_TOKENS - 1) {
      scores.push(...modelBackend.scoreTokens(
        tokens.slice(start, start + MAX_MODEL_WINDOW_TOKENS).map((token) => token.segment),
      ));
    }

    // Use a stack so repeated choices near one end cannot overflow the call stack.
    const pendingRanges = [{ start: 0, end: tokens.length }];
    while (pendingRanges.length > 0) {
      const { start, end } = pendingRanges.pop();
      if (
        end - start < 2 ||
        visualOffsets[end] - visualOffsets[start] <= SPLIT_LENGTH_THRESHOLD
      ) {
        ranges.push({ start, end });
        continue;
      }

      const boundaryAfter = selectBestBoundary(
        scores,
        (index) => !protectedBoundaryOffsets.has(tokens[index + 1].index) && (!USES_CONTEXT || (
          tokens[index + 1].index > tokens[index].index && tokens[index + 1].segment !== " "
        )),
        start,
        end - 1,
        visualOffsets,
      );
      if (boundaryAfter === null) {
        ranges.push({ start, end });
        continue;
      }

      // Visit the left fragment first to preserve source order in the output.
      pendingRanges.push({ start: boundaryAfter + 1, end }, { start, end: boundaryAfter + 1 });
    }

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

  /**
   * Process all texts from one viewport in order. Offsets are ascending UTF-16
   * positions inside each corresponding input; no DOM or task scheduling here.
   * @param {string[]} texts
   * @returns {number[][]}
   */
  function process(texts) {
    const segmenter = createSegmenter("zh-CN");
    return texts.map((text) => {
      const offsets = [];
      let offset = 0;
      for (const chunk of buildVisualChunks(text, { segmenter })) {
        if (chunk.separated) offsets.push(offset);
        offset += chunk.text.length;
      }
      return offsets;
    });
  }

  return Object.freeze({
    process,
    boundaryFallsInsideQuantityPhrase,
    boundaryFallsInsideWord,
    buildVisualChunks,
    chunkText,
    chunkTextByClause,
    createSegmenter,
    selectBestBoundary,
    splitClauses,
    tokenizeHanCharacters,
    tokenizeContext,
    visualLength,
  });
});
