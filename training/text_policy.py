"""Shared, versioned input and supervision contract (no ML dependencies)."""
import json
import re
import unicodedata
import hashlib
from pathlib import Path

DATA_POLICY = json.loads(Path(__file__).with_name("text-policy.json").read_text())
PROXY_PUNCTUATION = frozenset(DATA_POLICY["proxy_punctuation"])
LINE_BOUNDARIES = re.compile(DATA_POLICY['line_boundaries'])
SURFACE_PATTERNS = {key: re.compile(value) for key, value in DATA_POLICY['sample_filter']['fragment_patterns'].items()}
WHOLE_FRAGMENT = re.compile(DATA_POLICY['sample_filter']['whole_fragment_pattern'])
EMPTY_TEMPLATE = re.compile(DATA_POLICY['sample_filter']['empty_template_pattern'])
BARRIER = DATA_POLICY['source_filter']['barrier']
SOURCE_PATTERNS = [(re.compile(pattern, re.IGNORECASE if name.startswith('html_') else 0),
                    re.compile(DATA_POLICY['source_filter']['span_exemptions'][name])
                    if name in DATA_POLICY['source_filter']['span_exemptions'] else None)
                   for name, pattern in DATA_POLICY['source_filter']['span_patterns'].items()]
_symbol_bytes = Path(__file__).with_name(DATA_POLICY['symbol_window']['symbols_file']).read_bytes()
if hashlib.sha256(_symbol_bytes).hexdigest() != DATA_POLICY['symbol_window']['symbols_sha256']:
    raise ValueError('Symbol table does not match data policy')
SYMBOLS = frozenset(chr(i) for a, b in json.loads(_symbol_bytes)['ranges'] for i in range(a, b+1)) - PROXY_PUNCTUATION
SIGN = r'[+\-−﹢﹣＋－]'
NUMBER = SIGN + r'? *(?:\d+(?:[.．]\d*)?|[.．]\d+)(?:[eEｅＥ]' + SIGN + r'?\d+)?'
# Consume only the left value and separator, so 1，-2，+3 protects both commas.
# Do not restart at each digit inside a long numeric run if no separator follows.
NUMERIC_COMMAS = re.compile(r'(?<![\d.．])' + NUMBER + r' *(，) *(?=' + NUMBER + r')')
HAN = re.compile('[\u3007\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f\U00030000-\U000323af]')


def normalized_fragment(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip()


def discarded_span(text):
    """A hole is not whitespace; poison both exposed ends on every source line."""
    return BARRIER + ''.join(m.group() + BARRIER for m in LINE_BOUNDARIES.finditer(text))


def discard_balanced(text, opening, closing):
    pieces = []; start = 0; depth = 0; i = 0
    while i < len(text):
        if text.startswith(opening, i):
            if depth == 0:
                pieces.append(text[start:i]); start = i
            depth += 1; i += len(opening)
        elif depth and text.startswith(closing, i):
            depth -= 1; i += len(closing)
            if depth == 0:
                pieces.append(discarded_span(text[start:i])); start = i
        else:
            i += 1
    pieces.append(discarded_span(text[start:]) if depth else text[start:])
    return ''.join(pieces)


def prepare_source_text(text):
    text = str(text or '')
    for pattern, exemption in SOURCE_PATTERNS:
        text = pattern.sub(lambda m: m.group() if exemption and exemption.fullmatch(m.group())
                           else discarded_span(m.group()), text)
    for opening, closing in DATA_POLICY['source_filter']['balanced_spans']:
        if opening in text:
            text = discard_balanced(text, opening, closing)
    return text


def symbol_enclosures(line, maximum=None):
    """Code-point positions, bounded total width, no pairing across discarded text."""
    maximum = DATA_POLICY['symbol_window']['max_distance'] if maximum is None else maximum
    result = {}; left = None
    for right, character in enumerate(line):
        if character == BARRIER:
            left = None
        elif character in SYMBOLS:
            if left is not None and right-left <= maximum:
                for index in range(left+1, right):
                    if line[index] in PROXY_PUNCTUATION:
                        result[index] = right-left
            left = right
    return result


def valid_boundary_context(left, right):
    """Reject only when BOTH immediate normalized neighbors are non-Han."""
    return bool((left and HAN.fullmatch(left[-1])) or (right and HAN.fullmatch(right[0])))


def training_fragment_lines(text, *, symbol_window=None):
    """Keep rejected fragments as barriers; preserve source glyphs until labeling."""
    for raw_line in LINE_BOUNDARIES.split(prepare_source_text(text)):
        # Preserve distances for the window; collapse whitespace only in fragments.
        line = re.sub(r'\s', ' ', raw_line)
        empty_spans = [match.span() for match in EMPTY_TEMPLATE.finditer(line)]
        numeric_commas = {match.start(1) for match in NUMERIC_COMMAS.finditer(line)} if '，' in line else set()
        surrounded = symbol_enclosures(line, symbol_window)
        fragments = []
        start = 0
        def add(end, punctuation, boundary_reasons=None):
            raw = line[start:end]
            value = normalized_fragment(raw)
            if not value:
                if fragments:
                    fragments[-1]['punctuation'] += punctuation
                    fragments[-1]['boundary_reasons'].extend(boundary_reasons or [])
                return
            reasons = [name for name, pattern in SURFACE_PATTERNS.items() if pattern.search(raw) or pattern.search(value)]
            if DATA_POLICY['sample_filter']['require_han'] and not HAN.search(value):
                reasons.append('fragment_without_han')
            if WHOLE_FRAGMENT.fullmatch(raw.strip()) or WHOLE_FRAGMENT.fullmatch(value):
                reasons.append('web_control')
            if any(left < end and right > start for left, right in empty_spans):
                reasons.append('empty_template')
            fragments.append({'text': value, 'punctuation': punctuation, 'reasons': reasons,
                              'boundary_reasons': boundary_reasons or []})
        for i, character in enumerate(line):
            if character in PROXY_PUNCTUATION and i not in numeric_commas:
                # Still delimit fragments: a rejected target must not fuse its sides.
                add(i, character, ['symbol_window'] if i in surrounded else [])
                start = i+1
        add(len(line), '')
        yield fragments


def training_pairs(text, rejection_counts=None, *, symbol_window=None):
    for fragments in training_fragment_lines(text, symbol_window=symbol_window):
        for left, right in zip(fragments, fragments[1:]):
            if not any(c.isalnum() for c in left['text']) or not any(c.isalnum() for c in right['text']):
                continue
            reasons = set(left['reasons'] + right['reasons'] + left['boundary_reasons'])
            if not valid_boundary_context(left['text'], right['text']):
                reasons.add('both_neighbors_non_han')
            if reasons:
                if rejection_counts is not None:
                    rejection_counts['pairs'] = rejection_counts.get('pairs', 0) + 1
                    for reason in reasons:
                        rejection_counts[reason] = rejection_counts.get(reason, 0) + 1
                continue
            yield left['text'] + right['text'], len(left['text']) - 1, left['punctuation']


def valid_proxy_label(label):
    return isinstance(label, str) and bool(label) and all(c in PROXY_PUNCTUATION for c in label)


def require_data_policy(summary):
    proxies = summary.get("proxy_punctuation")
    if (summary.get("standard") != DATA_POLICY["standard"]
            or summary.get("tokenization") != "character"
            or summary.get("input_representation") != DATA_POLICY["input_representation"]
            or not isinstance(proxies, list)
            or not all(isinstance(c, str) for c in proxies)
            or set(proxies) != PROXY_PUNCTUATION
            or summary.get("normalization") != DATA_POLICY["normalization"]
            or summary.get("numeric_punctuation") != DATA_POLICY["numeric_punctuation"]
            or summary.get("boundary_context") != DATA_POLICY["boundary_context"]
            or summary.get("line_boundaries") != DATA_POLICY["line_boundaries"]
            or summary.get("source_filter") != DATA_POLICY["source_filter"]
            or summary.get("symbol_window") != DATA_POLICY["symbol_window"]
            or summary.get("sample_filter") != DATA_POLICY["sample_filter"]):
        raise ValueError(f"Data requires {DATA_POLICY['standard']}; regenerate original documents in a fresh directory")
