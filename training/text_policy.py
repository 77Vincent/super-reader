"""Shared, versioned input and supervision contract (no ML dependencies)."""
import json
import re
import unicodedata
from pathlib import Path

DATA_POLICY = json.loads(Path(__file__).with_name("text-policy.json").read_text())
PROXY_PUNCTUATION = frozenset(DATA_POLICY["proxy_punctuation"])
LINE_BOUNDARIES = re.compile(DATA_POLICY['line_boundaries'])
SURFACE_PATTERNS = {key: re.compile(value) for key, value in DATA_POLICY['sample_filter']['fragment_patterns'].items()}
WHOLE_FRAGMENT = re.compile(DATA_POLICY['sample_filter']['whole_fragment_pattern'])
EMPTY_TEMPLATE = re.compile(DATA_POLICY['sample_filter']['empty_template_pattern'])


def normalized_fragment(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip()


def training_fragment_lines(text):
    """Keep rejected fragments as barriers; preserve source glyphs until labeling."""
    for raw_line in LINE_BOUNDARIES.split(str(text or '')):
        line = re.sub(r'\s+', ' ', raw_line).strip()
        empty_spans = [match.span() for match in EMPTY_TEMPLATE.finditer(line)]
        fragments = []
        start = 0
        def add(end, punctuation):
            raw = line[start:end]
            value = normalized_fragment(raw)
            if not value:
                if fragments:
                    fragments[-1]['punctuation'] += punctuation
                return
            reasons = [name for name, pattern in SURFACE_PATTERNS.items() if pattern.search(raw) or pattern.search(value)]
            if WHOLE_FRAGMENT.fullmatch(raw.strip()) or WHOLE_FRAGMENT.fullmatch(value):
                reasons.append('web_control')
            if any(left < end and right > start for left, right in empty_spans):
                reasons.append('empty_template')
            fragments.append({'text': value, 'punctuation': punctuation, 'reasons': reasons})
        for i, character in enumerate(line):
            numeric = character == '，' and 0 < i < len(line)-1 and line[i-1].isdecimal() and line[i+1].isdecimal()
            if character in PROXY_PUNCTUATION and not numeric:
                add(i, character)
                start = i+1
        add(len(line), '')
        yield fragments


def training_pairs(text, rejection_counts=None):
    for fragments in training_fragment_lines(text):
        for left, right in zip(fragments, fragments[1:]):
            if not any(c.isalnum() for c in left['text']) or not any(c.isalnum() for c in right['text']):
                continue
            reasons = set(left['reasons'] + right['reasons'])
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
            or summary.get("line_boundaries") != DATA_POLICY["line_boundaries"]
            or summary.get("sample_filter") != DATA_POLICY["sample_filter"]):
        raise ValueError(f"Data requires {DATA_POLICY['standard']}; regenerate original documents in a fresh directory")
