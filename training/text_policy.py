"""Shared, versioned input and supervision contract (no ML dependencies)."""
import json
import unicodedata
from pathlib import Path

DATA_POLICY = json.loads(Path(__file__).with_name("text-policy.json").read_text())
PROXY_PUNCTUATION = frozenset(DATA_POLICY["proxy_punctuation"])


def normalize_context(text):
    # NFKC expands … to three ASCII periods, losing the proxy/context distinction.
    return "…".join(unicodedata.normalize("NFKC", part) for part in text.split("…"))


def valid_proxy_label(label):
    return isinstance(label, str) and bool(label) and all(c in PROXY_PUNCTUATION for c in label)


def require_data_policy(summary):
    if (summary.get("standard") != DATA_POLICY["standard"]
            or summary.get("tokenization") != "character"
            or summary.get("input_representation") != DATA_POLICY["input_representation"]
            or not set(DATA_POLICY["excluded_proxy_punctuation"]).issubset(summary.get("excluded_proxy_punctuation", []))):
        raise ValueError(f"Data requires {DATA_POLICY['standard']}; regenerate original documents in a fresh directory")
