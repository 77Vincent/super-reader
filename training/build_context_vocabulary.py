#!/usr/bin/env python3
"""Extend a checkpoint vocabulary using training shards only, retaining old IDs."""
import argparse
import json
import string
from collections import Counter
from pathlib import Path
from text_policy import require_data_policy

ROOT = Path(__file__).resolve().parent.parent


def extend_vocabulary(previous, counts, maximum=8192):
    if sorted(previous.values()) != list(range(len(previous))):
        raise ValueError("Source vocabulary IDs must be contiguous")
    result = dict(previous)
    # Ensure infrequent formatting characters and digits have distinct embeddings.
    required = string.printable[:95] + '、:：“”‘’「」『』（）()《》〈〉【】[]{}—–…·+-/%‰℃°=<>!?;,.。'
    for token in dict.fromkeys(required + ' '):
        if token not in result:
            result[token] = len(result)
    if len(result) > maximum:
        raise ValueError("Vocabulary cap cannot preserve the source and required context characters")
    for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if len(result) >= maximum:
            break
        if token not in result:
            result[token] = len(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--source-vocabulary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--maximum-size', type=int, default=8192)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    require_data_policy(manifest)
    counts = Counter()
    samples = 0
    for index, shard in enumerate(manifest['shards'], 1):
        path = ROOT / shard['path']
        if path.stat().st_size != shard['bytes']:
            raise ValueError(f'Shard size changed: {path}')
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                counts.update(row[0])
                samples += 1
        if index % 16 == 0:
            print(f'vocabulary shards={index}/{len(manifest["shards"])} samples={samples}', flush=True)
    previous = json.loads(args.source_vocabulary.read_text())
    vocabulary = extend_vocabulary(previous, counts, args.maximum_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(vocabulary, ensure_ascii=False, indent=2)+'\n')
    manifest['vocabulary_path'] = str(args.output.resolve())
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    total = sum(counts.values())
    print(json.dumps({'old_tokens':len(previous), 'new_vocabulary_size':len(vocabulary),
                      'training_samples':samples,'training_unknown_token_rate':sum(n for t,n in counts.items() if t not in vocabulary)/total}), flush=True)


if __name__ == '__main__':
    main()
