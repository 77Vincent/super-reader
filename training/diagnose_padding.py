"""Read-only Metal padding sensitivity check; does not train or export weights.

Run after diagnose_good_teacher_errors.mjs, using training/.deps on PYTHONPATH.
"""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'training/.deps'), str(ROOT / 'training')]
import numpy as np
import torch
from train_smoke import BoundaryChooser, configure_cpu, configure_mps, iterate_batches

OUT = ROOT / 'training/artifacts/good-teacher-diagnosis-v7-20260924'
release = ROOT / 'training/artifacts/learning-rate-192ch-full-246m-v7-20260923/epoch-1-backend'
configure_cpu(1, 1)
device = configure_mps()
state = torch.load(release / 'training-state.pt', map_location='cpu', weights_only=True)
vocab = state['vocabulary']
model = BoundaryChooser(len(vocab), 192, 8).to(device)
model.load_state_dict(state['best_state'], strict=True)
model.eval()
replay = json.loads((OUT / 'replay.json').read_text())
groups = defaultdict(list)
for row in replay['rows']:
    groups[row['n']].append(row)
results = {r['id']: {} for r in replay['rows']}
# Recover the exact padding used in the original 24,000-row audit, including
# rows that were the longest member of their batch and received no padding.
original_records = [json.loads(line) for line in (ROOT/'training/artifacts/training-teacher-audit-v7-20260924/sample.jsonl').open()]
for r in original_records:
    r['token_ids'] = [vocab.get(c, vocab['<unk>']) for c in r['text']]
    r['training_weight'] = 1.0
original_padding = {}
for batch in iterate_batches(original_records, 512, 8192, shuffle=False, seed=0):
    for r in batch['records']:
        if r['id'] in results:
            original_padding[r['id']] = batch['token_ids'].shape[1] - len(r['text'])
assert len(original_padding) == len(results)
del original_records

def describe(scores, row):
    scores = scores.astype(np.float64)
    p = np.exp(scores - scores.max()); p /= p.sum()
    g = row['gold']
    return {'predicted': int(scores.argmax()), 'confidence': float(p.max()),
            'gold_probability': float(p[g]), 'gold_rank': int(1 + (scores > scores[g]).sum()),
            'scores': scores.tolist()}

with torch.inference_mode():
    for length, rows in sorted(groups.items()):
        # Equal real lengths: this allows a no-padding batch as the control.
        for extra in (0, 1, 2, 16):
            ids = torch.zeros((len(rows), length + extra), dtype=torch.long, device=device)
            ids[:, :length] = torch.tensor([[vocab.get(c, vocab['<unk>']) for c in r['text']] for r in rows], device=device)
            mask = torch.zeros_like(ids, dtype=torch.float32); mask[:, :length] = 1
            gaps = torch.zeros((len(rows), length + extra - 1), dtype=torch.bool, device=device); gaps[:, :length-1] = True
            values = model(ids, mask, gaps)[:, :length-1].cpu().numpy()
            for r, scores in zip(rows, values):
                results[r['id']][str(extra)] = describe(scores, r)
        # Diagnostic control only: suppress padded activations inside each
        # block. This is not installed in the trainer or production backend.
        handles = []
        for block in model.blocks:
            for layer in (block.normalization, block.first):
                handles.append(layer.register_forward_hook(lambda module, inputs, output: output * mask.unsqueeze(-1)))
        try:
            values = model(ids, mask, gaps)[:, :length-1].cpu().numpy()
            for r, scores in zip(rows, values):
                results[r['id']]['masked16'] = describe(scores, r)
        finally:
            for handle in handles:
                handle.remove()
        if length % 10 == 0:
            print('Checked length', length, flush=True)

summary = {
    'n': len(results),
    'zero_padding_vs_js_top1_matches': sum(results[r['id']]['0']['predicted'] == r['predicted'] for r in replay['rows']),
    'zero_padding_vs_js_max_logit_difference': max(abs(np.array(results[r['id']]['0']['scores']) - r['scores']).max().item() for r in replay['rows']),
    'pad2_vs_pad16_top1_matches': sum(v['2']['predicted'] == v['16']['predicted'] for v in results.values()),
    'pad2_vs_pad16_max_logit_difference': max(abs(np.array(v['2']['scores']) - v['16']['scores']).max().item() for v in results.values()),
    'pad0_vs_pad2_changed_top1': sum(v['0']['predicted'] != v['2']['predicted'] for v in results.values()),
    'pad2_matches_historical_top1': sum(results[r['id']]['2']['predicted'] == r['storedPrediction']['predicted'] for r in replay['rows']),
    'reconstructed_original_padding_matches_historical_top1': sum(results[r['id']][str(min(2,original_padding[r['id']]))]['predicted'] == r['storedPrediction']['predicted'] for r in replay['rows']),
    'reconstructed_original_padding_max_confidence_difference': max(abs(results[r['id']][str(min(2,original_padding[r['id']]))]['confidence'] - r['storedPrediction']['confidence']) for r in replay['rows']),
    'masked_pad16_vs_pad0_top1_matches': sum(v['masked16']['predicted'] == v['0']['predicted'] for v in results.values()),
    'masked_pad16_vs_pad0_max_logit_difference': max(abs(np.array(v['masked16']['scores']) - v['0']['scores']).max().item() for v in results.values()),
    'pad0_accuracy': sum(results[r['id']]['0']['predicted'] == r['gold'] for r in replay['rows']) / len(results),
    'pad2_accuracy': sum(results[r['id']]['2']['predicted'] == r['gold'] for r in replay['rows']) / len(results),
}
payload = {'experiment': 'Identical weights, input characters, masks and real gaps; vary right padding only. Equal-length groups, Metal eval mode, no gradients.',
           'checkpoint_sha256': hashlib.sha256((release/'training-state.pt').read_bytes()).hexdigest(),
           'torch_version': torch.__version__, 'device': str(device), 'summary': summary,
           'original_batch_padding': original_padding, 'rows': results}
(OUT / 'padding.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(summary, indent=2))
