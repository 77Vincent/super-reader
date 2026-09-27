"""Check frozen production weights against padding variants and benchmark the fix.

This never writes weights. The microbenchmark's optimizer steps are discarded.
"""
import argparse
from collections import defaultdict
import gc
import importlib.util
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from train_smoke import BoundaryChooser, MODEL_EXECUTION_CONTRACT, configure_cpu, configure_mps
from run_web_continuation import sha, write

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    release = ROOT / 'training/artifacts/learning-rate-192ch-full-246m-v7-20260923'
    checkpoint = release / 'epoch-1-backend/training-state.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    configure_cpu(1, 1)
    device = configure_mps()
    model = BoundaryChooser(len(state['vocabulary']), 192, 8).to(device)
    model.load_state_dict(state['best_state'], strict=True)
    model.eval()
    rows = json.loads((ROOT / 'training/artifacts/good-teacher-diagnosis-v7-20260924/replay.json').read_text())['rows']
    groups = defaultdict(list)
    for row in rows:
        groups[row['n']].append(row)
    maximum_error, maximum_js_error, comparisons = 0., 0., 0
    with torch.inference_mode():
        for length, group in sorted(groups.items()):
            reference = None
            for extra in (0, 1, 2, 17):
                ids = torch.zeros((len(group), length + extra), dtype=torch.long, device=device)
                ids[:, :length] = torch.tensor([[state['vocabulary'].get(c, 1) for c in r['text']] for r in group], device=device)
                mask = torch.zeros_like(ids, dtype=torch.bool)
                mask[:, :length] = True
                scores = model(ids, mask, mask[:, :-1] & mask[:, 1:])[:, :length - 1].cpu()
                if reference is None:
                    reference = scores
                    js = torch.tensor([r['scores'] for r in group])
                    maximum_js_error = max(maximum_js_error, (scores - js).abs().max().item())
                    assert torch.equal(scores.argmax(1), js.argmax(1))
                    torch.testing.assert_close(scores, js, rtol=0, atol=.001)
                else:
                    maximum_error = max(maximum_error, (scores - reference).abs().max().item())
                    assert torch.equal(scores.argmax(1), reference.argmax(1))
                    torch.testing.assert_close(scores, reference, rtol=0, atol=.001)
                    comparisons += len(group)
    del model, scores, ids, mask
    gc.collect()
    torch.mps.empty_cache()
    # Load the immutable historical implementation only as a timing reference.
    old_path = release / 'source/training/train_smoke.py'
    spec = importlib.util.spec_from_file_location('frozen_unmasked_reference', old_path)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    timings = {}
    for name, factory in [('historical', old.BoundaryChooser), ('fixed', BoundaryChooser)]:
        model = factory(len(state['vocabulary']), 192, 8).to(device)
        model.load_state_dict(state['best_state'], strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4, foreach=True)
        measurements = []
        for batch_size, width in ((512, 16), (256, 32), (128, 64)):
            torch.manual_seed(2026092502)
            ids = torch.randint(2, len(state['vocabulary']), (batch_size, width), device=device)
            mask = torch.ones_like(ids, dtype=torch.bool)
            mask[:batch_size // 2, -3:] = False
            gaps = mask[:, :-1] & mask[:, 1:]
            targets = torch.full((batch_size,), width // 2 - 1, dtype=torch.long, device=device)
            for step in range(45):
                if step == 5:
                    torch.mps.synchronize()
                    started = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(ids, mask, gaps), targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
            torch.mps.synchronize()
            seconds = time.perf_counter() - started
            measurements.append({'batch_size': batch_size, 'width': width, 'steps': 40,
                                 'seconds': seconds, 'samples_per_second': 40 * batch_size / seconds})
        timings[name] = measurements
        del model, optimizer, loss, ids, mask, gaps, targets
        gc.collect()
        torch.mps.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {'passed': True, 'execution_contract': MODEL_EXECUTION_CONTRACT,
              'checkpoint_sha256': sha(checkpoint), 'samples': len(rows),
              'padding_comparisons': comparisons, 'maximum_padding_logit_error': maximum_error,
              'maximum_js_logit_error': maximum_js_error, 'top1_all_equal': True,
              'microbenchmark': timings,
              'timing_caveat': 'Synthetic fixed shapes; excludes data loading, evaluation, restarts and thermal variability.'}
    write(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
