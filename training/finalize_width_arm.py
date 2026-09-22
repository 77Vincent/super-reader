#!/usr/bin/env python3
"""Evaluate/export the equal-update endpoint, even if validation preferred step zero."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch

from train_sharded import read_evaluation_records
from train_smoke import BoundaryChooser, configure_cpu, configure_mps, evaluate, predict_record, save_browser_compatible_checkpoint
from widen_boundary import make_inputs
from run_web_continuation import read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    arm = args.artifact_dir.resolve()
    out = arm / 'final'
    out.mkdir(exist_ok=True)
    configure_cpu(1, 1)
    device = configure_mps()
    checkpoint = torch.load(arm / 'training-state.pt', map_location='cpu', weights_only=True)
    if checkpoint['progress']['epoch'] != 2 or len(checkpoint['history']) != 1:
        raise ValueError('The one-epoch comparison arm is not finished')
    metrics = read(arm / 'smoke-metrics.json')
    vocabulary = checkpoint['vocabulary']
    model = BoundaryChooser(len(vocabulary), metrics['architecture']['channels'], metrics['architecture']['residual_blocks'])
    model.load_state_dict(checkpoint['model_state'])
    model.to(device).eval()
    if metrics['best_epoch'] != 1:
        # The ordinary trainer selects its best validation checkpoint, possibly
        # the initialization. Comparing it to another arm's epoch 1 would hide
        # the equal-update result. Keep both views; never replace the raw run.
        for split in ('validation', 'test'):
            print(f'evaluating final epoch on complete {split}', flush=True)
            records = read_evaluation_records(args.evaluation / f'{split}.jsonl', vocabulary)
            metrics[split] = evaluate(model, records, 512, 8192)
            if split == 'test':
                metrics['sample_predictions'] = [predict_record(model, record, 8192) for record in records[:4]]
            del records
            gc.collect()
            torch.mps.empty_cache()
    metrics['selected_best_epoch_in_original_run'] = metrics['best_epoch']
    metrics['best_epoch'] = 1
    metrics['comparison_export'] = 'final epoch after equal updates; best-checkpoint results remain in parent directory'
    write(out / 'smoke-metrics.json', metrics)
    shutil.copy2(arm / 'boundary-smoke-vocabulary.json', out / 'boundary-smoke-vocabulary.json')
    save_browser_compatible_checkpoint(model, out / 'boundary-smoke.safetensors')
    reference = read(args.reference)
    texts = [case['text'] for case in reference['cases']]
    texts += ['即便是完全符合语法的通顺的但没有任何标点断句的句子模型依然能够找到恰当的切分点',
              '我们希望读者能够更加轻松地找到句子中的重点并理解不同段落之间的联系' * 4]
    cases = []
    with torch.inference_mode():
        for text in texts:
            scores = model(*make_inputs([text], vocabulary, device))[0].cpu().tolist()
            cases.append({'text': text, 'scores': scores, 'bestGap': max(range(len(scores)), key=scores.__getitem__)})
    write(out / 'reference.json', {'cases': cases})
    subprocess.run([sys.executable, str(Path(__file__).with_name('export_browser_model.py')),
                    '--artifact-dir', str(out), '--output', str(out / 'boundary-model-data.js')], check=True)
    write(out / 'complete.json', {'epoch': 1, 'validation_accuracy': metrics['validation']['accuracy'],
                                 'test_accuracy': metrics['test']['accuracy']})


if __name__ == '__main__':
    main()
