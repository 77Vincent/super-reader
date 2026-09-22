#!/usr/bin/env python3
"""Check identity depth expansion, then compare complete validation by context."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import subprocess
import sys

from run_web_continuation import read, write


def length_bucket(length):
    for upper, label in ((17, '2-17'), (34, '18-34'), (42, '35-42'),
                         (64, '43-64'), (128, '65-128')):
        if length <= upper:
            return label
    return '129+'


def coverage_bucket(length, target_index):
    # Gap indices are zero based. A 16-layer gap sees 17 tokens on each side;
    # unused context at an edge cannot be borrowed by the other side.
    if not 0 <= target_index < length - 1:
        raise ValueError('Invalid gold gap')
    left = target_index + 1
    furthest_side = max(left, length - left)
    if furthest_side <= 17:
        return 'covered-by-16'
    if furthest_side <= 21:
        return 'newly-covered-by-20'
    return 'not-fully-covered-by-20'


def export_model(model, vocabulary, folder, snapshot, *, epoch):
    import torch
    from text_policy import DATA_POLICY
    from train_smoke import save_browser_compatible_checkpoint
    from widen_boundary import make_inputs
    folder.mkdir(parents=True, exist_ok=True)
    write(folder / 'boundary-smoke-vocabulary.json', vocabulary)
    save_browser_compatible_checkpoint(model, folder / 'boundary-smoke.safetensors')
    write(folder / 'smoke-metrics.json', {
        **DATA_POLICY, 'best_epoch': epoch, 'test': {'accuracy': None},
        'purpose': 'Inference benchmark export; not a production promotion or a test evaluation',
        'architecture': {'channels': model.embedding.weight.shape[1], 'residual_blocks': len(model.blocks),
                         'convolutions_per_block': 2, 'kernel_size': 3, 'dilation': 1},
    })
    texts = [c['text'] for c in read(snapshot / 'test/model-backend-reference.json')['cases']]
    texts += ['即便是完全符合语法的通顺的但没有任何标点断句的句子模型依然能够找到恰当的切分点',
              '我们希望读者能够更加轻松地找到句子中的重点并理解不同段落之间的联系' * 4]
    cases = []
    model.eval()
    with torch.inference_mode():
        for text in texts:
            scores = model(*make_inputs([text], vocabulary, next(model.parameters()).device))[0].cpu().tolist()
            cases.append({'text': text, 'scores': scores, 'bestGap': max(range(len(scores)), key=scores.__getitem__)})
    write(folder / 'reference.json', {'cases': cases})
    subprocess.run([sys.executable, str(snapshot / 'training/export_browser_model.py'),
                    '--artifact-dir', str(folder), '--output', str(folder / 'boundary-model-data.js')], check=True)


def preflight(run, plan, device):
    import torch
    from torch.nn import functional as F
    from train_sharded import expanded_initialization
    from train_smoke import BoundaryChooser, seed_everything
    from widen_boundary import make_inputs

    checkpoint = torch.load(run / 'initialization.pt', map_location='cpu', weights_only=True)
    vocabulary = read(run / 'vocabulary.json')
    if checkpoint['vocabulary'] != vocabulary:
        raise ValueError('Initialization vocabulary differs from the frozen pilot')
    source = checkpoint['best_state'] or checkpoint['model_state']
    if checkpoint['configuration']['residual_blocks'] != 8:
        raise ValueError('Expected a 16-layer initialization')
    # Match the real trainer's model construction order and seed. The old model
    # is only constructed afterwards so it cannot consume the candidate's RNG.
    seed_everything(plan['seed'])
    model = BoundaryChooser(len(vocabulary), 192, 10)
    transfer = expanded_initialization(model, run / 'initialization.pt', vocabulary)
    for name, expected in source.items():
        assert torch.equal(model.state_dict()[name], expected), name
    assert transfer['copied_blocks'] == 8 and transfer['added_identity_blocks'] == 2
    pristine = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    old = BoundaryChooser(len(vocabulary), 192, 8)
    old.load_state_dict(source)
    old.to(device).eval()
    model.to(device).eval()
    texts = ['甲乙', '先读短句', '女：那可挺麻烦的吃点儿治疗过敏的药吧',
             '上午8:30出发下午4:30返回', '价格是1,000.50元型号是AI-20',
             '课程（A）在明天开始请提前准备',
             '我们希望读者能够更加轻松地找到句子中的重点并理解不同段落之间的联系' * 4]
    inputs = make_inputs(texts, vocabulary, device)
    with torch.inference_mode():
        expected, actual = old(*inputs), model(*inputs)
        assert torch.equal(expected, actual), 'Identity expansion changed Metal logits'
    optimizer = torch.optim.AdamW(model.parameters(), lr=plan['rates']['layers20'], weight_decay=1e-4, foreach=True)
    assert not optimizer.state
    targets = torch.tensor([len(text) // 2 - 1 for text in texts], device=device)
    audit = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(*inputs), targets)
        loss.backward()
        for index in (8, 9):
            block = model.blocks[index]
            scale_grad = block.residual_scale.grad.item()
            branch_grad = block.first.weight.grad.abs().sum().item()
            assert torch.isfinite(block.residual_scale.grad).all() and abs(scale_grad) > 0
            assert (branch_grad == 0) if step == 0 else (branch_grad > 0)
            audit.append({'step': step + 1, 'block': index, 'scale_gradient': scale_grad,
                          'convolution_gradient_l1': branch_grad})
        torch.nn.utils.clip_grad_norm_(model.parameters(), plan['training']['gradient_clip'])
        optimizer.step()
    for index in (8, 9):
        assert model.blocks[index].residual_scale.item() != 0
        assert not torch.equal(model.blocks[index].first.weight.cpu(), pristine[f'blocks.{index}.first.weight'])
    # Restore ALL weights, including the newly initialized branches. Nothing
    # from these arbitrary-target learning probes is used by the real trainer.
    model.load_state_dict(pristine)
    folder = run / 'preflight'
    folder.mkdir(exist_ok=True)
    if not (folder / 'source').exists():
        (folder / 'source').symlink_to(run / 'source', target_is_directory=True)
    export_model(old, vocabulary, folder / 'layers16/final', run / 'source', epoch=0)
    export_model(model, vocabulary, folder / 'layers20/final', run / 'source', epoch=0)
    report = {'passed': True, 'device': str(device), 'initial_logits_exact': True,
        'copied_tensors_exact': True, 'new_blocks_learnable': True, 'gradient_audit': audit,
        'probe_updates_used_for_training': False, 'transfer': transfer,
        'parameters': {16: sum(p.numel() for p in old.parameters()), 20: sum(p.numel() for p in model.parameters())},
        'gap_context_tokens': {16: 34, 20: 42}}
    write(folder / 'verification.json', report)
    print(report, flush=True)


def analyze(run, plan, device):
    import torch
    from train_sharded import read_evaluation_records
    from train_smoke import BoundaryChooser, batch_to_device, iterate_batches

    vocabulary = read(run / 'vocabulary.json')
    models = {}
    reports = {}
    for depth in (16, 20):
        arm = run / f'layers{depth}'
        state = torch.load(arm / 'training-state.pt', map_location='cpu', weights_only=True)
        assert state['vocabulary'] == vocabulary
        assert state['progress']['epoch'] == 2 and len(state['history']) == 1
        assert state['configuration']['residual_blocks'] == depth // 2
        model = BoundaryChooser(len(vocabulary), 192, depth // 2)
        model.load_state_dict(state['model_state'])
        model.to(device).eval()
        models[depth] = model
        reports[depth] = read(arm / 'validation-only.json')
        export_model(model, vocabulary, arm / 'final', run / 'source', epoch=1)
        del state
    print('Reading complete validation for paired length/context analysis', flush=True)
    records = read_evaluation_records(Path(plan['evaluation_dir']) / 'validation.jsonl', vocabulary)
    assert len(records) == plan['evaluation']['validation']['count']
    counters = defaultdict(lambda: {'samples': 0, 'correct16': 0, 'correct20': 0,
                                   'fixed_by20': 0, 'regressed_by20': 0})
    with torch.inference_mode():
        for index, batch in enumerate(iterate_batches(records, plan['training']['batch_size'],
                plan['training']['max_tokens_per_batch'], shuffle=False, seed=0)):
            batch = batch_to_device(batch, device)
            predictions = {depth: model(batch['token_ids'], batch['token_mask'], batch['gap_mask']).argmax(1).cpu().tolist()
                           for depth, model in models.items()}
            for row, record in enumerate(batch['records']):
                length, target = len(record['token_ids']), record['target_index']
                a, b = predictions[16][row] == target, predictions[20][row] == target
                keys = ['all', 'length/' + length_bucket(length),
                        'gold-gap-coverage/' + coverage_bucket(length, target), 'domain/' + record['domain']]
                for key in keys:
                    c = counters[key]
                    c['samples'] += 1
                    c['correct16'] += a
                    c['correct20'] += b
                    c['fixed_by20'] += not a and b
                    c['regressed_by20'] += a and not b
            if (index + 1) % 500 == 0:
                print({'batches': index + 1, 'samples': counters['all']['samples']}, flush=True)
                torch.mps.empty_cache()
    for depth in (16, 20):
        expected = round(reports[depth]['endpoint_validation']['accuracy'] * len(records))
        if counters['all'][f'correct{depth}'] != expected:
            raise ValueError(f'Paired analysis differs from full endpoint validation for {depth} layers')
    for c in counters.values():
        for depth in (16, 20):
            c[f'accuracy{depth}'] = c[f'correct{depth}'] / c['samples']
        c['delta_percentage_points'] = 100 * (c['accuracy20'] - c['accuracy16'])
    write(run / 'validation-strata.json', {
        'split': 'complete validation; no test data used', 'samples': len(records),
        'coverage_definition': 'Full input at the GOLD gap: max(target_index+1, length-target_index-1) <= 17 or 21. Edge gaps cannot borrow missing context.',
        'strata': dict(counters),
    })
    print(counters['all'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    from run_smoke import ensure_dependencies
    ensure_dependencies()
    from train_smoke import configure_cpu, configure_mps
    configure_cpu(1, 1)
    device = configure_mps()
    run = args.run_dir.resolve()
    (preflight if args.preflight else analyze)(run, read(run / 'run.json'), device)


if __name__ == '__main__':
    main()
