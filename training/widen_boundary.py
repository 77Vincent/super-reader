#!/usr/bin/env python3
"""Widen this LayerNorm residual CNN while preserving its initial function."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from train_smoke import BoundaryChooser, configure_cpu, configure_mps, save_training_state


def widen_state(source, target):
    """Embed each residual vector x as sqrt(D/C) * [x, mean(x), ...].

    This preserves its variance, including LayerNorm's epsilon. Compensate
    LayerNorm's affine scale and the gap head (including the product feature).
    Extra first-convolution/head neurons keep independent random features, with
    zero outgoing weights initially. They can learn without changing step zero.
    The vocabulary, block count, kernel size and activations must stay fixed.
    """
    source = {name: value.detach().cpu() for name, value in source.items()}
    result = {name: value.detach().cpu().clone() for name, value in target.items()}
    c, d = source['embedding.weight'].shape[1], result['embedding.weight'].shape[1]
    if d <= c or source.keys() != result.keys():
        raise ValueError('Only channel expansion with identical block structure is supported')
    if source['embedding.weight'].shape[0] != result['embedding.weight'].shape[0]:
        raise ValueError('Widening requires the identical vocabulary and row order')
    scale = math.sqrt(d / c)
    result['embedding.weight'][:, :c] = scale * source['embedding.weight']
    result['embedding.weight'][:, c:] = scale * source['embedding.weight'].mean(1, keepdim=True)
    for prefix in [name.removesuffix('.residual_scale') for name in source if name.endswith('.residual_scale')]:
        result[prefix + '.normalization.weight'][:c] = source[prefix + '.normalization.weight'] / scale
        result[prefix + '.normalization.bias'][:c] = source[prefix + '.normalization.bias']
        result[prefix + '.normalization.weight'][c:] = 1
        result[prefix + '.normalization.bias'][c:] = 0
        first = result[prefix + '.first.weight']
        first[:c].zero_()
        first[:c, :c] = source[prefix + '.first.weight']
        result[prefix + '.first.bias'][:c] = source[prefix + '.first.bias']
        second = result[prefix + '.second.weight']
        second.zero_()
        second[:c, :c] = scale * source[prefix + '.second.weight']
        second[c:, :c] = scale * source[prefix + '.second.weight'].mean(0, keepdim=True)
        result[prefix + '.second.bias'][:c] = scale * source[prefix + '.second.bias']
        result[prefix + '.second.bias'][c:] = scale * source[prefix + '.second.bias'].mean()
        result[prefix + '.residual_scale'] = source[prefix + '.residual_scale'].clone()
    result['boundary_hidden.weight'][:c].zero_()
    for group in range(4):
        divisor = scale ** (2 if group == 3 else 1)
        result['boundary_hidden.weight'][:c, group*d:group*d+c] = source['boundary_hidden.weight'][:, group*c:(group+1)*c] / divisor
    result['boundary_hidden.bias'][:c] = source['boundary_hidden.bias']
    result['boundary_output.weight'].zero_()
    result['boundary_output.weight'][:, :c] = source['boundary_output.weight']
    result['boundary_output.bias'] = source['boundary_output.bias'].clone()
    if not all(torch.isfinite(value).all() for value in result.values()):
        raise ValueError('Non-finite widened weights')
    return result


def make_inputs(texts, vocabulary, device):
    ids = torch.zeros((len(texts), max(map(len, texts))), dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for row, text in enumerate(texts):
        ids[row, :len(text)] = torch.tensor([vocabulary.get(c, vocabulary['<unk>']) for c in text])
        mask[row, :len(text)] = True
    return ids.to(device), mask.to(device), mask[:, 1:].to(device)


def verify(old, wide, inputs):
    """Check masked logits/probabilities, then discard two learning probes."""
    c = old.embedding.weight.shape[1]
    old.eval()
    wide.eval()
    with torch.inference_mode():
        expected, actual = old(*inputs), wide(*inputs)
        mask = inputs[2]
        torch.testing.assert_close(actual[mask], expected[mask], atol=2e-4, rtol=2e-5)
        probability_error = (actual.softmax(1) - expected.softmax(1)).abs().max().item()
        if probability_error > 5e-5 or not torch.equal(actual.argmax(1), expected.argmax(1)):
            raise AssertionError('Widening changed initial predictions')
        report = {'maximum_logit_error': (actual[mask] - expected[mask]).abs().max().item(),
                  'maximum_probability_error': probability_error, 'same_argmax': True,
                  'cases': len(expected), 'valid_gaps': mask.sum().item()}
    original = {name: value.detach().cpu().clone() for name, value in wide.state_dict().items()}
    optimizer = torch.optim.AdamW(wide.parameters(), lr=0.0003, weight_decay=1e-4, foreach=True)
    targets = (inputs[1].sum(1) // 2 - 1).long()
    gradients = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(wide(*inputs), targets)
        loss.backward()
        values = [block.second.weight.grad[:, c:].abs().sum().item() for block in wide.blocks]
        values.append(wide.boundary_output.weight.grad[:, c:].abs().sum().item())
        if step == 1:
            values.extend(block.first.weight.grad[c:].abs().sum().item() for block in wide.blocks)
            values.append(wide.boundary_hidden.weight.grad[c:].abs().sum().item())
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise AssertionError('New capacity has missing or non-finite gradients')
        gradients.append({'step': step + 1, 'minimum_new_branch_gradient_l1': min(values)})
        torch.nn.utils.clip_grad_norm_(wide.parameters(), 1)
        optimizer.step()
    wide.load_state_dict(original)
    report['gradient_probes'] = gradients
    report['passed'] = True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--channels', type=int, default=256)
    parser.add_argument('--seed', type=int, default=2026092201)
    args = parser.parse_args()
    configure_cpu(1, 1)
    device = configure_mps()
    torch.manual_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    state = checkpoint['best_state'] or checkpoint['model_state']
    vocabulary = checkpoint['vocabulary']
    blocks = sum(name.endswith('.residual_scale') for name in state)
    old = BoundaryChooser(len(vocabulary), state['embedding.weight'].shape[1], blocks)
    old.load_state_dict(state)
    wide = BoundaryChooser(len(vocabulary), args.channels, blocks)
    widened = widen_state(state, wide.state_dict())
    wide.load_state_dict(widened)
    texts = [case['text'] for case in json.loads(args.reference.read_text())['cases']]
    # Include padding, very short strings and contexts longer than the receptive field.
    texts += ['中文', '我们希望读者能够更加轻松地找到句子中的重点并理解不同段落之间的联系' * 4]
    report = verify(old.to(device), wide.to(device), make_inputs(texts, vocabulary, device))
    report.update({'source_channels': old.embedding.weight.shape[1], 'target_channels': args.channels,
                   'source_parameters': sum(p.numel() for p in old.parameters()),
                   'target_parameters': sum(p.numel() for p in wide.parameters()),
                   'method': 'variance-preserving mean padding with compensated LayerNorm and gap head',
                   'optimizer': 'fresh AdamW in both comparison arms; probe updates discarded'})
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    save_training_state(out / 'initialization.pt', {'best_state': widened, 'model_state': widened,
        'best_epoch': checkpoint['best_epoch'], 'vocabulary': vocabulary, 'widening': report})
    (out / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
