#!/usr/bin/env python3
"""Check 12-to-16-convolution transfer, learnability and exported inference."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    os.environ["DEBUG"] = "0"
    from run_smoke import ensure_dependencies
    ensure_dependencies()
    import torch
    from torch.nn import functional as F
    from train_sharded import expanded_initialization
    from train_smoke import BoundaryChooser, configure_cpu, save_browser_compatible_checkpoint
    from text_policy import DATA_POLICY
    configure_cpu(2, 1)
    torch.manual_seed(2026090405)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    assert checkpoint["best_epoch"] == 2
    assert checkpoint["configuration"]["channels"] == 192
    assert checkpoint["configuration"]["residual_blocks"] == 6
    vocabulary = checkpoint["vocabulary"]
    assert len(vocabulary) == 8192
    old = BoundaryChooser(len(vocabulary), 192, 6)
    old.load_state_dict(checkpoint["best_state"])
    model = BoundaryChooser(len(vocabulary), 192, 8)
    transfer = expanded_initialization(model, args.checkpoint, vocabulary)
    for name, expected in checkpoint["best_state"].items():
        assert torch.equal(model.state_dict()[name], expected), name
    assert transfer["copied_blocks"] == 6 and transfer["added_identity_blocks"] == 2
    assert sum(p.numel() for p in model.parameters()) == 3496329
    texts = ["女:那可挺麻烦的吃点儿治疗过敏的药吧", "上午8:30出发下午4:30返回",
             "价格是1,000.50元型号是AI-20", "他买了苹果、香蕉准备做果汁",
             "先读短句", "课程（A）在明天开始请提前准备",
             "我们希望读者能够更加轻松地找到句子中的重点并理解不同段落之间的联系" * 3]
    ids = torch.zeros((len(texts), max(map(len, texts))), dtype=torch.long)
    token_mask = torch.zeros_like(ids, dtype=torch.bool)
    for i, text in enumerate(texts):
        ids[i, :len(text)] = torch.tensor([vocabulary.get(c, vocabulary["<unk>"]) for c in text])
        token_mask[i, :len(text)] = True
    gap_mask = token_mask[:, 1:]
    old.eval()
    model.eval()
    with torch.inference_mode():
        expected = old(ids, token_mask, gap_mask)
        actual = model(ids, token_mask, gap_mask)
    assert torch.equal(expected, actual), "Identity expansion changed initial logits"
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4, foreach=True)
    assert not optimizer.state
    original = {i: model.blocks[i].first.weight.detach().clone() for i in (6, 7)}
    targets = torch.tensor([len(text) // 2 - 1 for text in texts])
    gradient_audit = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(ids, token_mask, gap_mask), targets)
        loss.backward()
        for index in (6, 7):
            block = model.blocks[index]
            assert torch.isfinite(block.residual_scale.grad).all()
            assert block.residual_scale.grad.abs().sum().item() > 0
            branch_gradient = block.first.weight.grad.abs().sum().item()
            assert (branch_gradient == 0) if step == 0 else (branch_gradient > 0)
            gradient_audit.append({"step": step + 1, "block": index,
                                   "scale_gradient": block.residual_scale.grad.item(),
                                   "convolution_gradient_l1": branch_gradient})
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    for index in (6, 7):
        assert model.blocks[index].residual_scale.item() != 0
        assert not torch.equal(original[index], model.blocks[index].first.weight)

    # Probe updates are discarded. Export and compare the untouched initialization.
    expanded_initialization(model, args.checkpoint, vocabulary)
    folder = args.output_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    save_browser_compatible_checkpoint(model, folder / "boundary-smoke.safetensors")
    (folder / "boundary-smoke-vocabulary.json").write_text(json.dumps(vocabulary, ensure_ascii=False))
    (folder / "smoke-metrics.json").write_text(json.dumps({
        **DATA_POLICY, "best_epoch": 0, "test": {"accuracy": None},
        "architecture": {"channels": 192, "residual_blocks": 8, "convolutions_per_block": 2,
                         "kernel_size": 3, "dilation": 1},
    }))
    subprocess.run([sys.executable, str(ROOT / "training/export_browser_model.py"),
                    "--artifact-dir", str(folder), "--output", str(folder / "boundary-model-data.js")], check=True)
    references = []
    with torch.inference_mode():
        for text in texts:
            tokens = torch.tensor([[vocabulary.get(c, vocabulary["<unk>"]) for c in text]])
            logits = model(tokens, torch.ones_like(tokens, dtype=torch.bool),
                           torch.ones((1, len(text) - 1), dtype=torch.bool))[0].tolist()
            references.append({"tokens": list(text), "scores": logits})
    reference_path = folder / "reference.json"
    reference_path.write_text(json.dumps(references, ensure_ascii=False))
    js = """const fs=require('node:fs'),vm=require('node:vm');
const c=vm.createContext({atob,Intl});
for(const p of process.argv.slice(1,3))vm.runInContext(fs.readFileSync(p,'utf8'),c);
let error=0;const rows=JSON.parse(fs.readFileSync(process.argv[3]));
for(const row of rows){const got=c.SuperReaderModelBackend.scoreTokens(row.tokens);
if(got.length!==row.scores.length)throw Error('Gap count mismatch');
got.forEach((s,i)=>{if(!Number.isFinite(s))throw Error('Nonfinite logit');error=Math.max(error,Math.abs(s-row.scores[i]));});
if(got.indexOf(Math.max(...got))!==row.scores.indexOf(Math.max(...row.scores)))throw Error('Prediction mismatch');}
if(error>0.0002)throw Error('Logit mismatch: '+error);
console.log(JSON.stringify({cases:rows.length,maximum_absolute_logit_error:error}));"""
    browser = json.loads(subprocess.check_output(["node", "-e", js,
        str(folder / "boundary-model-data.js"), str(ROOT / "src/backend/inference.js"), str(reference_path)]))
    report = {"passed": True, "source_best_epoch": 2, "copied_tensors_exact": True,
              "initial_logits_exact": True, "parameters": 3496329, "transfer": transfer,
              "new_blocks_learnable": True, "gradient_audit": gradient_audit,
              "browser": browser, "probe_updates_used_for_training": False}
    (folder / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
