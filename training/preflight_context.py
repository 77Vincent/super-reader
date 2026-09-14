#!/usr/bin/env python3
"""Verify checkpoint transfer, a real training step, export and browser parity."""
import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    os.environ['DEBUG'] = '0'
    from run_smoke import ensure_dependencies
    ensure_dependencies()
    import torch
    from text_policy import DATA_POLICY
    from build_context_vocabulary import extend_vocabulary
    from train_sharded import expanded_initialization
    from train_smoke import BoundaryChooser, configure_cpu
    configure_cpu(2, 1)
    torch.manual_seed(20260913)
    checkpoint = torch.load(run / 'initialization.pt', map_location='cpu', weights_only=True)
    assert checkpoint['best_epoch'] == 2
    old_vocabulary = checkpoint['vocabulary']
    vocabulary = extend_vocabulary(old_vocabulary, Counter('🌈𠮷é'), 8192)
    # Deliberately permute two old IDs to detect accidental row-based copying.
    a, b = [t for t, i in old_vocabulary.items() if i in (2, 3)]
    vocabulary[a], vocabulary[b] = vocabulary[b], vocabulary[a]
    model = BoundaryChooser(len(vocabulary), 192, 6)
    transfer = expanded_initialization(model, run / 'initialization.pt', vocabulary)
    source = checkpoint['best_state']
    for name, tensor in model.state_dict().items():
        if name != 'embedding.weight':
            assert torch.equal(tensor, source[name]), name
    for token, index in old_vocabulary.items():
        assert torch.equal(model.embedding.weight[vocabulary[token]], source['embedding.weight'][index]), token
    texts = ['女：那可挺麻烦的，吃点儿治疗过敏的药吧。', '上午8:30出发，下午4:30返回。',
             '价格是1,000.50元，型号是AI-20。', '先看“🌈𠮷”，再读café。',
             '他买了苹果、香蕉，准备做果汁。', '版本３．１４已经发布，可以开始使用。',
             '课程（A）在明天开始，请提前准备。', '我打算上午9:30出发，下午回来。']
    js = """import {buildAdjacentSamples} from './training/prepare_smoke_data.mjs';
let text=''; for await (const c of process.stdin) text+=c;
console.log(JSON.stringify(JSON.parse(text).flatMap((text,i)=>buildAdjacentSamples({id:'preflight:'+i,domain:'fixture',text}))));"""
    rows = json.loads(subprocess.check_output(['node', '--input-type=module', '-e', js], cwd=ROOT,
                                               input=json.dumps(texts).encode()))
    folder = run / 'preflight'
    data = folder / 'data'
    data.mkdir(parents=True, exist_ok=True)
    (folder / 'vocabulary.json').write_text(json.dumps(vocabulary, ensure_ascii=False))
    for split, items in [('train', rows[:4]), ('validation', rows[4:6]), ('test', rows[6:])]:
        (data / (split + '.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in items))
    (data / 'summary.json').write_text(json.dumps(DATA_POLICY, ensure_ascii=False))
    shard = folder / 'train.jsonl'
    cells = [[0] * 10 for _ in range(4)]
    compact = []
    for r in rows[:4]:
        n = len(r['tokens'])
        bucket = next((i for i, maximum in enumerate([8,16,32]) if n <= maximum), 3)
        position = min(9, int((r['target_index']+1)/n*10))
        cells[bucket][position] += 1
        compact.append([''.join(r['tokens']), r['target_index'], 0, bucket, position])
    shard.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in compact))
    manifest = {**DATA_POLICY, 'format':'super-reader-sharded-training-v1', 'domains':['fixture'],
                'vocabulary_path':str(folder/'vocabulary.json'), 'shards':[{'path':str(shard),'bytes':shard.stat().st_size}],
                'statistics':{'cells':cells, 'samples':len(compact),'tokens':sum(len(r[0]) for r in compact),
                              'random_baseline_sum':sum(1/(len(r[0])-1) for r in compact),
                              'center_correct':sum(r[1]==(len(r[0])-1)//2 for r in compact),
                              'domain_samples':{'fixture':len(compact)},'maximum_sequence_length':max(len(r['tokens']) for r in rows)}}
    (folder/'manifest.json').write_text(json.dumps(manifest))
    smoke = folder / 'candidate'
    subprocess.run([sys.executable,'training/run_sharded.py','--manifest',str(folder/'manifest.json'),
                    '--data-dir',str(data),'--artifact-dir',str(smoke),'--initialize-from',str(run/'initialization.pt'),
                    '--epochs','1','--channels','192','--residual-blocks','6','--threads','2',
                    '--learning-rate','0.0003','--batch-size','4'],cwd=ROOT,check=True)
    trained = torch.load(smoke/'training-state.pt',map_location='cpu',weights_only=True)
    assert trained['progress']['epoch'] == 2
    assert trained['optimizer_state']['state']
    embedding_moments = trained['optimizer_state']['state'][0]['exp_avg']
    assert embedding_moments[vocabulary[':']].abs().sum().item() > 0
    subprocess.run([sys.executable,'training/export_browser_model.py','--artifact-dir',str(smoke),
                    '--output',str(smoke/'boundary-model-data.js')],cwd=ROOT,check=True)
    model.load_state_dict(trained['best_state'])
    model.eval()
    reference = []
    for r in rows:
        tokens = r['tokens']
        ids = torch.tensor([[vocabulary.get(t,1) for t in tokens]])
        with torch.inference_mode():
            scores = model(ids,torch.ones_like(ids,dtype=torch.bool),torch.ones((1,len(tokens)-1),dtype=torch.bool))[0].tolist()
        reference.append({'tokens':tokens,'scores':scores})
    (folder/'reference.json').write_text(json.dumps(reference,ensure_ascii=False))
    browser_js = """const fs=require('node:fs'),vm=require('node:vm');const c=vm.createContext({atob,Intl});
for(const p of [process.argv[1],'src/backend/inference.js','src/backend/chunker.js'])vm.runInContext(fs.readFileSync(p,'utf8'),c);
const ref=JSON.parse(fs.readFileSync(process.argv[2]));let error=0;
for(const row of ref){const scores=c.SuperReaderModelBackend.scoreTokens(row.tokens);
if(scores.length!==row.scores.length)throw Error('Gap count mismatch');
scores.forEach((s,i)=>{if(!Number.isFinite(s))throw Error('Nonfinite');error=Math.max(error,Math.abs(s-row.scores[i]));});
if(scores.indexOf(Math.max(...scores))!==row.scores.indexOf(Math.max(...row.scores)))throw Error('Prediction mismatch');}
if(error>0.0001)throw Error('Logit mismatch '+error);console.log(JSON.stringify({maximum_absolute_error:error,cases:ref.length}));"""
    browser = json.loads(subprocess.check_output(['node','-e',browser_js,str(smoke/'boundary-model-data.js'),str(folder/'reference.json')],cwd=ROOT))
    result = {'passed':True,'source_best_epoch':checkpoint['best_epoch'],'transfer':transfer,
              'old_embeddings_verified':len(old_vocabulary),'non_embedding_weights_exact':True,
              'new_colon_embedding_updated':True,'browser':browser}
    (run/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
