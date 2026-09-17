#!/usr/bin/env python3
"""Rebuild local sources with the current surface policy, add web data, and warm-start the CNN."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys

ROOT=Path(__file__).resolve().parent.parent
NAME='chinese-line-web-100m-16conv-v7-20260917'
BEST_CHECKPOINT=ROOT/'training/artifacts/web-mix-20m-192ch-16conv-20260915/candidate/training-state.pt'
LAST_MANIFEST=ROOT/'training/data/processed/web-mix-40m-192ch-16conv-20260916/manifest.json'
OLD_LOCAL=ROOT/'training/data/processed/unicode-context-192ch-12conv-20260913-combined/manifest.json'
OLD_EVAL=LAST_MANIFEST.parent
FILES=['training/'+name for name in (
    'run_surface_retraining.py','prepare_retraining_base.mjs','prepare_smoke_data.mjs',
    'prepare_full_wikipedia_data.mjs','prepare_synthetic_data.py','run_prepare_synthetic.py',
    'prepare_web_data.py','filter_retraining_holdouts.py','text_policy.py','text_policy.mjs',
    'text-policy.json','unicode-symbols.json','run_sharded.py','run_smoke.py','train_sharded.py','train_smoke.py','export_browser_model.py')]

def read(path): return json.loads(Path(path).read_text())
def write(path,value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def absolute(path): return (ROOT/path).resolve()

def history():
    """Every inherited training source, including older label/tokenization generations."""
    manifests={LAST_MANIFEST,OLD_LOCAL};registries=set();directories=[];seen=set();directory=OLD_EVAL
    while directory not in seen:
        seen.add(directory);directories.append(directory)
        data=read(directory/'summary.json')
        manifests.update(absolute(item['path']) for item in data.get('training_manifests',[]))
        if (directory/'holdout-document-hashes.json').exists():registries.add(directory/'holdout-document-hashes.json')
        if not data.get('source_directory'):break
        directory=absolute(data['source_directory'])
    pending=list(manifests)
    while pending:
        parent=read(pending.pop()).get('base_manifest')
        if parent and absolute(parent) not in manifests:
            manifests.add(absolute(parent));pending.append(absolute(parent))
    return sorted(manifests),sorted(registries),directories

def validated_best(state):
    """Select a completed, validated epoch rather than a later partial epoch."""
    epoch=state.get('best_epoch',0)
    metrics=next((row for row in state.get('history',[]) if row['epoch']==epoch),None)
    if not state.get('best_state') or epoch<1 or metrics is None or 'selection_score' not in metrics:
        raise ValueError('Initialization requires best_state from a completed, validated epoch')
    return state['best_state'],metrics

def freeze_initialization(run, checkpoint):
    sys.path[:0]=[str(ROOT/'training/.deps'),str(ROOT/'training')]
    import torch
    from train_smoke import BoundaryChooser, configure_cpu
    from train_sharded import expanded_initialization
    configure_cpu(2,1)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    config=state['configuration'];assert (config['channels'],config['residual_blocks'])==(192,8)
    vocabulary=state['vocabulary'];assert len(vocabulary)==8192
    best,metrics=validated_best(state)
    weights={k:v.clone() for k,v in best.items()}
    frozen={'model_state':weights,'best_state':weights,'best_epoch':state['best_epoch'],'vocabulary':vocabulary,
        'configuration':config,'selection':'best_state from a completed, validated epoch'}
    torch.save(frozen,run/'initialization.pt')
    write(run/'source-vocabulary.json',vocabulary)
    model=BoundaryChooser(len(vocabulary),192,8)
    transfer=expanded_initialization(model,run/'initialization.pt',vocabulary)
    assert all(torch.equal(value,weights[key]) for key,value in model.state_dict().items())
    # A tiny numerical check of the selected weights; probe updates are discarded.
    from text_policy import training_pairs
    pairs=list(training_pairs('天气预报来了。今天下雨，我们留在家里。\n大家一起商量。明天放晴，大家出去散步。'))
    assert pairs, 'Initialization probe requires eligible interior fragments'
    length=max(len(t) for t,_,_ in pairs)
    tokens=torch.zeros((len(pairs),length),dtype=torch.long);mask=torch.zeros_like(tokens,dtype=torch.bool)
    for i,(text,_,_) in enumerate(pairs):
        tokens[i,:len(text)]=torch.tensor([vocabulary.get(c,1) for c in text]);mask[i,:len(text)]=True
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=1e-4,foreach=True)
    assert not optimizer.state
    losses=[]
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss=torch.nn.functional.cross_entropy(model(tokens,mask,mask[:,1:]),torch.tensor([k for _,k,_ in pairs]))
        assert torch.isfinite(loss)
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append(loss.item())
    report={'passed':True,'source_checkpoint':str(checkpoint),'source_sha256':sha(checkpoint),
        'selected_state':'best_state','source_progress':state['progress'],'source_best_epoch':state['best_epoch'],
        'source_best_metrics':metrics,
        'new_optimizer':'AdamW; no inherited moments','parameters':sum(p.numel() for p in model.parameters()),
        'all_parameters_equal_before_probe':True,'probe_losses':losses,'probe_updates_discarded':True,'transfer':transfer}
    write(run/'initialization-verification.json',report)
    return report

def main():
    os.environ['DEBUG']='0'
    sys.path.insert(0,str(ROOT/'training/.deps'))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,default=ROOT/'training/artifacts'/NAME)
    parser.add_argument('--target-samples',type=int,default=100_000_000)
    parser.add_argument('--epochs',type=int,default=1)
    parser.add_argument('--initialize-from',type=Path,default=BEST_CHECKPOINT)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if min(args.target_samples,args.epochs)<1:parser.error('Counts must be positive')
    run=args.run_dir.resolve();run.mkdir(parents=True,exist_ok=True)
    lock=(run/'.run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan_path=run/'run.json';snapshot=run/'source'
    data=ROOT/'training/data/processed'
    paths={name:data/(run.name+'-'+name) for name in ['base','base-eval','wiki','local','local-eval','web','eval']}
    if not plan_path.exists():
        if shutil.disk_usage(ROOT).free<60*1024**3:raise RuntimeError('Need at least 60 GiB free for this bounded rebuild')
        os.environ['DEBUG']='0'
        initial=freeze_initialization(run,args.initialize_from.resolve())
        manifests,registries,directories=history()
        for name in FILES:
            destination=snapshot/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,destination)
        for name in ['data','.deps']:
            link=snapshot/'training'/name
            if not link.exists():link.symlink_to(ROOT/'training'/name,target_is_directory=True)
            if link.resolve()!=(ROOT/'training'/name).resolve():raise ValueError(f'Unexpected snapshot link: {link}')
        write(plan_path,{'created_at':datetime.now(timezone.utc).isoformat(),'web_train_samples':args.target_samples,'epochs':args.epochs,
            'policy':read(ROOT/'training/text-policy.json'),'initialization':initial,
            'initialization_sha256':sha(run/'initialization.pt'),'vocabulary_sha256':sha(run/'source-vocabulary.json'),
            'source_hashes':{name:sha(snapshot/name) for name in FILES},
            'historical_manifests':{str(p):sha(p) for p in manifests},
            'holdout_registries':{str(p):sha(p) for p in registries},'old_evaluation_directories':[str(p) for p in directories],
            'paths':{k:str(v) for k,v in paths.items()},'architecture':{'channels':192,'residual_blocks':8,'vocabulary':8192},
            'training':{'learning_rate':.0003,'threads':5,'batch_size':512,'max_tokens_per_batch':8192,'domain_weight_power':.65,'selection_macro_weight':.5},
            'backend_sha256':sha(ROOT/'src/boundary-model-data.js')})
    plan=read(plan_path)
    if (args.target_samples,args.epochs)!=(plan['web_train_samples'],plan['epochs']):raise ValueError('Resume with recorded settings')
    if {k:str(v) for k,v in paths.items()}!=plan['paths']:raise ValueError('Output paths changed')
    for path,digest in [(run/'initialization.pt',plan['initialization_sha256']),
                        (run/'source-vocabulary.json',plan['vocabulary_sha256'])]+[(snapshot/name,value) for name,value in plan['source_hashes'].items()]+[(Path(name),value) for name,value in plan['historical_manifests'].items()]+[(Path(name),value) for name,value in plan['holdout_registries'].items()]:
        if sha(path)!=digest:raise ValueError(f'Frozen input changed: {path}')
    completed_path=run/'completed-stages.json';completed=read(completed_path) if completed_path.exists() else {}
    child=None;stopped=False
    def status(stage,**kwargs):
        value={'stage':stage,'updated_at':datetime.now(timezone.utc).isoformat(),'coordinator_pid':os.getpid(),**kwargs}
        write(run/'status.json',value);print(json.dumps(value,ensure_ascii=False),flush=True)
    def stop(signum,_frame):
        nonlocal stopped
        if stopped:return
        stopped=True
        if child is not None and child.poll() is None:child.send_signal(signum)
    for sig in [signal.SIGINT,signal.SIGTERM]:signal.signal(sig,stop)
    env=dict(os.environ,DEBUG='0',PYTHONUNBUFFERED='1',PYTHONPATH=str(ROOT/'training/.deps'),SUPER_READER_PROJECT_ROOT=str(ROOT))
    def execute(stage,command,output=None,restart=False):
        nonlocal child
        if stage in completed:
            for name,digest in completed[stage].get('outputs',{}).items():
                if sha(Path(name))!=digest:raise ValueError(f'Completed stage output changed: {name}')
            return
        if stopped:raise InterruptedError('Stop requested')
        if shutil.disk_usage(ROOT).free<12*1024**3:raise RuntimeError('Less than 12 GiB free; preparation paused before next stage')
        if restart and output.exists():
            output.rename(output.with_name(output.name+'.interrupted-'+datetime.now().strftime('%Y%m%dT%H%M%S')))
        for name,digest in plan['source_hashes'].items():
            if sha(snapshot/name)!=digest:raise ValueError(f'Frozen code changed: {name}')
        with (run/(stage+'.log')).open('ab') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            status(stage,child_pid=child.pid,log=str(run/(stage+'.log')))
            code=child.wait()
        if stopped:raise InterruptedError('Graceful stop requested')
        if code:raise RuntimeError(f'{stage} exited {code}; inspect its log')
        if output:
            marker='manifest.json' if stage in {'prepare-wikipedia','prepare-synthetic','prepare-web'} else 'summary.json'
            if not (output/marker).exists():raise InterruptedError(f'{stage} stopped before publishing {marker}')
        if stage=='training':
            import torch
            state=torch.load(run/'candidate/training-state.pt',map_location='cpu',weights_only=True)
            if state['progress']['epoch']<=args.epochs:raise InterruptedError('Training checkpoint saved before epoch completion')
            for name in ['smoke-metrics.json','boundary-smoke.safetensors','boundary-smoke-vocabulary.json']:
                if not (run/'candidate'/name).exists():raise InterruptedError(f'Training stopped before publishing {name}')
        markers={}
        if output:
            for name in ['manifest.json','summary.json']:
                if (output/name).exists():markers[str(output/name)]=sha(output/name)
        completed[stage]={'command':command,'finished_at':datetime.now(timezone.utc).isoformat(),'outputs':markers};write(completed_path,completed)
    def script(name):return str(snapshot/'training'/name)
    def protect(source,output,manifests,old_json=False):
        cmd=[sys.executable,script('filter_retraining_holdouts.py'),'--data-dir',str(source),'--output-dir',str(output),'--input-only']
        for manifest in manifests:cmd+=['--training-manifest',str(manifest)]
        if old_json:cmd+=['--training-jsonl',str(data/'train.jsonl')]
        return cmd
    awake=subprocess.Popen(['/usr/bin/caffeinate','-is','-w',str(os.getpid())],stdin=subprocess.DEVNULL)
    try:
        base=paths['base']
        execute('prepare-base',['node','--max-old-space-size=8192',script('prepare_retraining_base.mjs'),'--output-dir',str(base),'--all-local-clue','1'],base,restart=True)
        if 'reserve-historical-documents' not in completed:
            hashes=set(read(base/'holdout-document-hashes.json'))
            for name in plan['holdout_registries']:hashes.update(read(Path(name)))
            # Known web holdouts remain reserved even if the changed labels remove every pair.
            for split in ['validation','test']:
                with (OLD_EVAL/(split+'.jsonl')).open() as f:
                    for line in f:
                        doc=json.loads(line)['document_id']
                        if doc.startswith('web:'):hashes.add(doc.removeprefix('web:'))
            write(base/'holdout-document-hashes.json',sorted(hashes))
            completed['reserve-historical-documents']={'documents':len(hashes)};write(completed_path,completed)
        execute('protect-historical-training',protect(base,paths['base-eval'],plan['historical_manifests'],True),paths['base-eval'],restart=True)
        for name in ['train.jsonl']:
            destination=paths['base-eval']/name
            if not destination.exists():os.link(base/name,destination)
        execute('prepare-wikipedia',['node','--max-old-space-size=4096',script('prepare_full_wikipedia_data.mjs'),'--source-dir',str(paths['base-eval']),'--output-dir',str(paths['wiki']),'--shards','256','--max-samples-per-document','0','--max-sequence-length','0'],paths['wiki'])
        vocabulary=paths['wiki']/'vocabulary.json'
        if not vocabulary.exists():shutil.copyfile(run/'source-vocabulary.json',vocabulary)
        execute('prepare-synthetic',[sys.executable,script('run_prepare_synthetic.py'),'--base-manifest',str(paths['wiki']/'manifest.json'),'--output-dir',str(paths['local']),'--source-manifest',str(OLD_LOCAL),'--target-samples','0','--synthetic-shards','128','--max-samples-per-document','0','--max-sequence-length','0'],paths['local'])
        execute('protect-rebuilt-training',protect(paths['base-eval'],paths['local-eval'],[paths['local']/'manifest.json']),paths['local-eval'],restart=True)
        execute('prepare-web',[sys.executable,script('prepare_web_data.py'),'--base-manifest',str(paths['local']/'manifest.json'),'--evaluation',str(paths['local-eval']),'--raw-dir',str(ROOT/'training/data/raw/ultra-fineweb-zh'),'--output-dir',str(paths['web']),'--target-samples',str(args.target_samples),'--shards','1024','--seed','20260915'],paths['web'])
        manifest=read(paths['web']/'manifest.json');assert manifest['web_source']['added_training_samples']==args.target_samples
        assert sha(paths['web']/'vocabulary.json')==plan['vocabulary_sha256']
        evaluation=paths['eval'];evaluation.mkdir(exist_ok=True)
        for name in ['validation.jsonl','test.jsonl','summary.json','holdout-document-hashes.json']:
            destination=evaluation/name
            if not destination.exists():os.link(paths['web']/name,destination)
            if sha(destination)!=sha(paths['web']/name):raise ValueError(f'Published evaluation mismatch: {name}')
        write(run/'corpus-coverage.json',{'training_samples':manifest['statistics']['samples'],'domain_samples':manifest['statistics']['domain_samples'],
            'training_tokens':manifest['statistics']['tokens'],'evaluation':read(evaluation/'summary.json')['splits'],
            'synthetic_source_exhausted':read(paths['local']/'manifest.json')['source_exhausted'],
            'wikipedia_blocks':read(paths['wiki']/'manifest.json')['blocks']})
        candidate=run/'candidate';checkpoint=candidate/'training-state.pt'
        if checkpoint.exists() and not args.resume:raise FileExistsError('Use --resume for an existing training checkpoint')
        cmd=[sys.executable,script('run_sharded.py'),'--manifest',str(paths['web']/'manifest.json'),'--data-dir',str(evaluation),'--artifact-dir',str(candidate),
            '--epochs',str(args.epochs),'--channels','192','--residual-blocks','8','--learning-rate','0.0003','--domain-weight-power','0.65','--selection-macro-weight','0.5','--gradient-clip','1.0','--batch-size','512','--max-tokens-per-batch','8192','--checkpoint-shards','4','--threads','5','--interop-threads','1']
        cmd+=['--resume'] if checkpoint.exists() else ['--initialize-from',str(run/'initialization.pt')]
        execute('training',cmd)
        execute('export',[sys.executable,script('export_browser_model.py'),'--artifact-dir',str(candidate),'--output',str(candidate/'boundary-model-data.js')])
        status('complete',training_samples=manifest['statistics']['samples'],metrics=str(candidate/'smoke-metrics.json'))
    except InterruptedError as e:status('stopped',error=str(e))
    except BaseException as e:
        status('failed',error=str(e));raise
    finally:
        awake.terminate();awake.wait()

if __name__=='__main__':main()
