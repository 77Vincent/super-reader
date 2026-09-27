"""Read-only structural audit of fixed holdouts and 32 seeded training shards.

Does not run training or rewrite prepared data. Duplicate rates on the sampled
training shards are not estimates of full-corpus duplicate/conflict rates.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'training/artifacts/data-preparation-audit-v7-20260925'
MANIFEST = ROOT/'training/artifacts/learning-rate-192ch-full-246m-v7-20260923/manifest.json'
METRICS = MANIFEST.parent/'epoch-1-backend/smoke-metrics.json'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = json.loads(MANIFEST.read_text()); metrics = json.loads(METRICS.read_text())
    norm = metrics['training_weighting']['combined_weight_sample_mean_before_normalization']
    dw = metrics['training_weighting']['domain_weights']
    pw = [[sum(row)/sum(n>0 for n in row)/n if n else 0 for n in row] for row in m['statistics']['cells']]
    rng = random.Random(2026092501)
    chosen = sorted(sum([rng.sample(range(a,b),8) for a,b in [(0,256),(256,384),(384,1408),(1408,2432)]],[]))
    files = [(ROOT/m['shards'][i]['path'],0,i) for i in chosen]
    files += [(Path(m['evaluation_source_dir'])/(s+'.jsonl'),code,None) for s,code in [('validation',1),('test',2)]]
    seen = {}; duplicates = {}; duplicate_texts = {}; inventory = []
    stats = [Counter() for _ in range(3)]
    domains = [defaultdict(Counter) for _ in range(3)]
    lengths = [Counter() for _ in range(3)]
    positions = [[0]*10 for _ in range(3)]
    weighted_positions = [0.0]*10
    batch_domains = defaultdict(float); batch_lengths = defaultdict(float)
    nominal_domains = defaultdict(float); nominal_lengths = defaultdict(float)
    epoch_batches = Counter(); max_weight_examples = []
    started = time.monotonic()
    for path,split,shard in files:
        digest = hashlib.sha256(); nrows = 0; batches = defaultdict(list)
        if shard is not None:
            assert path.stat().st_size == m['shards'][shard]['bytes']
        with path.open('rb') as f:
            for line in f:
                digest.update(line); r = json.loads(line)
                if split==0:
                    text,target,domain,bucket,position = r
                    n=len(text); expected_bucket=next((i for i,v in enumerate([8,16,32]) if n<=v),3)
                    expected_position=min(9,int((target+1)/n*10))
                    if (bucket,position)!=(expected_bucket,expected_position):stats[split]['bucket_or_position_mismatch']+=1
                    name=m['domains'][domain]; weight=pw[bucket][position]*dw[name]
                    nominal_domains[name]+=weight; nominal_lengths[1<<(n-1).bit_length()]+=weight
                    weighted_positions[position]+=weight
                    batches[1<<(n-1).bit_length()].append((name,weight))
                    if weight>1000 and len(max_weight_examples)<20:
                        max_weight_examples.append({'text':text,'target':target,'domain':name,'weight':weight,'shard':shard})
                else:
                    tokens=r['tokens']; text=''.join(tokens); n=len(text); target=r['target_index'];name=r['domain']
                    position=min(9,int((target+1)/n*10))
                    if len(tokens)!=n or any(len(t)!=1 for t in tokens):stats[split]['non_character_tokens']+=1
                    if not r.get('punctuation') or any(c not in m['proxy_punctuation'] for c in r['punctuation']):stats[split]['invalid_proxy']+=1
                if not isinstance(target,int) or not 0<=target<n-1:stats[split]['invalid_target']+=1
                if unicodedata.normalize('NFKC',text)!=text:stats[split]['not_nfkc']+=1
                if text!=text.strip() or '  ' in text:stats[split]['whitespace_contract_mismatch']+=1
                key=hashlib.sha256(text.encode()).digest(); packed=(target<<2)|split
                previous=seen.get(key)
                if previous is None:seen[key]=packed
                else:
                    if key not in duplicates:duplicates[key]=Counter({previous:1})
                    duplicates[key][packed]+=1
                    if len(duplicate_texts)<100 or key in duplicate_texts:duplicate_texts[key]=text
                stats[split]['rows']+=1; stats[split]['tokens']+=n
                stats[split]['length_above_256']+=n>256
                domains[split][name]['rows']+=1; domains[split][name]['tokens']+=n
                lengths[split][1<<(n-1).bit_length()]+=1;positions[split][position]+=1
                nrows+=1
        if split==0:
            randomizer=random.Random(metrics['seed']+1000+shard)
            for ceiling, items in sorted(batches.items()):
                randomizer.shuffle(items)
                batch_size=max(1,min(512,8192//ceiling))
                for start in range(0,len(items),batch_size):
                    selected=items[start:start+batch_size];epoch_batches[ceiling]+=1
                    for name,weight in selected:
                        coefficient=weight/len(selected)/norm
                        batch_domains[name]+=coefficient;batch_lengths[ceiling]+=coefficient
        inventory.append({'path':str(path.relative_to(ROOT)),'split':split,'shard':shard,'rows':nrows,'bytes':path.stat().st_size,'sha256':digest.hexdigest()})
        print(f"files={len(inventory)}/{len(files)} rows={sum(s['rows'] for s in stats)} seconds={time.monotonic()-started:.1f}",flush=True)
    duplicate_summary=Counter();examples=[];floor=Counter()
    for key,counts in duplicates.items():
        splits={p&3 for p in counts};targets={p>>2 for p in counts}
        duplicate_summary['repeated_inputs']+=1
        if len(targets)>1:duplicate_summary['conflicting_inputs']+=1
        if len(splits)>1:
            duplicate_summary['cross_split_inputs']+=1
            if len(targets)>1:duplicate_summary['cross_split_conflicting_inputs']+=1
            if 0 in splits:duplicate_summary['sampled_train_holdout_inputs']+=1
            if {1,2}<=splits:duplicate_summary['validation_test_inputs']+=1
        for split in splits:
            label_counts=[n for p,n in counts.items() if p&3==split]
            floor[split]+=sum(label_counts)-max(label_counts)
        if len(targets)>1 and key in duplicate_texts and len(examples)<30:
            text=duplicate_texts[key]
            examples.append({'text':text,'rows':[{'split':['sampled_train','validation','test'][p&3],
                           'target':p>>2,'split_text':text[:(p>>2)+1]+'｜'+text[(p>>2)+1:],'count':n} for p,n in counts.items()]})
    result={'scope':'All validation/test rows and 8 seeded whole shards from each of four training blocks; train-overlap counts are lower bounds for this subset only',
            'manifest_sha256':hashlib.sha256(MANIFEST.read_bytes()).hexdigest(), 'selected_training_shards':chosen,
            'files':inventory,'stats':stats,'domains':domains,'length_ceilings':lengths,'positions':positions,
            'nominal_weighted_positions':weighted_positions,'position_weight_table':pw,'domain_weights':dw,
            'nominal_weight_mass_by_domain':nominal_domains,'nominal_weight_mass_by_length':nominal_lengths,
            'actual_batch_coefficient_mass_by_domain':batch_domains,'actual_batch_coefficient_mass_by_length':batch_lengths,
            'batch_counts_by_length':epoch_batches,
            'batch_coefficient_definition':'Sum of weight / actual batch example count / global mean, before clipping and Adam; NOT actual optimizer influence',
            'duplicates':duplicate_summary,'input_conflict_minimum_misses_by_split':floor,
            'duplicate_examples':examples,'high_weight_examples':max_weight_examples,
            'seconds':time.monotonic()-started}
    (OUT/'structural.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'stats':stats,'duplicates':duplicate_summary,'minimum_conflict_misses':floor,'seconds':result['seconds']},ensure_ascii=False))

if __name__=='__main__':main()
