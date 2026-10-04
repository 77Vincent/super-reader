"""Reproducible training-label review by nearest-edge distance; no model loaded.

Stage 1 reuses 20 previously randomly selected, hashed training shards (four
per historical block). Stage 2 samples 200 rows per distance stratum, allocating
to blocks proportional to expanded row counts. No training data is rewritten.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'training/reviews/position-quality-20261004'
OUT = ROOT / 'training/artifacts/position-quality-20261004'
INVENTORY = ROOT / 'training/artifacts/model-training-audit-20261004/structural.json'
MANIFEST = ROOT / 'training/data/processed/padding-fixed-web-350m-v7-20260925/manifest.json'
SEED = 2026100431
STRATA = ['1', '2', '3', '4', '5–6', '7–10', '11–20', '21+']
BLOCKS = [(0, 256, 'base-wiki'), (256, 384, 'synthetic'),
          (384, 1408, 'web-100m'), (1408, 2432, 'web-next-100m'),
          (2432, 3456, 'web-newest-150m')]

# These codes are entered by the reviewer after reading each complete pair;
# they are never inferred from text, length, source, or a model prediction.
REASONS = {
    'C': ('A', '分句、事件或说明之间的局部边界可接受。'),
    'T': ('A', '话题、主语或条件与后续说明之间可停顿。'),
    'L': ('A', '并列成分、字段或列举项目之间可停顿。'),
    'D': ('A', '应答、感叹、转折或提示语与陈述之间可停顿。'),
    'R': ('A', '被释词、术语与其定义或注音之间可停顿。'),
    'H': ('A', '可辨认的标题与正文或下一条目之间有边界。'),
    'F': ('U', '孤立字、残句或关系不完整，当前两片段不足以确定边界。'),
    'J': ('U', '编号、词条或排版结构不明，需要原文确认边界。'),
    'X': ('U', '抽取残留、错乱或文字损坏使局部边界难以判断。'),
    'K': ('U', '词语组合或句法关系存在歧义，无法确认当前切法。'),
    'E': ('E', '明确缺陷；必须另附逐条说明。'),
}


def annotate(start, codes, notes):
    rows = [json.loads(l) for l in (REVIEW/'sample.jsonl').open()]
    path = REVIEW/'annotations.jsonl'
    previous = [json.loads(l) for l in path.open()] if path.exists() else []
    assert len(previous) == start-1, 'Only append consecutive explicitly reviewed rows'
    assert not (REVIEW/'freeze.json').exists(), 'Annotations already frozen'
    decisions = codes.split()
    assert len(decisions) == 80, 'Each manual batch must contain exactly 80 explicit codes'
    assert start-1+len(decisions) <= len(rows)
    for offset, code in enumerate(decisions):
        row = rows[start-1+offset]
        verdict, reason = REASONS[code]
        override = notes.get(str(start+offset))
        if verdict == 'E':
            assert override, row['review_id']
        previous.append(dict(review_id=row['review_id'], verdict=verdict, reason_code=code,
                             reason=override or reason, reviewer='Codex AI first pass'))
    path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in previous))
    print(f'Saved explicit decisions for {start}–{len(previous)}; total {len(previous)}/1600')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_once(path, payload):
    if path.exists():
        assert path.read_text() == payload, f'Refusing to overwrite different frozen input: {path}'
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload)


def stratum(d):
    return next(i for i, limit in enumerate([1, 2, 3, 4, 6, 10, 20, math.inf]) if d <= limit)


def relative_bin(cut, n):
    d = min(cut, n-cut)
    return next(i for i, boundary in enumerate([10, 20, 30, 40, 50]) if 100*d <= boundary*n)


def sample():
    inventory = json.loads(INVENTORY.read_text())
    assert digest(MANIFEST) == inventory['manifest_sha256']
    manifest = json.loads(MANIFEST.read_text())
    files = [f for f in inventory['files'] if f['split'] == 0]
    assert len(files) == 20
    counts = collections.Counter()
    profile = collections.Counter()
    pools = collections.defaultdict(list)
    rngs = {(b, s): random.Random(SEED + b*100 + s) for b in range(5) for s in range(8)}
    for f in files:
        path = ROOT / f['path']
        assert path.stat().st_size == f['bytes'] and digest(path) == f['sha256']
        b = next(i for i, (lo, hi, _) in enumerate(BLOCKS) if lo <= f['shard'] < hi)
        line_no = 0
        for line_no, line in enumerate(path.open(), 1):
            row = json.loads(line)
            text, target, domain = row[:3]
            cut, n = target+1, len(text)
            assert 0 < cut < n
            d = min(cut, n-cut)
            s = stratum(d)
            key = b, s
            counts[key] += 1
            rb = relative_bin(cut, n)
            length = next(i for i, mx in enumerate([16, 32, 64, 128, math.inf]) if n <= mx)
            profile[b, s, rb, length, domain] += 1
            k = counts[key]
            j = k-1 if k <= 200 else rngs[key].randrange(k)
            if j < 200:
                chosen = dict(shard=f['shard'], path=f['path'], line_1based=line_no,
                              row_sha256=hashlib.sha256(line.encode()).hexdigest(),
                              text=text, cut=cut, length=n, distance=d, stratum=STRATA[s],
                              stratum_index=s, relative_distance=d/n, relative_bin=rb,
                              length_bin=length, block=BLOCKS[b][2], block_index=b,
                              domain=manifest['domains'][domain])
                if k <= 200:
                    pools[key].append(chosen)
                else:
                    pools[key][j] = chosen
        assert line_no == f['rows']
    selected, cells = [], []
    for s in range(8):
        mass = [counts[b, s]*(hi-lo)/4 for b, (lo, hi, _) in enumerate(BLOCKS)]
        ideal = [200*v/sum(mass) for v in mass]
        quota = [math.floor(v) for v in ideal]
        for b in sorted(range(5), key=lambda b: (-(ideal[b]-quota[b]), b))[:200-sum(quota)]:
            quota[b] += 1
        assert all(quota[b] > 0 or counts[b,s] == 0 for b in range(5)), 'Need every nonempty cell represented'
        for b in range(5):
            rngs[b, s].shuffle(pools[b, s])
            cell = dict(block=BLOCKS[b][2], block_index=b, stratum=STRATA[s], stratum_index=s,
                        pool_rows=counts[b,s], first_stage_expansion=(BLOCKS[b][1]-BLOCKS[b][0])/4,
                        expanded_rows=mass[b], n=quota[b])
            cells.append(cell)
            selected.extend(dict(r, weight=mass[b]/quota[b]) for r in pools[b,s][:quota[b]])
    random.Random(SEED+10000).shuffle(selected)
    rows = [dict(review_id=f'P{i:04d}', **r) for i, r in enumerate(selected, 1)]
    assert len(rows) == 1600
    assert len({(r['path'], r['line_1based']) for r in rows}) == 1600
    write_once(REVIEW / 'sample.jsonl', ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    protocol = dict(
        seed=SEED, manifest=str(MANIFEST.relative_to(ROOT)), manifest_sha256=digest(MANIFEST),
        sample_sha256=digest(REVIEW/'sample.jsonl'), standard=manifest['standard'],
        files=files, pool_rows=sum(counts.values()), sample_n=1600, cells=cells,
        population_profile=[dict(block_index=b, stratum_index=s, relative_bin=q, length_bin=l,
                                 domain=manifest['domains'][dm], n=n)
                            for (b,s,q,l,dm), n in sorted(profile.items())],
        selection='First stage: reuse 4 random whole shards per each of 5 historical blocks. Second stage: reservoir sample within block x distance, allocate 200/stratum by largest remainder of expanded row mass, shuffle reservoir, take allocated prefix. Review order randomly mixes strata. No predictions loaded.',
        weight='(number of block shards / 4) * observed block-distance rows / reviewed block-distance rows. Estimates conditional on selected shards; no full-corpus representativeness guarantee.',
        distance='min(cut, length-cut); cut=target_index+1 characters on the left. Absolute bins 1,2,3,4,5-6,7-10,11-20,21+. Relative bins (0,.1],(.1,.2],(.2,.3],(.3,.4],(.4,.5]; .5 is exact center.',
        verdicts={'A':'Marked boundary locally defensible for reading; need not be the only good cut.',
                  'E':'Clear lexical, named-entity, or tight-structure break visible in prepared input.',
                  'U':'Unresolved boundary / insufficient context / damaged extraction; not automatically wrong.'},
        review='Codex AI reads every selected normalized pair without model predictions. Single-reviewer first pass, not independent human gold. No original document or punctuation retained in compact training rows; source traced to shard and line only.',
        rubric=['Assess the marked boundary, not factual truth, sentence length, or model match.',
                'Single-character interjections, independent pronouns, gender fields, dictionary headword/definition boundaries can be A.',
                'Ambiguous isolated numeral/character, layout-dependent lists, or broken context are U unless local relation is clear.',
                'Keep multiple acceptable cuts inside A; this is not an exhaustive multiple-answer audit.',
                'No automatic semantic labels from regex; v8 escape rules assessed separately after review.',
                'Historical v7 data remain unchanged; current v8 filter has not been applied to this corpus.'],
        limitations=['Unequal cell sampling; raw pooled 1600 percentages are not corpus rates.',
                     'Four shards per block cannot establish a precise population noise rate.',
                     'No document IDs in compact train data: document clustering cannot be directly measured.',
                     'No test data or model inference; position associations do not establish causation.'])
    write_once(REVIEW/'protocol.json', json.dumps(protocol, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'pool_rows':sum(counts.values()), 'sample_n':len(rows),
                      'cells':cells, 'mean_length':sum(r['length'] for r in rows)/len(rows)}, ensure_ascii=False))


def freeze():
    rows = [json.loads(l) for l in (REVIEW/'sample.jsonl').open()]
    labels = [json.loads(l) for l in (REVIEW/'annotations.jsonl').open()]
    assert len(rows) == len(labels) == 1600
    assert all(r['review_id'] == a['review_id'] for r,a in zip(rows,labels))
    path = REVIEW/'freeze.json'
    assert not path.exists(), 'Already frozen'
    save(REVIEW/'reason-codes.json', REASONS)
    save(path, dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         **{name+'_sha256':digest(REVIEW/(name+ext)) for name,ext in
            [('sample','.jsonl'),('annotations','.jsonl'),('protocol','.json'),('reason-codes','.json')]},
         qa='Before freeze: reread all E and U decisions, all R codes, plus 80 randomly selected A rows (seed 2026100432). No predictions loaded; same reviewer, not independent adjudication. One missing transcription code corrected in batch 1121–1200; see review-log.json.'))


def wilson(k, n):
    if not n:
        return None, None
    z = 1.959963984540054
    den = 1+z*z/n
    center = (k/n + z*z/(2*n))/den
    half = z*math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/den
    return max(0, center-half), min(1, center+half)


def stats(rows, name):
    n = len(rows)
    total = sum(r['weight'] for r in rows)
    counts = collections.Counter(r['verdict'] for r in rows)
    result = dict(group=name, n=n, expanded_mass=total,
                  effective_n=total**2/sum(r['weight']**2 for r in rows) if n else 0,
                  mean_length=sum(r['weight']*r['length'] for r in rows)/total if n else None,
                  shards=len({r['shard'] for r in rows}))
    for v in ['A', 'E', 'U']:
        result[v] = counts[v]
        result[v+'_raw'] = counts[v]/n if n else None
        result[v+'_weighted'] = sum(r['weight'] for r in rows if r['verdict']==v)/total if n else None
        result[v+'_iid_wilson_low'], result[v+'_iid_wilson_high'] = wilson(counts[v], n)
    return result


def analyze():
    frozen = json.loads((REVIEW/'freeze.json').read_text())
    for name, ext in [('sample','.jsonl'),('annotations','.jsonl'),('protocol','.json'),('reason-codes','.json')]:
        assert digest(REVIEW/(name+ext)) == frozen[name+'_sha256'], name
    samples = [json.loads(l) for l in (REVIEW/'sample.jsonl').open()]
    annotations = [json.loads(l) for l in (REVIEW/'annotations.jsonl').open()]
    protocol = json.loads((REVIEW/'protocol.json').read_text())
    assert len(samples) == len(annotations) == 1600
    by_id = {a['review_id']:a for a in annotations}
    assert len(by_id)==1600 and by_id.keys()=={r['review_id'] for r in samples}
    rows = []
    policy = json.loads((ROOT/'training/text-policy.json').read_text())
    # Parse only the three named escape rules; unrelated filters are not a
    # substitute for the semantic judgments above.
    def find_rules(obj):
        found = {}
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in ['literal_line_escape','literal_control_escape','literal_unicode_residue']:
                    found[key] = value
                elif isinstance(value, (dict,list)):
                    found.update(find_rules(value))
        elif isinstance(obj,list):
            for value in obj:
                found.update(find_rules(value))
        return found
    rules = {k:re.compile(v) for k,v in find_rules(policy).items()}
    assert len(rules)==3
    for r in samples:
        a=by_id[r['review_id']]
        assert REASONS[a['reason_code']][0] == a['verdict'] and a['reason']
        t,c=r['text'],r['cut']
        assert 0<c<len(t) and min(c,len(t)-c)==r['distance']
        assert STRATA[stratum(r['distance'])]==r['stratum']
        assert relative_bin(c,len(t))==r['relative_bin']
        assert r['weight'] > 0
        side='left' if c<len(t)-c else 'right' if c>len(t)-c else 'symmetric'
        hits=[k for k,rx in rules.items() if rx.search(t[:c]) or rx.search(t[c:])]
        rows.append(dict(r, **a, marked_text=t[:c]+'｜'+t[c:], side=side,
                         literal_escape_hits=';'.join(hits)))
    cells = {(r['block_index'],r['stratum_index']):r for r in protocol['cells']}
    for (b,s), cell in cells.items():
        sample=[r for r in rows if r['block_index']==b and r['stratum_index']==s]
        assert len(sample)==cell['n']
        assert math.isclose(sum(r['weight'] for r in sample),cell['expanded_rows'],rel_tol=1e-12)
    absolute=[dict(order=s, **stats([r for r in rows if r['stratum_index']==s],name)) for s,name in enumerate(STRATA)]
    assert all(g['n']==200 for g in absolute)
    relative=[]
    for minimum in [2,16]:
        for q, name in enumerate(['0–10%','10–20%','20–30%','30–40%','40–50%（中间）']):
            relative.append(dict(order=q, minimum_length=minimum,
                **stats([r for r in rows if r['length']>=minimum and r['relative_bin']==q],name)))
    sides=[]
    for s,name in enumerate(STRATA):
        for side in ['left','right','symmetric']:
            sides.append(dict(stratum=name, **stats([r for r in rows if r['stratum_index']==s and r['side']==side],side)))
    sources=[stats([r for r in rows if r['block']==name],name) for _,_,name in BLOCKS]
    robustness=[]
    for scope,predicate in [('web only',lambda r:r['block_index']>=2),
                            ('web; 16–64 chars',lambda r:r['block_index']>=2 and 16<=r['length']<=64),
                            ('web; 16–32 chars',lambda r:r['block_index']>=2 and 16<=r['length']<=32),
                            ('all; <=256 chars',lambda r:r['length']<=256)]:
        for s,name in enumerate(STRATA):
            robustness.append(dict(scope=scope, order=s, **stats([r for r in rows if r['stratum_index']==s and predicate(r)],name)))
    reasons=[]
    for s,name in enumerate(STRATA):
        for code,(v,reason) in REASONS.items():
            subset=[r for r in rows if r['stratum_index']==s and r['reason_code']==code]
            reasons.append(dict(stratum=name,code=code,verdict=v,n=len(subset),reason=reason))
    # Descriptive sensitivity: remove each shard's reviewed rows in turn using
    # the fixed original design weights. This is not a confidence interval.
    influence=[]
    for s,name in enumerate(STRATA):
        sub=[r for r in rows if r['stratum_index']==s]
        estimates=[stats([r for r in sub if r['shard']!=sh],name) for sh in {r['shard'] for r in sub}]
        influence.append(dict(group=name, U_min=min(x['U_weighted'] for x in estimates),
                              U_max=max(x['U_weighted'] for x in estimates),
                              note='Fixed original weights; leave-one-reviewed-shard sensitivity, not a CI.'))
    findings=dict(freeze=frozen, pool_rows=protocol['pool_rows'],review_n=1600,
        absolute=absolute,relative=relative,sides=sides,sources=sources,
        robustness=robustness,reason_counts=reasons,shard_influence=influence,
        overall=stats(rows,'all'),literal_escape_matches=[r['review_id'] for r in rows if r['literal_escape_hits']],
        exact_duplicate_text_cut=1600-len({(r['text'],r['cut']) for r in rows}),
        total_characters=sum(r['length'] for r in rows),
        checks={'frozen_hashes_match':True,'all_1600_explicit_annotations':True,
                'all_sample_locations_distinct':len({(r['path'],r['line_1based']) for r in rows})==1600,
                'all_40_sampling_cell_weights_reconcile':True,'all_8_strata_have_200':True,
                'no_model_predictions_or_test_loaded':True},
        uncertainty='Wilson intervals shown only for unweighted within-stratum counts, as IID reference bounds conditional on AI labels. They do not account for clustered source documents, first-stage shard sampling, or reviewer disagreement. Zero observed E is not zero true error. Relative-bin weighted rates require weights; effective_n is Kish weight-balance information only, not independent documents.',
        exclusions='No new cleanup was applied to historical prepared v7 rows. A means the marked cut is locally defensible, not that the whole source is clean, factually correct, or has a unique acceptable cut.')
    save(REVIEW/'findings.json',findings)
    save(OUT/'findings.json',findings)
    save(OUT/'reviewed.json',rows)
    for name,fields in [('reviewed.csv',['review_id','stratum','distance','relative_distance','length','side','block','domain','marked_text','verdict','reason','weight','path','line_1based','row_sha256']),
                        ('human-review.csv',['review_id','marked_text','human_verdict','human_reason','path','line_1based'])]:
        with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    print(json.dumps({'absolute':[{k:r[k] for k in ['group','n','A','E','U','U_weighted']} for r in absolute],
                      'relative':[{k:r[k] for k in ['group','minimum_length','n','effective_n','E_weighted','U_weighted']} for r in relative],
                      'single_sides':[r for r in sides if r['stratum']=='1'],
                      'literal_escape_matches':findings['literal_escape_matches']},ensure_ascii=False,indent=2))
    return findings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--annotate', type=int)
    parser.add_argument('--codes')
    parser.add_argument('--notes', default='{}')
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--analyze', action='store_true')
    args = parser.parse_args()
    if args.sample:
        sample()
    if args.annotate:
        annotate(args.annotate, args.codes, json.loads(args.notes))
    if args.freeze:
        freeze()
    if args.analyze:
        analyze()
