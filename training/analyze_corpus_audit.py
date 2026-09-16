"""Summarize the frozen semantic review and bounded automatic corpus probes."""
from __future__ import annotations
import collections
import csv
import difflib
import hashlib
import json
import re
import statistics
import sys
import unicodedata
from pathlib import Path

from audit_corpus import ROOT, OUT, MANIFEST, GROUPS, containing_row


def analyze():
    rows = [json.loads(line) for line in (OUT / 'sample.jsonl').open()]
    labels = {r['id']: r for r in map(json.loads, (OUT / 'annotations.jsonl').open())}
    provenance = {r['id']: r for r in map(json.loads, (OUT / 'web-provenance.jsonl').open())}
    assert len(rows) == len({r['id'] for r in rows}) == 20000
    assert len(labels) == 1000
    assert set(labels) == {r['id'] for r in rows if r['review_sample']}
    population = sum(r[3] for r in GROUPS)
    manifest = json.loads(MANIFEST.read_text())
    assert population == manifest['statistics']['samples']
    # Verify all sampled rows still point to the exact immutable training bytes.
    by_path = collections.defaultdict(list)
    for row in rows:
        by_path[row['path']].append(row)
    for path, group_rows in by_path.items():
        with (ROOT / path).open('rb') as handle:
            for row in group_rows:
                handle.seek(row['offset'])
                assert hashlib.sha256(handle.readline()).hexdigest() == row['row_sha256']
    # Exhaustively verify the byte-to-containing-row mapping on mixed-length UTF-8.
    import io
    sample_lines = [b'["ab",0,0,0,0]\n', '["汉字",0,0,0,0]\n'.encode(), b'["'+b'x'*5000+b'",2,0,0,0]\n']
    buffer = io.BytesIO(b''.join(sample_lines))
    counts = collections.Counter(containing_row(buffer, pos)[0] for pos in range(len(buffer.getvalue())))
    assert list(counts.values()) == [len(line) for line in sample_lines]
    han = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002fa1f\U00030000-\U000323af]')
    profile, reviews = [], []
    weighted = collections.Counter()
    error_reasons = collections.Counter()
    source_reviews = collections.defaultdict(list)
    for name, _, _, count in GROUPS:
        group_rows = [r for r in rows if r['stratum'] == name]
        selected = [r for r in group_rows if r['review_sample']]
        audit = collections.Counter(labels[r['id']]['verdict'] for r in selected)
        quality = collections.Counter(labels[r['id']]['input_quality'] for r in selected)
        risks = collections.Counter()
        for row in group_rows:
            text = row['text']; gap = row['target_index'] + 1
            detected = set(row['flags'])
            if re.search(r'\$P\$|\(I_M_G\)|&nbsp|\ufffd|-\{[^}]*\}-', text):
                detected.add('extraction_marker')
            if not (han.fullmatch(text[gap-1]) and han.fullmatch(text[gap])):
                detected.add('non_han_adjacent_target')
            if not han.search(text):
                detected.add('no_han_input')
            for flag in detected:
                risks[flag] += 1
        share = count / population
        result = {'stratum':name, 'population':count, 'population_share':share, 'review_n':len(selected),
                  'acceptable':audit['A'], 'clear_error':audit['E'], 'uncertain':audit['U'],
                  'error_rate':audit['E']/len(selected), 'uncertain_rate':audit['U']/len(selected),
                  'severe_input':quality['severe'], 'minor_input':quality['minor'],
                  'severe_input_rate':quality['severe']/len(selected), 'automated_n':len(group_rows),
                  'automatic_flags':dict(risks), 'sampled_domain_counts':dict(collections.Counter(r['domain'] for r in selected)),
                  'mean_characters':statistics.mean(len(r['text']) for r in group_rows)}
        profile.append(result)
        for category in ['A','E','U']:
            weighted[category] += share*audit[category]/len(selected)
        weighted['severe_input'] += share*quality['severe']/len(selected)
        weighted['minor_input'] += share*quality['minor']/len(selected)
        for row in selected:
            label = labels[row['id']]; raw = provenance.get(row['id'], {})
            text = row['text']; gap = row['target_index']+1
            entry = {**row, **label, 'left':text[:gap], 'right':text[gap:],
                     'source':raw.get('source',row['domain']), 'punctuation':raw.get('punctuation'),
                     'upstream_score':float(raw['score']) if raw.get('score') is not None else None, 'context':raw.get('context'),
                     'document_sha256':raw.get('document_sha256'),
                     'han_adjacent':bool(han.fullmatch(text[gap-1]) and han.fullmatch(text[gap]))}
            reviews.append(entry)
            if row['stratum'].startswith('web'):
                source_reviews[entry['source']].append(entry)
            if label['verdict']=='E':
                reason = label['reason']
                category = ('structural_numbering' if any(s in reason for s in ['编号','序号','题号']) else
                            'closing_quote' if '闭引号' in reason else
                            'coordinate' if '坐标' in reason else
                            'name_or_initial' if any(s in reason for s in ['人名','专名','首字母']) else 'opening_bracket')
                error_reasons[category] += 1
    # Stratified nonparametric row bootstrap; covers sampling variation, not reviewer error.
    sys.path.insert(0, str(ROOT / 'training/.deps'))
    import numpy as np
    rng = np.random.default_rng(2026091602)
    bootstrap = np.zeros(50000)
    for p in profile:
        bootstrap += p['population_share'] * rng.binomial(p['review_n'], p['error_rate'], 50000)/p['review_n']
    interval = [float(x) for x in np.quantile(bootstrap, [.025,.975])]
    # Bounded near-duplicate probe: normalize numbers/spacing; rare shared 3-grams
    # generate candidates, followed by >=.85 Jaccard and >=.90 sequence similarity.
    def canonical(text):
        text = unicodedata.normalize('NFKC',text).lower()
        text = re.sub(r'\d+(?:[.,]\d+)*','#',text)
        return ''.join(c for c in text if c.isalnum() or c=='#')
    normalized = [canonical(r['text']) for r in rows]
    exact = collections.defaultdict(list)
    postings = collections.defaultdict(list)
    shingles = []
    for i,text in enumerate(normalized):
        exact[text].append(i)
        grams = {text[j:j+3] for j in range(len(text)-2)} if len(text)>=12 else set()
        shingles.append(grams)
        for gram in grams:
            postings[gram].append(i)
    candidates = set()
    for i,grams in enumerate(shingles):
        eligible = sorted((g for g in grams if 1<len(postings[g])<=40),key=lambda g:(len(postings[g]),g))[:6]
        for gram in eligible:
            for j in postings[gram]:
                if j>i and min(len(normalized[i]),len(normalized[j]))>=.8*max(len(normalized[i]),len(normalized[j])):
                    candidates.add((i,j))
    near = []
    for i,j in sorted(candidates):
        a,b=shingles[i],shingles[j]
        similarity=len(a&b)/len(a|b)
        if similarity>=.85 and difflib.SequenceMatcher(None,normalized[i],normalized[j],autojunk=False).ratio()>=.90:
            near.append({'a':rows[i]['id'],'b':rows[j]['id'],'similarity':similarity,
                         'text_a':rows[i]['text'],'text_b':rows[j]['text']})
    families = [[rows[i]['id'] for i in ids] for key,ids in exact.items() if len(ids)>1 and len(key)>=12]
    source_summary=[]
    for source,rr in sorted(source_reviews.items()):
        source_summary.append({'source':source,'n':len(rr),'clear_error':sum(r['verdict']=='E' for r in rr),
                               'uncertain':sum(r['verdict']=='U' for r in rr),'severe_input':sum(r['input_quality']=='severe' for r in rr),
                               'median_score':statistics.median(r['upstream_score'] for r in rr)})
    findings={'population':population,'automated_sample':len(rows),'semantic_review':len(reviews),
              'strata':profile,'weighted_rates':dict(weighted),'error_rate_bootstrap_95':interval,
              'error_categories':dict(error_reasons),'raw_error_count':sum(r['verdict']=='E' for r in reviews),
              'raw_uncertain_count':sum(r['verdict']=='U' for r in reviews),
              'errors_han_adjacent':sum(r['verdict']=='E' and r['han_adjacent'] for r in reviews),
              'web_sources':source_summary,'web_review_unique_documents':len({r['document_sha256'] for r in reviews if r['document_sha256']}),
              'near_duplicate_probe':{'candidate_pairs':len(candidates),'near_pairs':len(near),'number_normalized_exact_families':len(families),
                                      'scope':'20,000 uniform training rows; rare-gram candidate heuristic, not exhaustive whole-corpus deduplication'},
              'audit':{'sampled_bytes_verified':len(rows),'annotated_ids_match_review_sample':True,
                       'byte_sampler_mapping_test_passed':True,'all_web_review_rows_recovered':all(r['document_sha256'] for r in reviews if r['stratum'].startswith('web'))},
              'limitation':'Single AI reviewer; local-boundary judgments, not human consensus or factual audit. Population weighting is by row count, not domain/position training-loss weights.'}
    (OUT/'findings.json').write_text(json.dumps(findings,ensure_ascii=False,indent=2)+'\n')
    (OUT/'reviewed.json').write_text(json.dumps(reviews,ensure_ascii=False,indent=2)+'\n')
    (OUT/'near-duplicates.json').write_text(json.dumps({'near':near,'normalized_families':families},ensure_ascii=False,indent=2)+'\n')
    with (OUT/'reviewed.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        fields=['id','stratum','domain','source','left','right','verdict','input_quality','reason','punctuation','upstream_score','han_adjacent','context','path','offset']
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(reviews)
    return findings


if __name__=='__main__':
    result=analyze()
    print(json.dumps({k:v for k,v in result.items() if k not in ['strata','web_sources']},ensure_ascii=False,indent=2))
    for row in result['strata']:
        print(json.dumps(row,ensure_ascii=False))
    print(json.dumps(result['web_sources'],ensure_ascii=False))
