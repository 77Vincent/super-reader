"""Recompute the frozen score study and its executed notebook; never trains."""
from pathlib import Path
import ast
import contextlib
import datetime
import io
import json
import sqlite3

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'training/artifacts/ultra-score-review-20261004'
OUT.mkdir(parents=True, exist_ok=True)
cells = []
namespace = {'__name__': '__main__'}
execution_count = 0


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def md(value):
    cells.append(dict(id=f'cell-{len(cells)}', cell_type='markdown', metadata={},
                      source=value.splitlines(keepends=True)))


def code(value):
    global execution_count
    execution_count += 1
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(value, f'score-review-cell-{execution_count}', 'exec'), namespace)
    cells.append(dict(id=f'cell-{len(cells)}', cell_type='code', metadata={},
                      source=value.splitlines(keepends=True), execution_count=execution_count,
                      outputs=[dict(output_type='stream', name='stdout',
                                    text=stdout.getvalue().splitlines(keepends=True))]))


md('# Ultra-FineWeb Boundary Quality Review\n\n## tl;dr\n\n唯一质量指标是目标断点是否合理。复用冻结的 640 条盲审记录：637 条判为合理，0 条明确不合理，3 条不确定。目前无法区分不同分数段的断点质量，撤回优先选择 0.99 的建议。不修改原始标注，不把文字瑕疵或文体计作断点错误，没有新增样本或训练。')
md('## Context & Methods\n\n全量分数普查与分层抽样分开。640 条语义判断在隐藏分数时完成，44 条初始异常／不确定项及随机 80 条 A 类复核后冻结。AI 初审，不是人工金标准。样本抽取执行当前 v8 清理与最短边至少两字。\n\n### Key Assumptions\n\n每篇符合条件的训练文档随机取一个切分题，因此语义率是文档等权口径，不是训练行等权；不跨分数段直接平均。文档等权产出与训练行标点分布分别计算。事实真伪未评估。\n\nbuild.py 顺序执行下列单元并捕获输出，未使用 Jupyter 内核。依赖 Python 3 和已有 training/.deps；运行时从仓库根目录开始。')
code("""from pathlib import Path
import collections, hashlib, json, math, re, statistics, sys, unicodedata
root = Path.cwd()
while not (root/'training/reviews/ultra-score-review-20261004').exists():
    if root.parent == root: raise RuntimeError('Run inside repository')
    root = root.parent
review = root/'training/reviews/ultra-score-review-20261004'
out = root/'training/artifacts/ultra-score-review-20261004'
sys.path[:0] = [str(root/'training/.deps'), str(root/'training')]
from prepare_synthetic_data import clean_document, is_han
def read(name): return json.loads((review/name).read_text())
freeze = read('freeze.json')
for name, expected in freeze['sha256'].items():
    assert hashlib.sha256((review/name).read_bytes()).hexdigest() == expected, name
for entry in read('evidence/index.json'):
    assert hashlib.sha256((review/'evidence'/entry['file']).read_bytes()).hexdigest() == entry['sha256']
profile, protocol = read('profile.json'), read('protocol.json')
samples, annotations = read('sample.json'), read('annotations.json')
documents_path = out/'documents.jsonl'
documents = [json.loads(line) for line in documents_path.open()]
assert len(documents) == 4800 and len(samples) == len(annotations) == 640
assert len({r['document_id'] for r in samples}) == 640
docs_by_location = {(d['raw_file'], d['group'], d['row']):d for d in documents}
joined = []
for s,a in zip(samples,annotations):
    assert s['review_id'] == a['review_id']
    d = docs_by_location[s['raw_file'],s['group'],s['row']]
    assert d['split']=='train' and d['quality_document']
    assert d['content_sha256']==hashlib.sha256(d['content'].encode()).hexdigest()
    assert any(p['text']==s['text'] and p['cut']==s['cut'] for p in d['pairs'])
    assert min(s['cut'],s['length']-s['cut']) >= 2
    joined.append(dict(**s,**{k:v for k,v in a.items() if k!='review_id'}))
assert sum(sum(h) for h in profile['histograms'].values())==profile['rows']==131006462
assert sum(f['rows'] for f in profile['files'])==profile['rows']
assert sum(g['rows'] for g in protocol['groups'])==protocol['frame_rows']==1700594
print('Verified 256-file census, 4,800 sampled documents and 640 frozen, distinct train-only reviews.')
print('Raw contents hash:',hashlib.sha256(documents_path.read_bytes()).hexdigest())
""")
md('## Supporting diagnostics — not the decision metric\n\n保留先前的文本文字诊断供追溯，不参与此次断点质量排序。全量文档数精确计数，未应用本地清理／去重。分层率的分母为该组抽查的 80 个独立文档，每篇一个题；Wilson 区间仅是独立二项近似参考，不包含抽样簇、标注误差或多重比较校正，不据此宣称显著胜出。')
code("""def wilson(k,n):
    z=1.959963984540054; p=k/n; den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return center-half, center+half
thresholds=[]
source_composition=[]
for t in profile['thresholds']:
    tick=round(t['threshold']*10000)
    assert sum(sum(h[tick:]) for h in profile['histograms'].values())==t['documents']
    assert sum(t['sources'].values())==t['documents']
    label='≥ '+str(t['threshold']) if tick<10000 else '= 1.0'
    thresholds.append(dict(threshold=label, score=t['threshold'], documents=t['documents'],
                           retained=f\"{100*t['fraction']:.3f}%\", share=t['fraction']))
    for source,n in t['sources'].items():
        source_composition.append(dict(threshold=label,source=source,documents=n,share=n/t['documents']))
bands=[]
source_review=[]
for b,label in enumerate(protocol['strata']):
    rows=[r for r in joined if r['bucket']==b]; assert len(rows)==80
    counts=collections.Counter(r['code'] for r in rows)
    damaged=sum(r['text_type']=='damaged' for r in rows)
    structural=sum(r['severity']=='structural_or_severe' for r in rows)
    lo,hi=wilson(damaged,len(rows))
    bands.append(dict(band=label,n=len(rows),damage=damaged,damage_rate=damaged/len(rows),
                      damage_percent=f'{100*damaged/len(rows):.2f}%',
                      wilson_reference=f'{100*lo:.1f}–{100*hi:.1f}%',
                      structural=structural,minor=damaged-structural,boilerplate=counts['B'],
                      boundary_error=counts['E'],boundary_uncertain=counts['X']+counts['U'],
                      median_length=statistics.median(r['length'] for r in rows)))
    for source in sorted({r['source'] for r in rows}):
        ss=[r for r in rows if r['source']==source]
        source_review.append(dict(band=label,source=source,n=len(ss),
                                 damage=sum(r['text_type']=='damaged' for r in ss)))
print(json.dumps(bands,ensure_ascii=False,indent=2))
print('No confirmed wrong boundary labels in this review; uncertain:',sum(r['boundary_uncertain'] for r in bands))
print('Zero out of 80 Wilson upper reference:',wilson(0,80)[1])
""")
md('### 抽取适配性与明确的表面指标\n\n以下为 4,800 篇样本中训练划分的文档，含未通过本地过滤者。quality_document 要求汉字 ≥80 且 汉字/(汉字+ASCII 字母) ≥0.70；它不是人工质量分。产出只做文档内同文同切点去重，未做历史 Bloom、跨文档或近重复去重。字面转义和半角逗号只是可复算的表面指标，不自动等同于错误标签。')
code("""patterns=protocol['policy']['sample_filter']['fragment_patterns']
literal=re.compile('|'.join(patterns[k] for k in ['literal_line_escape','literal_control_escape','literal_unicode_residue']))
extraction=[]
for b,label in enumerate(protocol['strata']):
    ds=[d for d in documents if d['bucket']==b and d['split']=='train']
    assert len([d for d in documents if d['bucket']==b])==600
    ps=[p for d in ds for p in d['pairs']]
    eligible=sum(bool(d['pairs']) for d in ds)
    lit=sum(bool(literal.search(d['content'])) for d in ds)
    ascii_only=sum(',' in d['content'] and '，' not in d['content'] for d in ds)
    short=latin=0
    for d in ds:
        text=unicodedata.normalize('NFKC',clean_document(d['content']))
        han=sum(is_han(c) for c in text)
        lat=sum(c.isascii() and c.isalpha() for c in text)
        short+=han<80; latin+=han/max(1,han+lat)<.70
    extraction.append(dict(band=label,train_documents=len(ds),eligible=eligible,
        eligible_rate=eligible/len(ds),eligible_percent=f'{100*eligible/len(ds):.1f}%',
        quality_reject=sum(not d['quality_document'] for d in ds),short_han=short,low_han_ratio=latin,
        no_pair_after_quality=sum(d['quality_document'] and not d['pairs'] for d in ds),
        pairs=len(ps),mean_pairs=round(len(ps)/len(ds),2),
        literal_escape=lit,literal_rate=lit/len(ds),ascii_comma_only=ascii_only,
        ascii_comma_percent=f'{100*ascii_only/len(ds):.1f}%',
        median_length=statistics.median(p['length'] for p in ps),
        period_share=sum(p['punctuation']=='。' for p in ps)/len(ps),
        review_period_share=sum(r['punctuation']=='。' for r in joined if r['bucket']==b)/80))
    assert eligible+extraction[-1]['quality_reject']+extraction[-1]['no_pair_after_quality']==len(ds)
print(json.dumps(extraction,ensure_ascii=False,indent=2))
print('Sample duplicate Han signatures:',len(documents)-len({d['document_id'] for d in documents}))
print('Sample duplicate exact content:',len(documents)-len({d['content_sha256'] for d in documents}))
""")
md('### 稳健性与边界\n\n避免用“所有高分段合并 160 个样本”的简单平均估全库：各分层人口和可抽取率不同。分来源复查仅描述，不将来源命名为领域，也不引入来源权重。没有运行模型，因此不能输出准确率增益或最优阈值。')
code("""sensitivity=[]
for label in protocol['strata']:
    rr=[r for r in joined if protocol['strata'][r['bucket']]==label and r['source']=='Tele']
    k=sum(r['text_type']=='damaged' for r in rr)
    sensitivity.append(dict(band=label,source='Tele',n=len(rr),damage=k,rate=k/len(rr)))
print('Same-source diagnostic (small n; not causality):',json.dumps(sensitivity,ensure_ascii=False))
print('Structural/severe-only counts by ascending band:',[r['structural'] for r in bands])
assert sum(r['damage'] for r in bands)==40
assert sum(r['boundary_error'] for r in bands)==0
assert sum(r['boundary_uncertain'] for r in bands)==3
print('Key count reconciliation passed. Training remains paused.')
""")
md('## Decision metric — boundary acceptability only\n\n读取冻结标注中独立的 boundary 字段，A 为合理、E 为明确不合理、U 为不确定。初始代码 B（附属文字）和 D（文字有瑕疵）若 boundary=A，仍完整计入合理。不得因文字问题扣分，不确定不得计作正确或错误。一个位置可合理而非唯一最优；原文有标点也不是合理的充分依据。下表使用所有 80 个样本作分母，并保留未决数量。')
code("""boundary_bands=[]
for b,label in enumerate(protocol['strata']):
    rows=[r for r in joined if r['bucket']==b]
    c=collections.Counter(r['boundary'] for r in rows)
    assert set(c) <= {'A','E','U'}
    assert c['A']+c['E']+c['U']==len(rows)==80
    boundary_bands.append(dict(band=label,n=len(rows),acceptable=c['A'],
        wrong=c['E'],uncertain=c['U'],acceptable_count=f\"{c['A']}/{len(rows)}\",
        wrong_count=f\"{c['E']}/{len(rows)}\",acceptable_share=c['A']/len(rows)))
assert sum(r['acceptable'] for r in boundary_bands)==637
assert sum(r['wrong'] for r in boundary_bands)==0
assert sum(r['uncertain'] for r in boundary_bands)==3
assert all(r['boundary']=='A' for r in joined if r['code'] in ['B','D'])
print(json.dumps(boundary_bands,ensure_ascii=False,indent=2))
print('Uncertain IDs:',[r['review_id'] for r in joined if r['boundary']=='U'])
print('Previously flagged text-damage rows still accepted:',sum(r['text_type']=='damaged' and r['boundary']=='A' for r in joined))
print('Reference only: zero errors among 80 independent fully adjudicated examples gives Wilson upper bound',wilson(0,80)[1])
""")
md('## Takeaways\n\n所有分数段均未发现明确错误断点；这表示当前初审无法区分，并非证明各段等质或真实错误率为零。撤回优先选择 0.99 的建议，不按文字瑕疵率筛阈值。下一轮若扩大研究，应继续以断点合理性为唯一指标，并预先冻结判定标准、隐藏 score、保留不确定类、增加独立复核及相同口径的对照样本。现有数据和训练配置不变，训练仍暂停。')

notebook = dict(nbformat=4, nbformat_minor=5,
                metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python'),
                              execution_note='Sequential Python execution by build.py, not a Jupyter kernel.'),cells=cells)
for cell in cells:
    if cell['cell_type']=='code': ast.parse(''.join(cell['source']))
assert len({c['id'] for c in cells})==len(cells)
assert [c['execution_count'] for c in cells if c['cell_type']=='code']==list(range(1,execution_count+1))
save(HERE/'research.ipynb',notebook)

datasets={k:namespace[k] for k in ['boundary_bands','thresholds','source_composition','bands','source_review','extraction','sensitivity']}
examples=[]
for ident in ['S0269','S0288','S0421','S0610']:
    r=next(r for r in namespace['joined'] if r['review_id']==ident)
    text=r['text'][:r['cut']]+' | '+r['text'][r['cut']:]
    examples.append(dict(id=ident,score=r['tick']/10000,example=text,reason=r['reason']))
datasets['examples']=examples
database=OUT/'evidence.sqlite'
with sqlite3.connect(database) as connection:
    connection.row_factory=sqlite3.Row
    for name,data in datasets.items():
        keys=list(data[0])
        connection.execute('DROP TABLE IF EXISTS '+name)
        connection.execute('CREATE TABLE '+name+' ('+','.join('"'+k+'"' for k in keys)+')')
        connection.executemany('INSERT INTO '+name+' VALUES ('+','.join('?' for _ in keys)+')',
                               [tuple(row[k] for k in keys) for row in data])
        result=[dict(row) for row in connection.execute('SELECT * FROM '+name)]
        assert result==data
        datasets[name]=result
    assert tuple(connection.execute('SELECT SUM(n),SUM(damage),SUM(boundary_uncertain) FROM bands').fetchone())==(640,40,3)
    assert tuple(connection.execute('SELECT SUM(n),SUM(acceptable),SUM(wrong),SUM(uncertain) FROM boundary_bands').fetchone())==(640,637,0,3)
    assert connection.execute("SELECT documents FROM thresholds WHERE score=0.99").fetchone()[0]==4023576

timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
sources=[]
for name in datasets:
    sources.append(dict(id=name,label=name+' · frozen local score study',path=str(database.relative_to(ROOT)),
        query=dict(engine='SQLite',language='sql',sql='SELECT * FROM '+name,executed_at=timestamp,
                   tables_used=[name],description='Read-only census, stratified sample and frozen score-blind AI annotations.',
                   filters=['No training; semantic review uses train hash partition only; min side 2'],
                   metric_definitions=['Document census is exact. Review rates are one pair per eligible document, 80 per band. No pooled unweighted population quality estimate.'],
                   source_notes='See protocol.json, freeze.json and executed research.ipynb. Wilson intervals are unadjusted reference only. Bars start at zero; categorical bands are unequal-width ranges.')))

manifest=dict(version=1,title='Ultra-FineWeb Boundary Quality Review',generatedAt=timestamp,sources=sources,blocks=[],charts=[],tables=[])


def section(ident,body,source_id=None):
    block=dict(id=ident,type='markdown',body=body)
    if source_id: block['sourceId']=source_id
    manifest['blocks'].append(block)


def table(ident,title,dataset,columns):
    manifest['tables'].append(dict(id=ident,title=title,dataset=dataset,
        source=next(s for s in sources if s['id']==dataset),density='comfortable',
        columns=[dict(field=k,label=v) for k,v in columns]))
    manifest['blocks'].append(dict(id=ident+'-block',type='table',tableId=ident))


def chart(ident,title,dataset,x,y,xlabel,ylabel):
    manifest['charts'].append(dict(id=ident,title=title,type='bar',dataset=dataset,sourceId=dataset,valueFormat='percent',
        encodings=dict(x=dict(field=x,type='nominal',label=xlabel),y=dict(field=y,type='quantitative',label=ylabel)),
        rationale='Zero-baseline categorical bar with one common denominator definition per series; categories ordered by score. Not a time series or continuous density. No inferential claim from visual separation.'))
    manifest['blocks'].append(dict(id=ident+'-block',type='chart',chartId=ident))


section('title','# Ultra-FineWeb Boundary Quality Review')
section('summary','## 只按断点合理性，目前无法区分各分数段质量\n\n复用已经冻结的 640 条盲审记录：**637 条判为合理，0 条明确不合理，3 条不确定**。八个分数段都未发现明确错切，因此本轮不足以判断 score 与断点质量的关系。\n\n撤回原先优先选择 0.99 的建议。之前的文字瑕疵比例不用于此次质量排序；没有修改原始标注，没有新增抽样或模型训练。')
section('definitions','## 唯一标准：标记处是否构成合理阅读停顿\n\n- **合理**：在给定文本及必要原文上下文中，这个位置可以自然停顿、分组；不要求它是唯一或最佳切点。\n- **明确不合理**：该位置拆断应连续理解的词或结构，且能说明为何不适合作为阅读边界。原文有标点本身不能保证合理。\n- **不确定**：上下文不足、损坏或歧义使位置无法可靠判断，单独保留，不计为对或错。\n\n错字、空格、广告、文体、事实真伪、内容是否有教育价值均不单独扣分。文字有瑕疵但目标断点合理，仍计为合理。分数段的分母均为 80 篇不同文档各取的一道题。')
section('result','## 所有组明确错切均为 0，不确定项共 3 条\n\n这是一轮已有 AI 初审的重新汇总，不是新一轮独立验证。比较的是当前清理和抽取规则下产出的题目，最短边至少两字；不评价未通过过滤的原始文档。','boundary_bands')
table('boundary-detail','各分数段的断点判断 · 合理、不合理、不确定互斥','boundary_bands',[('band','分数区间'),('acceptable_count','AI 判合理 / 抽查数'),('wrong_count','AI 判明确不合理 / 抽查数'),('uncertain','不确定')])
chart('boundary-acceptance','AI 初审可判为合理的占比 · 差额均为不确定，不是错切','boundary_bands','band','acceptable_share','文档分数区间','判为合理 / 全部抽查样本')
section('interpretation','0.9–0.95 组有 2 条不确定，0.999–1.0 组有 1 条不确定。因此 78/80 与 79/80 不能解释为对应组有 2 条和 1 条错误，图中高度差也不能作为质量排名。\n\n不删除不确定样本后只报“全部 100% 正确”，也不把它们全算错。所有组均没有明确错误样本，只能得出本轮未检出差异，不能得出各组真实质量相同。','boundary_bands')
section('mapping','## 有文字瑕疵的 37 条依然计入合理\n\n冻结记录已经分别保存 text_type 和 boundary。此前文字有问题的 40 条中，37 条的 boundary=A，本轮全数计入合理；另外 3 条 boundary=U 保持不确定。另有 24 条推广等附属文字，切点合理，也全数计入合理。\n\n例如“有耐心、关心、细心4. 与家长进行沟通｜及时了解家长需求”：前面存在编号拼接，但目标“沟通｜及时了解”可以停顿，因此按本次唯一标准仍算合理。不能用文字质量问题代替标签问题。')
section('unknown','## 三条不确定记录保持未决，不据此扣分\n\nS0241 的食谱表达中多处关系混乱；S0378 的医学表述难以还原；S0399 的基因／疾病表格被压成连续文字。已有审查在查看原文后仍未能可靠判断目标边界，因此保留 U。\n\n这些属于需复核的个案，不代表其所在分数段质量更差。原文定位、目标位置与逐条理由保留在 sample.json 和 annotations.json 中。')
section('method','## 复用冻结的盲审记录，保持比较口径一致\n\n- 固定 Ultra-FineWeb 版本 02c85641e3d19a854be2e09139c25adaa9518063；本地完整分数普查覆盖 256 个文件、131,006,462 篇文档。\n- 抽样种子 2026100451：先均匀随机抽 32 个行组（1,700,594 行），再每个分数段抽 600 篇，共 4,800 篇；每段从可抽取的训练文档选 80 篇，各随机取一道题。\n- 首轮隐藏 score 与来源，复核全部初始异常／不确定项以及随机 80 个 A 类，然后冻结标注。此次只读取冻结的 boundary 字段，未重新打分；哈希校验通过。\n- 使用当前 v8 清理及两侧至少两字规则，文档内同文同切点去重。语义样本为文档等权，不是训练行等权，也未完成跨语料近重复处理。\n- 按已有哈希划分种子 20260915 排除该划分的验证／测试文档，但未检查其他历史语料的留出注册表。这批是开发审计样本，不是正式测试集。')
section('limits','## 每组 80 条与单一 AI 审核限制了区分能力\n\n假设独立抽样、判断完全准确，即便某组真实错切率为 1%，抽 80 条时一个错误也碰不到的概率仍约 45%。这是用于说明抽样能力的假设例子，不是对真实错误率的估计。实际还有行组抽样和 AI 判断误差，不能把 0/80 当成零错误证明。\n\n本轮没有独立人工金标准，也没有先量化审核者漏判率。最终入选样本受已有过滤器影响，因此结论仅覆盖过滤后的、最短边至少两字的题目。已知表面噪声比例和抽取产出率不参与质量判断。')
section('decision','## 没有依据按 score 收紧阈值来提高断点质量\n\n当前证据不支持选择 0.99、0.999 或任何一个分数段作为更优质量档位；也不证明 score 与断点质量完全无关。因此保持现有数据和筛选配置，不基于文字瑕疵率作决定，训练仍暂停。\n\n如果继续扩大研究，唯一主指标仍是合理／明确不合理／不确定三个计数。应预先冻结同一判定标准、隐藏分数、增加各档独立文档数，并用独立复核检查漏判；抽样数量与断点长度、标点类型等只用于控制可比性，不作为额外质量指标。')
section('questions','## 后续需要区分的是未检出差异与确实没有差异\n\n扩大样本并校验审核可靠性后，能否检出不同 score 的错切率差异？同一断点长度和标点类型内，差异是否仍存在？只有对这些问题取得证据后，才能按我们的唯一质量标准选择 score 门槛。')

payload=dict(surface='report',manifest=manifest,
             snapshot=dict(version=1,generatedAt=timestamp,status='ready',datasets=datasets),sources=sources)
save(OUT/'artifact.json',payload)
save(HERE/'metrics.json',datasets)
save(HERE/'checks.json',dict(assessment='Share with caveats',raw_documents=namespace['profile']['rows'],
    sampled_documents=len(namespace['documents']),semantic_reviews=len(namespace['joined']),
    primary_quality_metric='Boundary acceptability only; text damage excluded from decision',
    reused_frozen_annotations=True,new_semantic_reviews=0,preferred_score_threshold=None,
    acceptable=637,wrong=0,uncertain=3,
    frozen_hashes_verified=True,sql_reconciliation=True,notebook_code_cells_executed=execution_count,
    notebook_execution_mode='Sequential Python standard library, not a Jupyter kernel',
    statistical_caveats=['Clustered sample; unadjusted Wilson reference only','AI review, no independent human gold','Document-weighted semantic sample','No measured model improvement'],
    no_training_started=True,production_configuration_unchanged=True,
    required_structure='summary; visual findings; definitions before findings; methodology; limitations and sensitivity; recommendations; open questions',
    visual_qa='Run packaged report renderer next; browser rendering may be unavailable.'))
print(json.dumps(dict(report_input=str(OUT/'artifact.json'),notebook=str(HERE/'research.ipynb'),checks='passed'),ensure_ascii=False))
