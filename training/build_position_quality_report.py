"""Canonical portable report and executed-notebook companion for the audit."""
from __future__ import annotations

import copy
import datetime
import json
import sqlite3

from audit_position_quality import OUT, REVIEW, ROOT, save


def build():
    f = json.loads((OUT/'findings.json').read_text())
    rows = json.loads((OUT/'reviewed.json').read_text())
    indexed = {r['review_id']:r for r in rows}
    protocol = json.loads((REVIEW/'protocol.json').read_text())
    title = '断点位置与标签质量：1,600 条训练样本分层审查'
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    source = dict(id='audit',label='冻结训练样本与 Codex AI 逐条初审 · 2026-10-04',
        path='training/reviews/position-quality-20261004/',query=dict(
            engine='Python 3',language='python',
            description='audit_position_quality.py 分层抽样、验证冻结标注并计算加权比例；SQL 校验和导出报告数据。',
            tables_used=['sample.jsonl','annotations.jsonl','protocol.json','freeze.json'],
            filters=['历史 v7 训练数据；20 个已随机选择并校验哈希的完整分片；共 2,475,865 行。',
                     '每个距离区间 200 条；来源分配按扩展行数最大余数法；第二阶段种子 2026100431。',
                     'Codex AI 逐条阅读全部 1,600 个规范化文本对；不加载预测，不读取 test，不恢复原文档。'],
            metric_definitions=['A=当前标记处可接受，非唯一答案声明；E=可定位的明确边界缺陷；U=尚不能确定，不算错误。',
                                '距离 d=min(cut,n-cut)，相对距离 q=d/n；短边可能在左也可能在右。',
                                '权重=(来源块总分片数/4)×来源块距离层中的候选行数/该层抽查数；不是训练损失权重。'],
            executed_at=timestamp,sample_sha256=f['freeze']['sample_sha256'],
            annotations_sha256=f['freeze']['annotations_sha256'],
            source_notes='Technical report; definitions moved ahead of findings for interpretability. Two single-series zero-baseline bars compare categories, not time: absolute U raw share and relative U weighted share. Same blue-root shared palette; ordered category labels carry identity, no redundant color legend. Full-width charts; exact counts, small side groups, examples and sensitivity use native tables. No precision or human-gold claims from reviewer judgments. No causal claim.'))
    manifest=dict(version=1,title=title,generatedAt=timestamp,sources=[source],blocks=[],charts=[],tables=[])
    datasets={}

    def md(key,body):
        manifest['blocks'].append(dict(id=key,type='markdown',body=body,sourceId='audit'))

    def table(key,title,data,columns,sort):
        datasets[key]=[{k:v for k,v in r.items() if v is None or isinstance(v,(str,int,float,bool))} for r in data]
        manifest['tables'].append(dict(id=key,title=title,dataset=key,source=source,density='comfortable',
            defaultSort=dict(field=sort,direction='asc'),
            columns=[dict(field=k,label=label,**({'format':fmt} if fmt else {})) for k,label,fmt in columns]))
        manifest['blocks'].append(dict(id=key+'-block',type='table',tableId=key))

    def bar(key,title,data,y,subtitle):
        datasets[key]=data
        manifest['charts'].append(dict(id=key,title=title,subtitle=subtitle,type='bar',dataset=key,sourceId='audit',
            encodings=dict(x=dict(field='group',type='nominal',label='断点位置'),
                           y=dict(field=y,type='quantitative',label='存疑比例'),
                           tooltip=[dict(field='n',label='抽查条数'),dict(field='U',label='存疑条数'),
                                    dict(field=y,label='存疑比例',format='percent')]),
            valueFormat='percent',xAxisTitle='断点位置',yAxisTitle='存疑比例',
            rationale='按语义顺序比较互斥位置类别的比例，零基线柱图；U 与 E 分开，精确分母与判定限制在相邻正文和表格。'))
        manifest['blocks'].append(dict(id=key+'-block',type='chart',chartId=key))

    md('title','# '+title)
    md('summary','## 单字边界明显更常存疑，其余位置没有严格单调的质量排序\n\n'
       '**单字组 200 条中，110 条可接受、5 条明确有问题、85 条存疑；其余各组的存疑比例为 0.5%–4.5%。** '
       '八组共审查 1,600 条，是前次 452 条审查的约 3.5 倍；本次来自训练分片，前次来自验证集，不能当作同一母体的前后变化。\n\n'
       '单字在右侧时更容易缺上下文：58/109 条存疑，左侧为 23/86。相对位置接近中间的样本总体更易判断，但不能由此证明位置本身导致质量变化。'
       '**本结果支持优先追查单字尾片段和抽取残缺，不支持把全部单字或双字样本直接删除。**')
    md('definitions','## 先分清距离、位置和三种判定\n\n'
       '“｜”是原训练标签。cut 表示左侧字符数，等于 target_index + 1；字符按规范化 Python 字符串计数，包括汉字、数字、字母等，并非分词后的词数。'
       '**d=min(左侧字符数,右侧字符数)**。因此第 1 字与倒数第 1 字都归入 d=1；每个距离区间抽 200 条。\n\n'
       '**q=d/句长** 区分真正靠边和接近中间，q=50% 表示正中间。例如 10 字句的 5|5 和 100 字句的 5|95 同属 d=5，q 却分别为 50% 与 5%。\n\n'
       '- **A 可接受**：当前边界可以自然停顿；未核查事实真伪，也未证明其它位置都错。\n'
       '- **E 明确问题**：可从当前输入指出拆词、拆术语或紧密结构等缺陷。\n'
       '- **U 存疑**：孤立字、残句、排版、抽取损坏或语义歧义，需要更多上下文。**U 不能计为已确认坏标签。**\n\n'
       '所有判断均由同一个 AI 逐条作出，未使用模型预测，也未经独立人工裁决。A 不等于“整条样本完全干净”，更不等于“单标签监督没有冲突”。')
    md('absolute-intro','## 从单字到双字，存疑比例下降最明显\n\n'
       'd=1 的存疑比例是 **42.5%**，d=2 为 **3.5%**。但 d=3 为 4.5%，比 d=2 略高；d=7–10 也略高于 d=5–6。'
       '因此当前观察是“单字组明显特殊、较内侧大体更稳定”，而不是“每往中间移一个字都更好”。')
    bar('absolute-chart','各距离区间的存疑比例',f['absolute'],'U_raw',
        '2026-10-04 · 每组 200 条 · U 条数/200 · AI 初审，U 不等于错标')
    table('absolute','八个距离区间的实际审查计数',f['absolute'],
          [('order','顺序','number'),('group','距最近一端的字符数',None),('n','抽查数','number'),
           ('A','可接受 A','number'),('A_raw','A 比例','percent'),('E','明确问题 E','number'),
           ('E_raw','E 比例','percent'),('U','存疑 U','number'),('U_raw','U 比例','percent'),
           ('U_weighted','来源校正 U 比例','percent')],'order')
    md('absolute-note','各组内按来源规模分配名额，来源校正前后的存疑比例差异均小于 **0.1 个百分点**。'
       '本表给出实际 n=200 的计数，便于直接检查。其它七组没有发现 E，**不代表它们不存在坏标签**。')
    md('relative-intro','## 按相对位置看，中间区域较少存疑，但仍非逐档递减\n\n'
       '按来源与距离抽样权重还原后，最近一端 0–10% 区域的 U 估计为 **4.14%**，40–50% 中间区域为 **0.42%**。'
       '20–30% 与 30–40% 两档为 2.27% 和 2.58%，不能排列成严格递减关系。\n\n'
       '这里必须使用抽样权重：单字样本被特意多抽，不同距离的 200 条不能直接合并。'
       '相对位置组中的“实际审查数”也不是加权比例的简单分母。')
    relative=[r for r in f['relative'] if r['minimum_length']==2]
    bar('relative-chart','相对位置区间的加权存疑比例',relative,'U_weighted',
        '2026-10-04 · 全部 1,600 条重新分组 · 来源及距离抽样权重校正 · q=d/句长')
    table('relative','相对位置：样本量与加权比例',relative,
          [('order','顺序','number'),('group','q 区间',None),('n','实际审查数','number'),
           ('effective_n','Kish 权重有效量','number'),('E_weighted','加权 E','percent'),
           ('U_weighted','加权 U','percent'),('A_weighted','加权 A','percent')],'order')
    md('relative-note','Kish 有效量仅描述权重不均带来的信息折损，不是独立文档数。限定句长至少 16 字后，'
       '两端 0–10% 的加权 U 为 3.47%，中间 40–50% 为 0.11%；方向未反转，但句长与内容类型仍未被完全控制。')
    md('sides-intro','## 单字尾片段比单字开头更常缺上下文\n\n'
       '单字在左侧：**23/86（26.7%）** 存疑；单字在右侧：**58/109（53.2%）** 存疑。'
       '另有 5 条两边都只有一个字，单独列出，避免重复计算。\n\n'
       '单字组 85 条 U 中，59 条归入孤立字或残句，11 条排版/词条结构不明，7 条文字损坏，8 条词语关系有歧义。'
       '这与“句尾只剩下一句开头或下一词条的一个字”这一风险相符；缺少原文，尚不能确认其抽取成因。')
    table('sides','单字组的左右两侧比较',[r for r in f['sides'] if r['stratum']=='1'],
          [('group','较短侧',None),('n','抽查数','number'),('A','A','number'),('E','E','number'),
           ('U','U','number'),('U_raw','存疑比例','percent')],'group')
    md('examples-intro','## 明确缺陷集中在单字组，同时存在大量有效的单字监督\n\n'
       '五条 E 如下。它们是 AI 根据当前规范化输入作出的局部判断，不是对原作者标点的裁决。'
       '其余 E=0 的组不能据此认为绝对无噪声。')
    table('errors','全部五条明确问题', [r for r in rows if r['verdict']=='E'],
          [('review_id','样本',None),('marked_text','原标签',None),('reason','问题说明',None)],'review_id')
    table('contrasts','同为短边界，判断取决于内容',
          [indexed[k] for k in ['P0003','P0148','P0602','P1364','P1170','P0556','P0926']],
          [('review_id','样本',None),('marked_text','原标签',None),('verdict','初审',None),('reason','理由',None)],'review_id')
    md('robust-intro','## 来源和句长限制没有消除单字组的差异\n\n'
       '只看网页来源时，单字组 U 为 **84/194（加权 43.3%）**；进一步限定网页样本且句长 16–64 字，'
       '仍有 **13/33（加权 39.4%）** 存疑。后者样本数明显减少，只能作粗敏感性检查。\n\n'
       '逐一剔除某个分片的已审查行、保持其余行原权重，单字组 U 落在 **39.3%–44.0%**；'
       '未发现由单个已抽分片独自制造这一大差异。该范围不是置信区间，也没有控制同文档样本相关性。')
    table('robustness','网页来源与句长范围内的对比',
          [r for r in f['robustness'] if r['scope'] in ['web only','web; 16–64 chars']],
          [('scope','范围',None),('group','d 区间',None),('n','抽查数','number'),('E','E','number'),
           ('U','U','number'),('U_weighted','加权 U','percent')],'scope')
    md('methods','## 抽样只读取历史训练数据，未改语料和模型\n\n'
       '母体为当前历史 **unicode-context-v7** 准备数据。第一阶段沿用此前固定随机抽取的 20 个完整训练分片，'
       '分属 base/wiki、synthetic 与三批 web 数据，各 4 个，共 **2,475,865 行**；文件字节数、行数与 SHA-256 均核验。\n\n'
       '第二阶段在“来源块 × 距离区间”的 40 个单元内作无放回 reservoir 抽样，每个距离层总计 200 条。'
       '来源名额按“来源块总分片数/4 × 抽中分片该距离行数”的比例分配；最大余数法取整。'
       '种子 2026100431，审查顺序再次打乱，逐条标注冻结后才汇总。\n\n'
       '跨距离汇总的行权重为 **(来源块总分片数/4) × 单元候选数 / 单元审查数**。'
       '这是抽样还原权重，与训练的断点位置损失权重无关。\n\n'
       '新增 v8 字面量转义规则尚未重建这批历史语料。本次 1,600 条中没有命中三项字面转义规则的样本；'
       '这并不否认大扫描已发现的残留，也说明字面量清理不能覆盖这里观察到的所有词条、残句、乱码问题。')
    md('uncertainty','## 当前证据能识别优先级，不能给出全库真实坏样本率\n\n'
       '- 这是单个 AI 的初审，既不是人工共识，也不是独立语义 gold。已复核全部 E/U、释义类 A，以及固定随机选取的 80 条 A。\n'
       '- 紧凑训练行缺少 document_id、原标点和完整原文，只能追溯到分片、行号与文本哈希。1600 条文本和切点组合均不同，仍无法排除同文档或同模板相关性。\n'
       '- 20 个分片不等于全库随机独立样本。名额和权重已考虑来源规模，仍可能受第一阶段分片差异影响。\n'
       '- 若仅作独立同分布二项抽样参考，单字 E=5/200 的 95% Wilson 区间约为 1.1%–5.7%；单字 U 为 35.9%–49.4%。'
       '普通组 E=0/200 的对应上界约为 1.9%。这些参考范围不覆盖文档相关性或审查者误判，不能当全库置信界。\n'
       '- 长度、左右侧、资料类型和位置相互关联；范围限制未构成完全匹配或因果识别。\n'
       '- 本次只判断被标出的断点是否合理。没有枚举其他可接受断点，不能用 A 比例证明“一条样本只有一个正确位置”。')
    md('next','## 优先查清孤立尾片段，再决定过滤规则\n\n'
       '先人工复核五条 E 和单字尾片段的 U，利用来源文档定位它们是断行、校勘文字、词条拼接还是原始语料问题。'
       '再按可验证的成因设清理规则，并保留正常感叹词、字段和释义监督。**不应直接按 d=1、d=2 一刀切。**\n\n'
       '仍待确认：U 中多少经原文可恢复为有效标签？同一文档或模板贡献了多少相似噪声？'
       '剔除确认错误后，在固定训练预算与独立多切点语义验证上是否有提升？本轮没有训练或更换模型。')
    md('all','## 1,600 条记录可以逐条复核\n\n'
       '下面保留全部冻结判断。sample.jsonl 保存原文本与分片位置，annotations.jsonl 保存判定，'
       'protocol.json 保存抽样计数和哈希，audit.ipynb 可复算。human-review.csv 留有空白人工判定列，不包含 AI 答案或模型预测。')
    table('review','全部冻结审查记录',rows,
          [('review_id','样本',None),('stratum','d 区间',None),('marked_text','原标签',None),
           ('verdict','判定',None),('reason','理由',None),('block','来源块',None)],'review_id')
    # Executable SQL lineage and independent reconciliation from individual rows.
    db=OUT/'report-evidence.sqlite'
    if db.exists():
        db.unlink()
    quote=lambda s:'"'+s.replace('"','""')+'"'
    with sqlite3.connect(db) as con:
        con.row_factory=sqlite3.Row
        for name,data in datasets.items():
            assert data
            fields=list(data[0]); assert all(set(r)==set(fields) for r in data)
            con.execute(f'CREATE TABLE {quote(name)} ({", ".join(quote(k) for k in fields)})')
            con.executemany(f'INSERT INTO {quote(name)} VALUES ({", ".join("?" for _ in fields)})',
                            [tuple(r[k] for k in fields) for r in data])
            assert [dict(r) for r in con.execute(f'SELECT * FROM {quote(name)}')]==data
        query='''SELECT stratum, COUNT(*) AS n,
          SUM(verdict='A') AS A, SUM(verdict='E') AS E, SUM(verdict='U') AS U,
          SUM(weight*(verdict='U'))/SUM(weight) AS weighted_U
          FROM review GROUP BY stratum'''
        aggregate={r['stratum']:dict(r) for r in con.execute(query)}
        for r in f['absolute']:
            other=aggregate[r['group']]
            assert all(r[k]==other[k] for k in ['n','A','E','U'])
            assert abs(r['U_weighted']-other['weighted_U'])<1e-12
        source['path']=str(db.relative_to(ROOT))
        source['query'].update(engine='SQLite',language='sql',sql='SELECT * FROM "review"',
                              independent_validation_sql=query)
        for item in manifest['charts']+manifest['tables']:
            name=item['dataset']; item_source=copy.deepcopy(source);item_source['id']='source-'+name
            item_source['query']['sql']=f'SELECT * FROM {quote(name)}'
            item_source['query']['tables_used']=[name,'training/reviews/position-quality-20261004/findings.json']
            manifest['sources'].append(item_source)
            if 'sourceId' in item:item['sourceId']=item_source['id']
            if 'source' in item:item['source']=item_source
    payload=dict(surface='report',manifest=manifest,
                 snapshot=dict(version=1,generatedAt=timestamp,status='ready',datasets=datasets),sources=manifest['sources'])
    save(OUT/'artifact.json',payload)
    save(OUT/'analysis-qa.json',dict(sql_absolute_counts_and_weights_match=True,
         frozen_annotations_verified=True,individual_decisions=1600,
         all_export_queries_executed=True,independent_validation_sql=query,
         source_files_verified_at_sampling=True,visual_verification='pending portable delivery'))
    return f


def notebook():
    import nbformat as nbf
    nb=nbf.v4.new_notebook()
    nb.metadata['kernelspec']=dict(display_name='Python 3',language='python',name='python3')
    md,code=nbf.v4.new_markdown_cell,nbf.v4.new_code_cell
    nb.cells=[
        md('# 断点位置与标签质量\n\n## tl;dr\n\n1,600 条训练样本，每个距离区间 200 条。单字组 A=110、E=5、U=85；其它区间 U 为 0.5%–4.5%。U 不是已确认错误。AI 逐条初审，尚待人工确认。'),
        md('## Context & Methods\n\n20 个已固定随机选择的分片，候选 2,475,865 行。来源块 × 距离分层；名额按扩展行数分配；种子 2026100431。\n\n### Key Assumptions\n\n抽样权重与训练损失权重不同。当前原标签可接受不代表唯一答案或全文无噪声。未恢复原文档，无法估计同文档聚类。未加载预测、验证或 test。'),
        code("from pathlib import Path\nimport sys, json, io, contextlib, sqlite3, math\nroot = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'training/audit_position_quality.py').exists())\nsys.path.insert(0, str(root/'training'))\nfrom audit_position_quality import analyze, OUT, REVIEW, sample\n"),
        md('## Data\n\n冻结的 sample.jsonl、annotations.jsonl、protocol.json 和 freeze.json 足以复算统计。重新抽样另需原 20 个 ignored 训练分片及 structural.json；可运行 `python3 training/audit_position_quality.py --sample`，脚本会拒绝覆盖不同样本。'),
        code("with contextlib.redirect_stdout(io.StringIO()):\n    result = analyze()\nassert all(result['checks'].values())\nprint(json.dumps(result['checks'],ensure_ascii=False,indent=2))"),
        md('## Results\n\n### 逐层计数及加权比例'),
        code("for r in result['absolute']:\n    assert r['n']==200 and r['A']+r['E']+r['U']==200\n    print(r['group'], {k:r[k] for k in ['n','A','E','U','U_weighted']})"),
        md('### 从逐条记录独立 SQL 复算\n\nSQLite 对原始 review 表重新分组计算，而不是直接读取汇总表。'),
        code("query = \"SELECT stratum, COUNT(*) n, SUM(verdict='A') A, SUM(verdict='E') E, SUM(verdict='U') U, SUM(weight*(verdict='U'))/SUM(weight) U_weighted FROM review GROUP BY stratum\"\nwith sqlite3.connect(OUT/'report-evidence.sqlite') as con:\n    con.row_factory=sqlite3.Row\n    sql={r['stratum']:dict(r) for r in con.execute(query)}\nfor r in result['absolute']:\n    assert all(r[k]==sql[r['group']][k] for k in ['n','A','E','U'])\n    assert math.isclose(r['U_weighted'], sql[r['group']]['U_weighted'],abs_tol=1e-12)\nprint('Eight strata: independent SQL counts and weights match.')"),
        md('### 相对位置、左右侧及句长敏感性\n\n相对位置跨多个抽样距离层，只读加权比例，不能直接用 U/n 当母体估计。'),
        code("for r in result['relative']:\n    if r['minimum_length']==2:\n        print(r['group'],r['n'],r['U_weighted'])\nfor r in result['sides']:\n    if r['stratum']=='1':\n        print(r['group'],r['n'],r['U'],r['U_raw'])\nfor r in result['robustness']:\n    if r['scope']=='web; 16–64 chars':\n        print(r['scope'],r['group'],r['n'],r['U_weighted'])"),
        md('## Takeaways\n\n优先回溯单字尾片段和明确 E，保留正常感叹词、字段、释义等监督。暂不按位置一刀切。独立同分布区间只是抽样参考，不覆盖文档聚类、分片偏差或 AI 判断误差。下一步需原文与人工复核。'),
        code("assert result == json.loads((REVIEW/'findings.json').read_text())\nprint('Frozen annotations and report findings reconciled; no model or corpus changed.')")]
    nbf.validate(nb);nbf.write(nb,REVIEW/'audit.ipynb')


if __name__=='__main__':
    build();notebook();print(OUT/'artifact.json')
