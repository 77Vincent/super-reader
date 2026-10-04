"""Build the canonical report payload and executable companion notebook.

Run audit_boundary_labels.py --analyze first. Package artifact.json with the
data-analytics plugin's report:deliver command; this script has no HTML runtime.
"""
from __future__ import annotations

import datetime
import copy
import json
import sqlite3
from pathlib import Path

from audit_boundary_labels import OUT, REVIEW, ROOT, write_json


def build():
    f = json.loads((OUT / 'findings.json').read_text())
    rows = json.loads((OUT / 'reviewed.json').read_text())
    follow = json.loads((REVIEW / 'prediction-followup.json').read_text())['rows']
    indexed = {r['review_id']: r for r in rows}
    title = '断点标签质量审查：单字过滤会同时删去噪声和有效监督'
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    source = {
        'id': 'audit', 'label': '2026-10-04 固定验证样本与 Codex AI 初审',
        'path': 'training/reviews/boundary-quality-20261004/',
        'query': {
            'engine': 'Python 3', 'language': 'python',
            'description': 'audit_boundary_labels.py 复现抽样、验证冻结标注并关联现有四组预测；所有语义判断由 AI 逐条作出，未经独立人工确认。',
            'tables_used': ['sample.jsonl', 'annotations.jsonl', 'protocol.json', 'freeze.json', 'prediction-followup.json',
                            'training/artifacts/label-quality-20261004/findings.json'],
            'filters': ['固定100,000条验证；全部252条单字样本与200条普通随机样本；种子2026100417。',
                        '先隐藏预测审查452条，再冻结标注；29条可接受标签的新增失配另作可见预测的事后诊断。',
                        '仅保留规范化后的两片段文本，未回溯完整原文，未评估test。'],
            'metric_definitions': ['A=局部断点可接受；M=可接受且明确列出其他切法；E=明确缺陷；U=无法确定。',
                                   '匹配分数是与原标点代理标签的一致性，不是人工语义准确率。',
                                   '单字组分类用单系列计数柱图；其余精确分母、逐条判断和反例用表格。'],
            'executed_at': timestamp,
            'validation_sha256': json.loads((REVIEW / 'protocol.json').read_text())['validation_sha256'],
            'annotations_sha256': f['freeze']['annotations_sha256'],
            'source_notes': 'Technical report: definitions precede findings; model design and validation dedicated sections; limitations and next steps cover open questions. Chart contract: compare 4 reviewed single-character label categories, n=252; canonical single-series bar, zero baseline, one approved blue root via shared theme, labeled categories not color encoding, full report width; remaining tables for exact audit lookup. No extrapolation. This is a single AI reviewer, not a human gold benchmark.',
        },
    }
    manifest = dict(version=1, title=title, generatedAt=timestamp, sources=[source], blocks=[], charts=[], tables=[])
    datasets = {}

    def paragraph(key, body, sourced=True):
        manifest['blocks'].append(dict(id=key, type='markdown', body=body,
                                       **({'sourceId': 'audit'} if sourced else {})))

    def table(key, title, data, columns, sort):
        datasets[key] = [{k: v for k, v in row.items() if v is None or isinstance(v, (str, int, float, bool))}
                         for row in data]
        manifest['tables'].append(dict(id=key, title=title, dataset=key, source=source,
            density='comfortable', defaultSort={'field': sort, 'direction': 'asc'},
            columns=[dict(field=k, label=label, **({'format': fmt} if fmt else {})) for k, label, fmt in columns]))
        manifest['blocks'].append(dict(id=key+'-block', type='table', tableId=key))

    paragraph('title', '# '+title, False)
    paragraph('summary', '## 有坏标签，但不能用单字长度代替质量判断\n\n'
              '**452 条 AI 初审确认了明确拆词、上下文残缺和单一答案过窄三类问题。** 单字组中 137/252 条的原断点可接受，4 条明确有问题，111 条无法确定；普通组 200 条中，50 条已列出另一种合理切法。\n\n'
              '删去单字训练样本后，一些原标签失配只是换了合理切法，也有预测开始拆开“王师”“土脉”等词。**应按具体缺陷改进监督；当前证据不支持直接删除所有单字样本。** 这是单个 AI 的初审与诊断，尚非人工共识。')
    paragraph('definitions', '## 审查两片段的边界，分别统计两组样本\n\n'
              '母体是已有 smoke 的固定 **100,000 条验证样本**。单字组是左右任一侧只有一个规范化字符的全部 **252 条**；普通组从其余 **99,748 条**中无放回随机抽取 **200 条**。252 条不是全部训练语料中的单字样本总数。\n\n'
              '- **A：可接受**，未枚举其他切法，不表示答案唯一。\n'
              '- **M：多种合理切法**，原断点可接受且至少保存一个替代位置，替代集合不穷尽。\n'
              '- **E：明确不合适**，局部能确认拆词、术语或紧密结构等缺陷。\n'
              '- **U：无法确定**，需要更多原文或排版；不能算作错误。\n\n'
              '文中的“｜”表示指定断点。数字切点表示左侧的字符数，即模型 gap 索引加 1。两组抽样比例不同，不能把 452 条合并计算全库错标率。')
    table('verdicts', '冻结初审的分类计数', f['review_counts'],
          [('cohort','组别',None),('n','审查数','number'),('documents','文档数','number'),
           ('A','可接受 A','number'),('M','多切法 M','number'),('E','明确问题 E','number'),('U','不确定 U','number')], 'cohort')
    single = next(r for r in f['review_counts'] if r['cohort'] == 'single')
    datasets['single-verdicts'] = [dict(category=label, count=single[k], denominator=252,
                                       share=single[k]/252, reviewer='AI 初审', verdict=k)
                                  for k,label in [('A','A 可接受'),('M','M 多切法'),('E','E 明确问题'),('U','U 不确定')]]
    manifest['charts'].append({
        'id':'single-verdicts', 'title':'单字样本初审分类', 'subtitle':'2026-10-04 · 252 条 · 单位：样本条数 · AI 初审，未人工确认',
        'type':'bar', 'dataset':'single-verdicts', 'sourceId':'audit',
        'encodings': {'x':{'field':'category','type':'nominal','label':'初审分类'},
                      'y':{'field':'count','type':'quantitative','label':'样本条数'},
                      'tooltip':[{'field':'count','label':'条数'}, {'field':'denominator','label':'单字组总数'},
                                 {'field':'share','format':'percent','label':'组内比例'}]},
        'valueFormat':'number', 'xAxisTitle':'初审分类', 'yAxisTitle':'样本条数',
        'rationale':'四个互斥类别的精确计数，零基线单系列柱图；不确定不能算错误；不显示置信区间暗示人工共识。',
    })
    manifest['blocks'].append({'id':'single-verdicts-block','type':'chart','chartId':'single-verdicts'})
    paragraph('quality', '## 单字组的主要风险是缺上下文，4 条能明确定位缺陷\n\n'
              '单字组 A+M 为 **137/252（54.37%）**，E 为 **4/252（1.59%）**，U 为 **111/252（44.05%）**。这四条的缺陷在当前输入上可见；不能进一步断言原作者用了错误标点，因为尚未恢复原文。\n\n'
              '“男｜湖北巴东人”“懵｜母亘切”也可能是有效的字段或释义边界。编号、词典和残句应结合原结构判断。一个词典类文档贡献了 27 条，其中 15 条 A、12 条 U；不能把这些行当作独立来源来夸大确定性。普通组未发现明确 E，不代表其真实错误率为零。')
    table('errors', '4 条明确问题及理由', [r for r in rows if r['verdict']=='E'],
          [('review_id','样本',None),('marked_text','代理标签',None),('reason','判断理由',None)], 'review_id')
    paragraph('design', '## 比较已有实验的同一批预测，不重新训练\n\n'
              '“过滤前”与“过滤后”分别使用同一起始 checkpoint，以带位置权重 inverse-cell 和不带位置权重 none 两种配置继续训练一轮。过滤从 1,000,000 条训练样本中删去 2,226 条单字样本，验证 100,000 条保持不变。\n\n'
              '初始化已见过单字监督，所以这不是“从未学习过单字”的从零训练。删除行也把更新次数从 5,329 改为 5,302，并改变批次组合。仅有一个随机种子、沿用已有验证集，不能把小幅指标差异当作稳定的因果效应。')
    paragraph('losses', '## 新增的标签失配，并不主要集中在已确认坏标签上\n\n'
              '带权重模型新增 39 条失配：22 条原断点可接受（A/M），1 条明确坏标签（E），16 条不确定（U）。不带位置权重模型新增 26 条：14 条 A/M、0 条 E、12 条 U；两者单字组均没有新增精确匹配。\n\n'
              '这里的“失配”只是没选中原标签。A/M 行上失配仍可能选中了另一个合理位置，所以下一步必须检查实际预测。E 行上不再匹配坏标签也不自动代表预测正确。')
    table('lost', '单字组：过滤前后原标签匹配数',
          [{**s, 'key': s['mode']+' / '+s['verdict']} for s in f['exact_gold_scores'] if s['cohort']=='single'],
          [('key','模式 / 初审类别',None),('n','样本数','number'),('before','过滤前匹配','number'),
           ('after','过滤后匹配','number'),('newly_wrong','新增失配','number'),('newly_right','新增匹配','number')], 'key')
    paragraph('followup', '## 实际预测同时包含合理改切和明显拆词\n\n'
              '对原断点 A/M 且新增失配的 **29 条去重样本**补做可见模型预测的事后诊断。带权重模型的 22 条中，8 条改切仍可接受、7 条明确变差、7 条不能确定；不带权重的 14 条中，6 条可接受、5 条明确变差、3 条不能确定。两模式样本重叠，不能将分母相加。\n\n'
              '例如“冬｜王师围魏”变成“冬王｜师围魏”拆开“王师”；“如｜第二自然段和第四自然段”改到两个段落之间仍合理。**实际错误与代理答案失配需要分开统计。** 此轮是带预测信息的 AI 事后判断，不能用于宣称盲测语义准确率，也没有回改冻结初审。')
    table('posthoc-counts', '新增失配中的预测质量（仅原 A/M）', f['unblinded_followup']['by_mode'],
          [('mode','位置模式',None),('A','改切可接受','number'),('E','明确变差','number'),('U','无法确定','number')], 'mode')
    follow_rows = []
    for r in follow:
        original = indexed[r['review_id']]
        c = r['after_cut']; t = original['text']
        follow_rows.append(dict(**r, modes_text=', '.join(r['modes']), gold=original['marked_text'], after=t[:c]+'｜'+t[c:]))
    table('posthoc-rows', '29 条事后诊断，保留全部判断', follow_rows,
          [('review_id','样本',None),('modes_text','模式',None),('gold','原标签',None),('after','过滤后预测',None),
           ('after_verdict','结论',None),('reason','理由',None)], 'review_id')
    paragraph('multiple', '## 预先列出的替代位置，也能解释部分“错误”\n\n'
              '普通样本中的 50/200 条被标为 M。它们原标点并没有错，而是存在其他可接受的阅读分块。冻结初审列出的替代位置，在两模式、过滤前后均解释了普通组的 N076 与 N194 两条失配。\n\n'
              '单字组 S012、S174 也出现这种情况。因此，只用单一标点恢复的 top-1 分数会漏掉部分合理答案。但这次没有穷举所有正确切点，不能将“原标签或已列替代位置的命中率”称为完整语义准确率。')
    table('multiple-scores', 'M 类：精确匹配与已列可接受位置的命中数',
          [{**s, 'key':s['cohort']+' / '+s['model']} for s in f['multiple_boundary_scores']],
          [('key','组别 / 模型',None),('n','M 类样本','number'),('exact','原标签命中','number'),
           ('listed_acceptable','原标签或已列替代','number')], 'key')
    paragraph('methods', '## 冻结初审和事后诊断各自保留完整记录\n\n'
              '固定种子抽取普通组，先隐藏预测逐条写分类、理由和替代位置，再保存样本与标注 SHA-256。然后检查四组预测的 targets 与完整验证顺序逐项一致，再做关联。最后单独审查新增失配的 A/M 行，保存事后诊断及其对应切点。\n\n'
              '抽样复现、452 条标注覆盖、切点范围、替代位置合法性、标注冻结值、预测顺序，以及之前单字组 203→164、179→153 的匹配计数均通过检查。前面对部分例子已有接触，因此只称“审查时隐藏预测”，不称严格独立盲审。')
    paragraph('limits', '## 结论适用于这批样本，不能代替人工语义测试\n\n'
              '- 单个 AI 审查者，没有独立人工裁决；U 需原文，E 也应复核。\n'
              '- 输入是规范化后的相邻片段，并未恢复原始网页、排版或更长上下文。保留的 document_id 不是已验证的原文链接。\n'
              '- 252 条只是固定 smoke 验证中的全量单字子集；200 条普通抽样无法精确估计稀有错误。未将二者合并外推全库噪声率。\n'
              '- M 的替代集合不穷尽；A 中也可能有其他合理切法。事后诊断不能回流成独立测试集。\n'
              '- 只审查模型与训练监督，没有用页面渲染或后端过滤掩盖模型问题。')
    paragraph('next', '## 下一步按缺陷核实清理规则，再验证训练收益\n\n'
              '1. 优先人工复核 4 条 E 和 111 条单字 U，依据 document_id 补原文；把词内标点、格式损坏、题号/词条拼接等原因分开。短字、文体或来源本身不作为删除依据。\n'
              '2. 对确认的错误回到样本生成环节修复，并用正常单字例子做反例检查。保留独立的旧验证用于回归，再建立同一质量规范下的人工验证子集；不按某模型是否猜对删验证题。\n'
              '3. 对合理的多切点监督，先建立独立人工可接受位置集合，再比较单标签目标与多可接受位置目标；当前替代清单不是完整训练标注。\n\n'
              '尚待回答：U 中究竟多少来自截断或词条拼接？移除确认坏标签后，在相同训练预算、多随机种子和独立语义验证上是否改善？本轮没有修改正式训练、验证、测试语料或发布模型。')
    paragraph('all', '## 全部 452 条初审可逐条复核\n\n'
              '下表只展示冻结初审。人工复核用 human-review.csv，预留空白 verdict、替代切点和理由列；其中不包含 AI 结论和模型预测。完整来源标识与验证行号保存在 sample.jsonl。')
    table('review', '全部冻结初审记录',
          [{**r, **r['predictions'], 'alternatives': '；'.join(r['text'][:c]+'｜'+r['text'][c:] for c in r['alternative_cuts'])} for r in rows],
          [('review_id','样本',None),('cohort','组别',None),('marked_text','原标签',None),('verdict','初审',None),
           ('reason','理由',None),('alternatives','已列替代切法',None)], 'review_id')
    # The portable reader requires executable SQL lineage. Materialize precisely
    # the reviewed, Python-derived tables, then actually execute every export query.
    database = OUT / 'report-evidence.sqlite'
    if database.exists():
        database.unlink()
    with sqlite3.connect(database) as con:
        con.row_factory = sqlite3.Row
        for name, data in datasets.items():
            assert data, name
            fields = list(data[0])
            assert all(set(r) == set(fields) for r in data), name
            quoted = lambda s: '"' + s.replace('"', '""') + '"'
            con.execute(f'CREATE TABLE {quoted(name)} ({", ".join(quoted(k) for k in fields)})')
            con.executemany(f'INSERT INTO {quoted(name)} VALUES ({", ".join("?" for _ in fields)})',
                            [tuple(r[k] for k in fields) for r in data])
            query = f'SELECT * FROM {quoted(name)}'
            assert [dict(r) for r in con.execute(query)] == data
        source['query'].update(engine='SQLite', language='sql', sql='SELECT * FROM "review"',
                                description=source['query']['description']+' SQLite 导出已计算的精确报告数据，并逐行核对。')
        source['path'] = str(database.relative_to(ROOT))
        for item in manifest['charts'] + manifest['tables']:
            name = item['dataset']
            item_source = copy.deepcopy(source)
            item_source['id'] = 'source-' + name
            item_source['query']['sql'] = f'SELECT * FROM {quoted(name)}'
            item_source['query']['tables_used'] = [name, 'training/artifacts/label-quality-20261004/findings.json']
            manifest['sources'].append(item_source)
            if 'sourceId' in item:
                item['sourceId'] = item_source['id']
            if 'source' in item:
                item['source'] = item_source
    payload = dict(surface='report', manifest=manifest,
                   snapshot=dict(version=1, generatedAt=timestamp, status='ready', datasets=datasets), sources=manifest['sources'])
    write_json(OUT / 'artifact.json', payload)
    write_json(REVIEW / 'findings.json', f)
    return f


def notebook(findings):
    import nbformat as nbf
    nb = nbf.v4.new_notebook()
    nb.metadata['kernelspec'] = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
    nb.cells = [
        md('# 断点标签质量审查\n\n## tl;dr\n\n252 条单字 + 200 条普通样本完成 AI 初审。单字组 137 条可接受、4 条明确问题、111 条不确定。原标点标签有噪声，全部删单字会同时删掉有效监督。**不是人工 gold；事后预测诊断单独统计。**'),
        md('## Context & Methods\n\n种子 2026100417；普通样本从固定100k验证其余99748条中无放回抽200。先隐藏预测作AI初审，冻结后关联4组已存预测。29条原标签可接受但新增失配的样本另作可见预测的诊断。\n\n### Key Assumptions\n\nA不等于唯一答案；M替代位置不穷尽；U不算错。没有恢复原文。前文已有部分例子暴露，非严格独立盲审。不同采样比例不可混合外推。模型是单种子继续训练，并非从零未见单字。'),
        code("from pathlib import Path\nimport sys, json\nroot = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'training/audit_boundary_labels.py').exists())\nsys.path.insert(0, str(root / 'training'))\nfrom audit_boundary_labels import analyze, OUT, REVIEW\n"),
        md('## Data\n\n使用冻结的 sample.jsonl、annotations.jsonl、freeze.json 和两份包含四组预测的 NPZ。重跑依赖原有 ignored smoke artifacts。脚本验证源文件 SHA、标注覆盖与预测 targets 顺序。安装依赖可在独立环境使用 `pip install nbformat nbclient ipykernel numpy`。'),
        code("result = analyze()\nassert all(result['checks'].values())\nprint(json.dumps(result['checks'], indent=2, ensure_ascii=False))"),
        md('## Results\n\n### 1. 冻结标签分布'),
        code("for group in result['review_counts']:\n    assert sum(group[k] for k in ['A','M','E','U']) == group['n']\n    print(group)"),
        md('### 2. 单字组精确标签匹配变化\n\n每行分母按初审类别定义；精确失配不等于语义错误。'),
        code("for row in result['exact_gold_scores']:\n    assert row['after'] - row['before'] == row['newly_right'] - row['newly_wrong']\n    if row['cohort'] == 'single':\n        print({k:v for k,v in row.items() if k not in ['lost_ids','gained_ids']})"),
        md('### 3. 新增失配中的预测质量\n\n只包含原A/M、且从匹配原标签变成失配的29条去重样本。下列是事后AI诊断，不能当独立盲测。'),
        code("for row in result['unblinded_followup']['by_mode']:\n    print(row)\nfor row in result['multiple_boundary_scores']:\n    assert row['exact'] <= row['listed_acceptable'] <= row['n']\n    if row['alternative_hit_ids']:\n        print(row)"),
        md('## Takeaways\n\n按具体错误原因改进数据，保留正常单字监督。先人工核实E/U，并补原文；为多种可接受切法建立独立语义验证。没有修改正式语料、重新训练或发布模型。完整逐条表与来源在 canonical report 和旁边 JSONL 中。'),
        code("assert result == json.loads((REVIEW / 'findings.json').read_text())\nprint('Frozen inputs, independent counts, and report findings reconciled.')"),
    ]
    nbf.validate(nb)
    nbf.write(nb, REVIEW / 'audit.ipynb')


if __name__ == '__main__':
    f = build()
    notebook(f)
    print(OUT / 'artifact.json')
