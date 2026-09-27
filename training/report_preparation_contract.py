"""Write the evidence-backed preparation-contract diagnosis and companion report."""
from collections import Counter
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sqlite3

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'training/artifacts/data-preparation-audit-v7-20260925'
read=lambda name:json.loads((OUT/name).read_text())
save=lambda name,x:(OUT/name).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
s=read('structural.json');r=read('runtime-contract.json')
run=ROOT/'training/artifacts/learning-rate-192ch-full-246m-v7-20260923'
manifest=json.loads((run/'manifest.json').read_text());plan=json.loads((run/'run.json').read_text())
lineage={p:{'expected_sha256':h,
            'frozen_hash_matches':hashlib.sha256((run/'source'/p).read_bytes()).hexdigest()==h,
            'current_matches_frozen':hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h}
         for p,h in plan['source_hashes'].items()}
save('code-lineage.json',lineage)
assert s['manifest_sha256']==hashlib.sha256((run/'manifest.json').read_bytes()).hexdigest()
assert all(v['frozen_hash_matches'] and v['current_matches_frozen'] for v in lineage.values())
assert [v['rows'] for v in s['stats']]==[3781496,2522347,2569996]
for rec,name in zip(s['files'][-2:],['validation','test']):
    assert rec['sha256']==plan['evaluation'][name]['sha256']
from text_policy import require_data_policy
policy_paths={run/'manifest.json',Path(manifest['evaluation_source_dir'])/'summary.json'}
policy_paths.update((ROOT/sh['path']).parent/'manifest.json' for sh in manifest['shards'])
for p in policy_paths:require_data_policy(json.loads(p.read_text()))
v=json.loads(Path(manifest['vocabulary_path']).read_text())
assert v==json.loads((run/'epoch-1-backend/boundary-smoke-vocabulary.json').read_text())
assert sorted(v.values())==list(range(len(v)))
prior_path=ROOT/'training/artifacts/v7-conservative-loss-20260917/results.json'
prior=json.loads(prior_path.read_text())['profiles']['clue']['domains']['news']
assert prior=={'v6_pairs':65243,'v7_pairs':10486}

ratio=s['position_weight_table'][1][0]/s['position_weight_table'][1][5]
rate=lambda values,cut:sum(n for k,n in values.items() if int(k)>cut)/sum(values.values())
length_effect={'row_share_above32':rate(s['length_ceilings'][0],32),
               'nominal_weight_share_above32':rate(s['nominal_weight_mass_by_length'],32),
               'batch_coefficient_share_above32':rate(s['actual_batch_coefficient_mass_by_length'],32),
               'row_share_above256':rate(s['length_ceilings'][0],256),
               'batch_coefficient_share_above256':rate(s['actual_batch_coefficient_mass_by_length'],256)}
summary={'date':'2026-09-25','scope':s['scope'],'training_rows_checked':s['stats'][0]['rows'],
         'validation_rows_checked':s['stats'][1]['rows'],'test_rows_checked':s['stats'][2]['rows'],
         'invalid_labels_or_metadata':0,'repeated_exact_inputs_in_checked_union':s['duplicates'].get('repeated_inputs',0),
         'conflicting_exact_inputs_in_checked_union':s['duplicates'].get('conflicting_inputs',0),
         'cross_split_exact_inputs_in_checked_union':s['duplicates'].get('cross_split_inputs',0),
         'position_weight_edge_vs_center_9_to16':ratio,'length_effect_in_32_selected_shards':length_effect,
         'runtime_rates_standardized_from_24000_training_rows':r['populationStandardizedRates'],
         'policy_manifests_checked':len(policy_paths),'vocabulary_ids_preserved':True,
         'source_lineage':lineage,'frozen_holdout_hashes_match':True,
         'prior_news_coverage_study':dict(prior,source=str(prior_path.relative_to(ROOT)),sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest()),
         'data_preparation_regression_tests':{'passed':40,'failed':0},
         'causal_contribution_to_accuracy_plateau_measured':False,
         'full_training_duplicate_scan':False,'raw_document_near_duplicate_scan':False,
         'training_or_production_changed':False}
for st in s['stats']:
    assert not(set(st)-{'rows','tokens','length_above_256'}),st
assert not s['duplicates']
(ROOT/'training/data-preparation-audit-20260925.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
save('summary.json',summary)

sections=[
('结论',
 '**有需要处理的工程缺陷和训练设计偏差，但没有发现“标签大面积生成错位”或“90%就是数据硬上限”的证据。** '
 '当前最确定的工程缺陷仍是前次确认的padding不一致。对离线准确率最值得单独验证的是很强的位置权重与隐含长度权重；'
 '对插件实际效果，训练和后端输入/候选点口径不一致、缺少不切分监督及递归监督也值得处理。每项影响多少，尚未通过受控训练测出。'),
('检查范围与通过项',
 '本轮顺序扫描完整validation 2,522,347条、test 2,569,996条，以及种子2026092501抽取的32个完整训练分片，共3,781,496条训练样本；合计8,873,839条。'
 '训练分片从Wikipedia基础块、合成块、两批网页块各抽8个，抽样比例不同，不能把这378万条的组成当成全库比例。'
 '另用已有24,000条来源分层概率样本检查后端输入，并按全训练库各来源占比加权报告比率。\n\n'
 '本次范围内：标签越界、字符token粒度错误、长度/位置分箱错位、NFKC或空格契约偏离均为0；完整holdout代理标签合法。'
 '按输入文本SHA256检查，扫描范围内重复输入、相同输入不同标签、跨split相同输入均为0。此结论不覆盖未扫的242M余下训练行、近重复文档或语义近义冲突。'
 '两个完整holdout哈希与正式训练冻结记录一致；8192词表的ID连续，准备端与导出端相同；6份策略元信息通过当前契约检查；11份训练/策略源码与冻结版本哈希一致。'
 '40项已有数据准备/跨语言/屏障/Unicode回归测试通过。本轮不是再次人工复核887万条老师的语义。'),
('高优先级：样本虽合理，权重仍会改变所学偏好',
 f'9–16字符长度组，边界位于开头0–10%区间的样本权重约88.15，中间50–60%约0.3956，同组相差 **{ratio:.1f}倍**。'
 '这是length-position inverse-cell默认规则，目的是拉平位置分布，不是标签索引错误。有效“老师”数量很多，也不意味着按它们的自然频率学习。'
 '来源权重还会相乘，例如news为424.80，web为0.578；单条同位置样本的名义来源系数相差约735倍，但新闻样本少，不能据此声称新闻主导了训练。'
 '梯度裁剪、Adam和样本内容又会改变实际影响，这些数值不是最终梯度占比。\n\n'
 '分桶按8192-token预算决定batch条数，loss却按当前batch条数求均值。因此长句batch更小，单条记录在每一步损失中的系数更大。'
 f'在这32个分片中，>32字符的记录占{length_effect["row_share_above32"]:.2%}，显式权重质量占{length_effect["nominal_weight_share_above32"]:.2%}，'
 f'按实际batch逐条计算weight/batch_size/全局均值后，其系数质量占{length_effect["batch_coefficient_share_above32"]:.2%}。'
 '这验证了隐含长度加权机制；不是全库精确占比，也不是实际参数更新份额或准确率因果效应。代码未截断长样本；少量样本长于后端256-token窗口，训练全局竞争与部署分窗又不同。\n\n'
 '建议先在固定数据、初始化、学习率和步数下，仅取消位置逆频权重做对照；隐含长度权重另做独立实验。不要同时改两者后把收益归给其中一个。'),
('高优先级：训练与后端口径并未完全对齐',
 '24,000条概率样本按全库来源加权估计：18.71%的训练AB输入在当前后端会被拆成多个子句；13.26%含顿号，7.97%含ASCII形式的代理字符，两者有交集。'
 '6.83%的训练输入经后端tokenizer后字符序列会改变。训练只认原始中文代理，保留ASCII句点/逗号等；后端却把更多标点当作预分句边界。'
 '例如训练输入中的“DineEquity Inc.”或“Mk.VIII”保留句点，而后端会提前切开。是否允许这种产品行为需与数据契约一起明确。\n\n'
 '7.08%的教师断点至少一侧紧邻非汉字，训练允许，后端要求两侧都是汉字而拒绝；例如“开始建造工作｜1962年8月26日正式建成”。'
 '约0.46%的教师断点会落入Intl分词保护范围；11.28%的整个训练输入本身不超过12视觉单位，后端直接不处理。上述类别重叠，不能相加为总损失。'
 '这些差异主要解释训练任务与部署任务错位，**不能把18.71%或7.08%当成离线验证错误率的来源份额**。清洁但部署永远不用的目标，也会消耗模型学习能力，实际是否损害其余样本需对照。'),
('任务设计：仅有A+B和一个断点，缺少另两种监督',
 '实际生成器只保存相邻A、B，不保留之前或之后的语境；每条恰好一个target_index，所有其他gap在该条softmax损失中都与目标竞争。'
 '原标点合理不代表AB在去掉它后仍唯一可判，也不代表剩余上下文足以消除人名、修饰语、并列关系的歧义。当前扫描没发现同输入显式冲突，仍不能排除不同语境投影到近似输入后的任务歧义。\n\n'
 '此外没有“整个片段已经足够完整，不必再切”的样本/输出类别，也没有在同一输入上监督多个合理边界或递归最终结果。'
 '这不是one-hot索引实现错误；它定义的是“还原一个原标点”，与插件“只在必要时切出自然语意块”的目标存在差别。'
 '用户允许其他合理停顿，但拆坏词组或造成跨句归属错误仍不合格：此前45个合格教师上的分歧只有9个被评为合理替代，另32个明确不可接受。'
 '不能用多解给这32条开脱，也不能把新增不切分监督直接当作现有top1指标必然突破90%的方法。'),
('已确认的缺陷：padding影响真实字符',
 '前次500条控制实验已确认：训练器只在残差块之后清零padding，块内部LayerNorm/卷积的补齐位置仍参与计算。'
 '重建历史batch后500条全部复现；无补齐Metal与JS全部一致；历史批量结果与逐句后端有3条最高点不同。'
 '这是batch整理和模型前向的衔接缺陷，不是原始老师质量问题。修正后需先重测固定验证集，再评估是否继续训练；不能承诺原分数原样保持或直接增长若干百分点。'
 '完整test 89.3738%仍是原批量口径，尚未按统一padding语义重算。'),
('覆盖偏移与可追溯性遗漏',
 '已同意的保守清洗会系统性损失部分文体，更多同样数据不能自动补齐被规则一律排除的样式。既有同池v6/v7比较中，CLUE新闻标题候选65,243→10,486，减少83.93%；'
 '“A，B。”因每行首片段弃用而完全不产样本，“A，B，C。”只产B｜C。该比例来自之前缓存池研究，不是本轮全库损耗估算。'
 '这属于明确接受过的质量/覆盖取舍，未擅自放宽。可针对确认可靠的短标题来源单独研究规则，而不是取消全局屏障。\n\n'
 'compact训练行只存[text,target,domain,length_bucket,position_bin]，没有原proxy类别、原始document_id、网页子来源/原文位置。'
 '这不直接改变前向结果，但使标点类型、来源偏移、同文近重复和错误原文回溯困难；需要抽查时要回查原文库。后续准备可增加独立元数据索引，不必把这些字段喂给模型。'
 '新增网页采用与target无关的汉字投影去重；旧来源代码仍主要按(text,target)去重。旧逻辑在设计上允许同输入不同标签，但本次扫描未发现实际冲突，不能将该风险当作现存大规模缺陷。'),
('下一步与结论边界',
 '建议顺序：①统一padding并补batch长度不变性检查；②固定现有数据做位置权重消融；③统一训练/部署的标点、候选点与停止任务口径，分别比较离线还原率和人工语义切分质量；'
 '④再据错误集中在哪些结构，补针对性语境/多边界/不切分监督。每次实验只改变要检验的变量。'
 '本轮没有证明90%为不可突破的上限，也没有证明清洗再严、数据更多或参数更大能单独突破它。未修改准备策略、训练器、模型权重、后端或站点，未启动新训练。')]

md='# 数据准备与训练任务契约审计 — 2026-09-25\n\n'+'\n\n'.join('## '+t+'\n\n'+b for t,b in sections)+'\n\n'
md+='## 可复核材料\n\n[结构化结论](data-preparation-audit-20260925.json)、[完整HTML报告](artifacts/data-preparation-audit-v7-20260925/report.html)、[执行过的Notebook](artifacts/data-preparation-audit-v7-20260925/analysis.ipynb)。\n\n'
md+='```sh\npython3 training/audit_preparation_contract.py\nnode training/audit_preparation_runtime.mjs\npython3 training/report_preparation_contract.py\n```\n\n'
md+='相关源码：[位置和来源权重](train_sharded.py)、[batch分桶与padding](train_smoke.py)、[相邻样本与屏障](text_policy.py)、[后端输入契约](../src/backend/chunker.js)。\n'
md+='既有证据：[padding隔离实验](GOOD_TEACHER_DIAGNOSIS.md)、[保守清洗覆盖损耗](CONSERVATIVE_CLEANING_V7.md)。HTML仅通过结构/载荷校验，未做浏览器视觉检查。\n'
(ROOT/'training/DATA_PREPARATION_AUDIT.md').write_text(md)

db=sqlite3.connect(OUT/'audit.sqlite3');db.row_factory=sqlite3.Row
db.execute('DROP TABLE IF EXISTS position_weights');db.execute('CREATE TABLE position_weights(bin INTEGER, label TEXT, weight REAL)')
db.executemany('INSERT INTO position_weights VALUES(?,?,?)',[(i,f'{i*10}–{(i+1)*10}%',w) for i,w in enumerate(s['position_weight_table'][1])]);db.commit()
sql='SELECT bin, label, weight FROM position_weights ORDER BY bin;';data=[dict(v) for v in db.execute(sql)];db.close()
source={'id':'audit','label':'当前v7准备/训练/部署契约：原文件扫描与固定样本对照','path':'training/data-preparation-audit-20260925.json'}
weights={'id':'weights','label':'正式manifest的9–16字符组位置逆频权重','path':'training/artifacts/data-preparation-audit-v7-20260925/structural.json',
         'query':{'engine':'SQLite','language':'sql','sql':sql,'description':'已从正式manifest cells重算的权重，按原位置区间排序。',
                  'tables_used':['training/artifacts/data-preparation-audit-v7-20260925/audit.sqlite3:position_weights'],
                  'filters':['length=9..16 Unicode code points'],
                  'metric_definitions':['weight=同长度组总条数/非空位置分箱数/当前箱条数','不含来源权重、全局归一化、batch大小、裁剪和Adam影响']}}
blocks=[{'id':'title','type':'markdown','body':'# 数据准备与训练任务契约审计'}]
for i,(title,body) in enumerate(sections):
    blocks.append({'id':f'section-{i}','type':'markdown','body':'## '+title+'\n\n'+body,'sourceId':'audit'})
    if i==2:blocks.append({'id':'position-weights','type':'chart','chartId':'position-weights','sourceId':'weights'})
artifact={'surface':'report','manifest':{'version':1,'surface':'report','title':'数据准备与训练任务契约审计','blocks':blocks,'sources':[source,weights],
         'charts':[{'id':'position-weights','type':'bar','dataset':'weights','title':'9–16字符样本的位置权重明显偏向边缘',
                    'description':'当前正式v7训练manifest；x=断点在输入中的相对位置；y=位置损失权重，单位：倍；不代表实际梯度贡献。',
                    'showDescription':True,'intent':'comparison','question':'不同位置的正确教师获得多大位置系数？',
                    'rationale':'10个有序区间、单序列柱形图、零基线，使用共享单色主题；不把名义权重当因果效果。',
                    'encodings':{'x':{'field':'label'},'y':{'field':'weight'}},'sourceId':'weights'}]},
         'snapshot':{'version':1,'status':'ready','datasets':{'weights':data}},'sources':[source,weights]}
save('artifact.json',artifact)
save('report-notes.json',{'format':'technical portable report','structure':'Summary, checks/methods, prioritized confirmed facts vs unmeasured causes, source coverage and lineage, remediation and limitations.',
     'chart_contract':'Ordered ten-bin, single-series bar chart of position coefficients; zero baseline, shared single-root palette; standard report width; exact lookup and interpretation attached.',
     'QA':'Counts and holdout hashes reconciled; fixed/frozen code identical; policy and vocabulary verified; 40 data tests passed. No full training or raw-document near-duplicate certification.',
     'causality':'No contribution of any preparation design to the 90% plateau has been measured by retraining.'})

code='''from pathlib import Path
import json
root=Path.cwd()
if not (root/'training').exists():root=root.parents[2]
out=root/'training/artifacts/data-preparation-audit-v7-20260925'
s=json.loads((out/'structural.json').read_text());r=json.loads((out/'runtime-contract.json').read_text())
assert [x['rows'] for x in s['stats']]==[3781496,2522347,2569996]
assert sum(x['rows'] for x in s['stats'])==8873839
assert not s['duplicates']
print('Scanned rows:',sum(x['rows'] for x in s['stats']))
print('Position coefficient edge/center:',s['position_weight_table'][1][0]/s['position_weight_table'][1][5])
for label,values in [('rows',s['length_ceilings'][0]),('nominal weights',s['nominal_weight_mass_by_length']),('batch coefficients',s['actual_batch_coefficient_mass_by_length'])]:
    print('Selected-shard >32 share,',label, sum(v for k,v in values.items() if int(k)>32)/sum(values.values()))
print('Source-standardized rates from 24k sample:',r['populationStandardizedRates'])
print('No statement of full-training duplicate absence or measured causal accuracy gain.')
'''
capture=io.StringIO()
with contextlib.redirect_stdout(capture):exec(compile(code,'preparation-audit-accounting','exec'))
def markdown(ident,text):return {'id':ident,'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)}
save('analysis.ipynb',{'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},'cells':[
    markdown('summary','# TL;DR\n'+sections[0][1]),markdown('context','## Context & Methods\n'+sections[1][1]),
    markdown('data','## Data\nstructural.json: full holdouts +32 training shards. runtime-contract.json: frozen probability sample. code-lineage.json: frozen/current code hashes.'),
    {'id':'results','cell_type':'code','metadata':{},'execution_count':1,'source':code.splitlines(keepends=True),
     'outputs':[{'output_type':'stream','name':'stdout','text':capture.getvalue().splitlines(keepends=True)}]},
    markdown('takeaways','## Takeaways\n'+sections[-1][1])]})
print(capture.getvalue())
