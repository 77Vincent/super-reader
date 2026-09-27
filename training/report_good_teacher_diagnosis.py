"""Summarize frozen error replay and padding controls without changing labels."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'training/artifacts/good-teacher-diagnosis-v7-20260924'
read = lambda name: json.loads((OUT/name).read_text())
save = lambda name, value: (OUT/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
replay = read('replay.json')
padding = read('padding.json')
rows = replay['rows']
bad = [r for r in rows if r['category'] in ('poor', 'missing_boundary')]
assert len(rows) == 500 and len(bad) == 32
assert padding['summary']['reconstructed_original_padding_matches_historical_top1'] == 500
assert padding['summary']['zero_padding_vs_js_top1_matches'] == 500
assert padding['summary']['masked_pad16_vs_pad0_top1_matches'] == 500
for p, digest in replay['inputs'].items():
    assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == digest, p

# A simple query makes the count denominators explicit and reproducible.
db = sqlite3.connect(OUT/'diagnosis.sqlite3')
db.row_factory = sqlite3.Row
db.execute('DROP TABLE IF EXISTS errors')
db.execute('CREATE TABLE errors(id TEXT PRIMARY KEY, category TEXT, gold_top3 INTEGER, both_full INTEGER, unknown INTEGER, recovered INTEGER, no_cut INTEGER)')
db.executemany('INSERT INTO errors VALUES(?,?,?,?,?,?,?)', [(r['id'], r['category'],
    int(r['storedPrediction']['gold_rank']<=3),
    int(r['goldCoverage']['full'] and r['predictedCoverage']['full']),
    int(bool(r['unknownCharacters'])), int(r['backend']['goldRecovered']),
    int(not r['backend']['cuts'])) for r in bad])
db.execute('DROP TABLE IF EXISTS ranks')
db.execute('CREATE TABLE ranks(id TEXT PRIMARY KEY, rank INTEGER)')
db.executemany('INSERT INTO ranks VALUES(?,?)',[(r['id'], r['storedPrediction']['gold_rank']) for r in bad])
db.commit()
sql = ('SELECT COUNT(*) AS n, SUM(gold_top3) AS gold_top3, SUM(both_full) AS both_full, '
       'SUM(unknown) AS unknown, SUM(recovered) AS gold_recovered, SUM(no_cut) AS no_cut FROM errors;')
counts = dict(db.execute(sql).fetchone())
groups = [dict(r) for r in db.execute('SELECT category,COUNT(*) AS n,SUM(recovered) AS recovered,SUM(no_cut) AS no_cut FROM errors GROUP BY category')]
rank_sql = 'SELECT rank, COUNT(*) AS n FROM ranks GROUP BY rank ORDER BY rank;'
rank_data = [dict(r) | {'label':f"第{r['rank']}名"} for r in db.execute(rank_sql)]
db.close()
assert counts == {'n':32, 'gold_top3':28, 'both_full':22, 'unknown':1, 'gold_recovered':11, 'no_cut':10}
summary = {'scope':'Frozen 500 training rows; fixed 32 previously judged unacceptable errors, not a new random error sample',
           'counts':counts, 'groups':groups, 'padding':padding['summary'],
           'production_model_changed':False, 'training_started':False,
           'reviewer':'Codex; inherited semantic labels, no independent human adjudication',
           'inputs':replay['inputs'], 'cases':[
               {k:r[k] for k in ('id','category','goldSplit','predictedSplit','goldRank','goldProbability','confidence','goldCoverage','predictedCoverage','unknownCharacters')}
               | {'historical':r['storedPrediction'], 'backend':{k:r['backend'][k] for k in ('split','goldRecovered','rawBadCutPresent','cuts')}}
               for r in rows if r['category']]}
tracked = ROOT/'training/good-teacher-diagnosis-20260924.json'
tracked.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
save('summary.json', summary)

intro = ('**更直接的证据指向停顿排序和语义归属判断不足，不能把剩余错误都归因于坏老师。** '
         '在上轮确认不可接受的32条结果中，28条的原正确断点已经排进前三；22条在正确点和所选点都能覆盖完整输入；只有1条包含未知字符。'
         '本轮还发现了独立的工程问题：批量训练/评估的补齐位置会进入卷积计算，导致同一句话与后端逐句推理的结果不完全一致。'
         '它值得先修，但在这批500条上只改变了历史评估与后端的3个最高点，解释不了大部分语义错误。')
scope = ('复用上轮500条分层抽查的实际训练样本，逐行用分片路径、字节偏移、SHA256核验；没有重抽或删除困难样本。'
         '其中492条原教师断点合理，历史批量评估45条不命中；旧逐条判断为15条拆词/紧密结构、17条关键边界遗漏、9条合理替代、4条待定。'
         '本轮32条分母固定为前两类，并非按新预测重新挑选。语义分类沿用既有Codex判断，未经独立人工复核。'
         '这些是训练样本且来源配额不等于总体比例，不能将本批比例称为全库、独立测试或插件实际错误率。')
ranking = ('**正确点常被另一个强烈的局部停顿压过。** 历史32条错误里，正确点排第2的有22条、第3的6条，另4条排第4–6。'
           '这说明它通常能给原断点相对较高的排序，但没有稳定判断哪个停顿对保留整句话的关系更重要。'
           '“本刊为黑白印刷｜图版必须图像清晰”被切为“本刊为黑白印刷图版｜必须图像清晰”，就是局部主谓停顿压过原句界。'
           '“并给出了实例分析｜实例分析结果表明”则被切为“并给出了实例分析实例｜分析结果表明”，重复名词结构造成错误分组。'
           '这是从文本与分数得到的机制解释，不是已读出模型内部的语法表示。')
coverage = ('**“看不全”和“字不认识”都不能概括这一批。** 固定32条中22条在两个比较位置都能看到全文，且31条没有未知字符。'
            '这排除了这些样本缺少感受野外字符或词表缺字的解释，但不等于模型已经学会充分利用上下文，也不证明更大模型永远无效。'
            '例如16字的“幸好博士及時來到馬蒂才免於被勒死”，正确“來到｜馬蒂”只有17.79%，所选“馬蒂｜才免於”有68.97%；全文可见依然发生人名归属错误。')
objective = ('**训练惩罚选错位置，但没有直接验收整套切分后的语义关系。** 当前模型为每个gap生成局部上下文logit，'
             '对所有gap做softmax，以一个原标点位置为标签，使用带样本权重的交叉熵。错误gap确实受到损失惩罚，不能说没有负例约束。'
             '不过它没有显式词组/句法归属任务，也没有针对后端递归后的整段切分做监督。'
             '“常见主谓停顿很强、关键句界更弱”仍可能在共享参数的折中中残留。样本被训练过不代表被逐条记住。'
             '现有证据还不能区分容量、优化、稀有结构覆盖不足及标签冲突各占多少；不能直接宣布3.5M参数不够或坏语料是主因。')
runtime = ('**递归能补一部分，但也会把错误保留下来。** 按当前正式后端（50%门槛、原logit递归softmax、12视觉单位停止、词保护和汉字相邻保护）实际运行全部45条分歧。'
           '固定32条中11条补回了原边界；17条“遗漏关键边界”中只补回6条。10条完全没有新增切分，其中5条原输入本身不超过12视觉单位。'
           '这些集合交叉，不应相加当作互斥类别；补回原边界也不等于整个输出正确。'
           '例如“相较于此｜前几个赛季｜防守能力得到了显著的改善”虽补回正确点，错误的“此｜前”仍然存在。'
           '“幸好博士及時來到馬蒂｜才免於被勒死”切完后两侧都不超过12单位，后端直接结束，不能指望递归补回“來到｜馬蒂”。'
           '这解释实际残留方式，不豁免模型第一次排序错误。')
padding_body = ('**发现并通过隔离实验确认：补齐长度会改变预测。** 权重与真实字符不变，在Metal上只改变右侧补齐0/1/2/16个位置。'
                '500条里，0与2个补齐的最高点有4条不同；重建历史批次的实际补齐长度后，500条最高点全部复现，最大置信度差约8.1e-8。'
                '历史批量评估与当前JS逐句推理是497/500最高点相同；不补齐的Metal与JS则500/500相同。'
                '因此这不是权重导出错误，也不是笼统的CPU/GPU数值差异。\n\n'
                '代码只在整个残差块结束后清零padding。块内部，LayerNorm偏置会使补齐位置变为非零；第一层卷积还会把尾部真实字的信息传入补齐位置，'
                '第二层卷积再把它带回真实字。逐句无补齐推理没有这些中间位置。仅给最终gap加mask不能消除这条路径。\n\n'
                '诊断进程中临时在LayerNorm和第一卷积后清零补齐位置后，补齐16个位置与无补齐的500条最高点全部一致，最大logit差约1.15e-5；'
                '没有把这个变更安装到训练器或后端。示例 wikipedia-0001：无补齐选第15个字后（52.79%），补齐2个位置后选原目标第39个字后（52.02%）。'
                '两种边界条件都可能纠正或制造单例错误，不能把统一处理直接当作准确率提升。当前89.3738%的完整测试分数仍是历史批量口径，未重算为逐句后端口径。')
next_steps = ('**先统一训练、评估与后端的补齐语义，再做语义难例实验。** 第一项需要补充“同一句随batch长度变化”的一致性检查，'
              '随后在固定验证集重测；已有权重能加载，但修正后的继续训练属于行为变化，需要对照，不能承诺旧指标不变。'
              '第二项应在训练划分中收集“正确点位列第2/3、错误点会拆坏词组或语义归属”的难例，人工确认后做小规模重加权/对照，'
              '完整验证集保持固定，并另设未用于训练的语义切分验收样本。不要把这45条直接拿来调参后再报告泛化收益。'
              '当前仍不能量化每种原因对约10%总体错误的贡献。本轮未训练、未换模型、未改线上后端或站点。')

sections = [('发现',intro),('范围与口径',scope),('正确位置存在，但优先级排错',ranking),
            ('感受野和生僻字不是这批错误的主要解释',coverage),('训练目标与实际阅读验收有差别',objective),
            ('真实后端并不能全部补救',runtime),('新发现的补齐一致性问题',padding_body),('下一步与未决问题',next_steps)]
selected = ['wikipedia-0047','wikipedia-0100','web_first-0001','web_first-0047','academic-0009','dialogue-0022']
case_rows = []
for ident in selected:
    r = next(x for x in rows if x['id']==ident)
    case_rows.append({'id':ident,'target':r['goldSplit'],'prediction':r['predictedSplit'],
                     'gold_probability':f"{r['goldProbability']:.2%}",'prediction_probability':f"{r['confidence']:.2%}",
                     'backend':r['backend']['split']})
md = '# 正确教师上的模型错误诊断 — 2026-09-24\n\n'
md += '\n\n'.join('## '+title+'\n\n'+body for title,body in sections)
md += '\n\n## 六个可复核例子\n\n以下分数均为当前JS原始输入的一次softmax，不是校准后的语义正确概率。\n\n'
for r in case_rows:
    md += f"### {r['id']}\n\n原目标：{r['target']}（{r['gold_probability']}）\n\n模型最高点：{r['prediction']}（{r['prediction_probability']}）\n\n正式后端：{r['backend']}\n\n"
md += ('## 复算与证据\n\n'
       '```sh\nnode training/diagnose_good_teacher_errors.mjs\nPYTHONPATH=training/.deps:training python3 training/diagnose_padding.py\npython3 training/report_good_teacher_diagnosis.py\n```\n\n'
       '[结构化结论](good-teacher-diagnosis-20260924.json)；[离线报告](artifacts/good-teacher-diagnosis-v7-20260924/report.html)；'
       '[核算Notebook](artifacts/good-teacher-diagnosis-v7-20260924/analysis.ipynb)。完整logit、真实后端递归轨迹和补齐实验位于同一产物目录。'
       'HTML交付检查若为structural_only，仅表示结构与载荷通过，未运行浏览器视觉检查。\n')
(ROOT/'training/GOOD_TEACHER_DIAGNOSIS.md').write_text(md)
source = {'id':'diagnosis','label':'冻结训练样本的真实后端重放与Metal补齐对照',
          'path':'training/good-teacher-diagnosis-20260924.json',
          'query':{'engine':'SQLite','language':'sql','sql':sql,
                   'description':'对既有32条不可接受错误计数；语义分类来自原复核，非SQL推断。',
                   'tables_used':['training/artifacts/good-teacher-diagnosis-v7-20260924/diagnosis.sqlite3:errors'],
                   'filters':['原审查category为poor或missing_boundary；固定32条，不以新结果重选'],
                   'metric_definitions':['gold_top3=历史批量评估中原目标排前3的数量','both_full=原目标与当前所选位置均覆盖全文','recovered=后端新增切分中包含原目标；不等于全输出合格']}}
blocks = [{'id':'title','type':'markdown','body':'# 正确教师上的模型错误诊断'}]
for i,(title,body) in enumerate(sections):
    blocks.append({'id':f'section-{i}','type':'markdown','body':f'## {title}\n\n{body}','sourceId':'diagnosis'})
    if i == 2:
        blocks.append({'id':'ranks','type':'chart','chartId':'ranks','sourceId':'ranking'})
blocks.append({'id':'cases','type':'table','tableId':'cases','sourceId':'diagnosis'})
rank_source = {'id':'ranking','label':'32条固定错误的原正确点排名','path':'training/good-teacher-diagnosis-20260924.json',
               'query':dict(source['query']) | {'sql':rank_sql,'tables_used':['training/artifacts/good-teacher-diagnosis-v7-20260924/diagnosis.sqlite3:ranks'],
                                                'metric_definitions':['rank=历史批量原目标排名','n=固定32条中各排名条数']}}
chart = {'id':'ranks','type':'bar','title':'28/32条错误的正确点已排进前三','dataset':'ranks','intent':'comparison',
         'question':'已判不可接受的32条中，正确断点被排在第几名？','rationale':'按有序名次展示条数，零起点、单序列；不推断总体比例。',
         'description':'2026-09-24冻结审查；n=32条训练样本；单位：条；历史批量评分。','showDescription':True,
         'encodings':{'x':{'field':'label'},'y':{'field':'n'}},'sourceId':'ranking'}
artifact = {'surface':'report','manifest':{'version':1,'surface':'report','title':'正确教师上的模型错误诊断','blocks':blocks,'sources':[source,rank_source],
            'charts':[chart],
            'tables':[{'id':'cases','title':'六个固定错误样本：原目标、当前模型分数与真实后端结果','dataset':'cases','sourceId':'diagnosis',
                       'columns':[{'field':k,'label':v} for k,v in [('id','样本'),('target','原目标'),('prediction','模型最高点'),('gold_probability','原目标softmax'),('prediction_probability','最高点softmax'),('backend','正式后端')]]}]},
            'snapshot':{'version':1,'status':'ready','datasets':{'cases':case_rows,'ranks':rank_data}},'sources':[source,rank_source]}
save('artifact.json',artifact)
save('report-notes.json',{'format':'technical report, portable HTML','chart_decision':'Ordered single-series count bars for five gold-rank categories, n=32, zero baseline, shared single-root palette; 8-column report footprint. Exact sentence comparisons remain a table. Packaging enforces a chart block.',
                        'structure':'Summary, definitions, methods and causal limits, observed mechanisms, controlled padding experiment, next steps; sections combined where evidence and interpretation need to remain adjacent.',
                        'analysis_QA':'Input/source SHA256, exact 500-row training-byte checks, unchanged model hash, all original padded top1 reproduced, Metal/JS 500-case agreement, traced/uninstrumented backend parity on all45; fixed cohort maintained.',
                        'remaining_limits':['No independent human adjudication','No population error estimate','Full unpadded validation/test not rerun','The 2 newly wrong JS rows have not received the original semantic classification']})

code = '''from pathlib import Path
import json, hashlib
root = Path.cwd()
if not (root/'training').exists(): root = root.parents[2]
out = root/'training/artifacts/good-teacher-diagnosis-v7-20260924'
r = json.loads((out/'replay.json').read_text())
p = json.loads((out/'padding.json').read_text())
for path, digest in r['inputs'].items():
    assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest
bad = [x for x in r['rows'] if x['category'] in ('poor','missing_boundary')]
assert len(r['rows'])==500 and len(bad)==32
print('Fixed unacceptable cohort:',len(bad))
print('Historical gold top3:',sum(x['storedPrediction']['gold_rank']<=3 for x in bad))
print('Both compared positions see whole input:',sum(x['goldCoverage']['full'] and x['predictedCoverage']['full'] for x in bad))
print('Unknown characters:',sum(bool(x['unknownCharacters']) for x in bad))
print('Backend recovered original boundary:',sum(x['backend']['goldRecovered'] for x in bad))
print('Backend no new cuts:',sum(not x['backend']['cuts'] for x in bad))
assert p['summary']['reconstructed_original_padding_matches_historical_top1']==500
assert p['summary']['masked_pad16_vs_pad0_top1_matches']==500
print('Padding controls:',p['summary'])
'''
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    exec(compile(code,'good-teacher-diagnosis','exec'))
def markdown(ident, text):
    return {'id':ident,'cell_type':'markdown','metadata':{},'source':text.splitlines(keepends=True)}
nb = {'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}},'cells':[
    markdown('summary','# TL;DR\n'+intro),markdown('methods','## Context & Methods\n'+scope),
    markdown('data','## Data\nScores and hashes: replay.json; Metal padding controls: padding.json. No training performed.'),
    {'id':'results','cell_type':'code','metadata':{},'execution_count':1,'source':code.splitlines(keepends=True),
     'outputs':[{'output_type':'stream','name':'stdout','text':stdout.getvalue().splitlines(keepends=True)}]},
    markdown('takeaways','## Takeaways\n'+next_steps)]}
save('analysis.ipynb',nb)
print(stdout.getvalue())
