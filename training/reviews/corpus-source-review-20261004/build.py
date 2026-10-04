"""Rebuild the bounded corpus-source review; never loads corpus shards or trains."""
from pathlib import Path
import ast
import contextlib
import datetime
import hashlib
import io
import json
import sqlite3

ROOT = Path(__file__).resolve().parents[3]
REVIEW = Path(__file__).resolve().parent
OUT = ROOT / 'training/artifacts/corpus-source-review-20261004'
OUT.mkdir(parents=True, exist_ok=True)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


# Standard-library execution keeps this notebook runnable without training/Jupyter
# dependencies. Outputs below are captured from execution, never fabricated.
cells = []
namespace = {'__name__': '__main__'}
execution_count = 0


def md_cell(text):
    cells.append(dict(id=f'cell-{len(cells)}', cell_type='markdown', metadata={},
                      source=text.splitlines(keepends=True)))


def code_cell(text):
    global execution_count
    execution_count += 1
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exec(compile(text, f'corpus-review-cell-{execution_count}', 'exec'), namespace)
    cells.append(dict(id=f'cell-{len(cells)}', cell_type='code', metadata={},
                      source=text.splitlines(keepends=True), execution_count=execution_count,
                      outputs=[dict(output_type='stream', name='stdout',
                                    text=stdout.getvalue().splitlines(keepends=True))]))


md_cell('# Chinese Corpus Review\n\n## tl;dr\n\n现用 Ultra-FineWeb 的固定版本与本次查询的上游 main 一致。现有抽取器读取 content/source，未按 score 进一步筛选。FineWeb2-HQ 中文是首个外部对照候选；没有完成新语料的断点质量抽样或模型实验，不能宣称优于现用数据。')
md_cell('## Context & Methods\n\n2026-10-04，核对作者数据卡、版本元数据、本地清单和抽取器。原始网页证据位于 evidence/，index.json 记录固定 revision 和数据卡 SHA256。\n\n### Key Assumptions\n\n本文“质量”是文本完整性与抽取后切分标签的可接受性；不是知识测验成绩。历史 v7 计数尚未应用新的单字过滤。没有用数据集卡规模推断可保留样本数。\n\n环境：Python 3 标准库。build.py 按顺序执行下列代码并捕获输出；未使用 Jupyter 内核，不需要模型依赖或网络。')
code_cell("""from pathlib import Path
import ast, hashlib, json
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'training/train_sharded.py').exists())
review = root/'training/reviews/corpus-source-review-20261004'
manifest_path = root/'training/data/processed/padding-fixed-web-350m-v7-20260925/manifest.json'
manifest = json.loads(manifest_path.read_text())
index = json.loads((review/'evidence/index.json').read_text())
assert len(index) == len({x['repository'] for x in index}) == 10
print('Dataset repositories inspected:', len(index))
""")
md_cell('## Data\n\n检查已保存证据的完整性。门控仓库的原始 README 下载被拒绝，明确记录，不将其描述为已下载语料。')
code_cell("""verified = 0
for item in index:
    if 'readme_sha256' in item:
        path = review/'evidence'/(item['repository'].replace('/', '--')+'.md')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['readme_sha256']
        verified += 1
    else:
        assert item.get('error') and item.get('gated')
print('Pinned README hashes verified:', verified)
print('Gated README downloads unavailable:', len(index)-verified)
""")
md_cell('## Results\n\n### 当前版本、抽取字段与样本份额\n\n检查代码实际传给 Parquet 的列；并不把“未进一步使用分数”误解成“上游完全没筛选”。')
code_cell("""upstream = next(x for x in index if x['repository'] == manifest['web_source']['repository'])
assert manifest['web_source']['revision'] == upstream['revision']
tree = ast.parse((root/'training/prepare_web_data.py').read_text())
columns = [ast.literal_eval(k.value) for n in ast.walk(tree)
           if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'iter_batches'
           for k in n.keywords if k.arg == 'columns']
assert columns == [['content', 'source']]
counts = manifest['statistics']['domain_samples']  # historical schema, not semantic domains
total = sum(counts.values())
assert total == manifest['statistics']['samples'] == 396266210
assert counts['web'] == 350000000
print('Pinned Ultra-FineWeb revision equals observed upstream:', upstream['revision'])
print('Parquet columns:', columns)
print('Historical training rows:', total)
print('Historical web rows / all rows: %.4f%%' % (100*counts['web']/total))
""")
md_cell('### 文档证据核验\n\n仅检查候选子集、字段及授权说明确实出现在作者数据卡；不是对样本语义的自动评分。')
code_cell("""def card(name):
    return (review/'evidence'/(name.replace('/', '--')+'.md')).read_text()
hq = card('epfml/FineWeb2-HQ')
assert all(s in hq for s in ['cmn_Hani', '54,211,986', 'quality_score', 'top 10%', 'ODC-By'])
edu = card('opencsg/Fineweb-Edu-Chinese-V2.1')
assert all(s in edu for s in ['17,790,513', '70 GB', '获得许可', '46 billion', '46 亿'])
assert 'CASIA-LM/ChineseWebText2.0' in card('openbmb/Ultra-FineWeb')
print('FineWeb2-HQ Chinese subset and quality/provenance fields: confirmed in card')
print('Edu V2.1: commercial-permission clause present; bilingual token totals conflict')
print('Ultra-FineWeb upstream card already links ChineseWebText2.0')
print('No new corpus quality rates or model results measured.')
""")
md_cell('## Takeaways\n\n保留现用语料作为基线，加入它自身的高分子集，与 FineWeb2-HQ 中文作独立对照；Edu V2.1 高分档需先澄清商用条件。跨库先去重，再用同一抽取器和两侧至少两字规则检查标签，不恢复来源或位置加权。只有来源配额统计，不将来源标签当语义分类。全量训练仍暂停。')
notebook = dict(nbformat=4, nbformat_minor=5,
                metadata=dict(kernelspec=dict(name='python3', display_name='Python 3', language='python'),
                              execution_note='Code cells executed sequentially by Python standard library; no Jupyter kernel.'),
                cells=cells)
assert len({c['id'] for c in cells}) == len(cells)
assert [c['execution_count'] for c in cells if c['cell_type'] == 'code'] == list(range(1, execution_count+1))
for cell in cells:
    if cell['cell_type'] == 'code':
        ast.parse(''.join(cell['source']))
        assert cell['outputs'][0]['output_type'] == 'stream'
save(REVIEW/'research.ipynb', notebook)

index = namespace['index']
timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
inventory = {entry['repository']: entry for entry in index}
sources = []


def source(key, label, path, **extra):
    value = dict(id=key, label=label, path=path, query=dict(
        engine='Document review / Python 3', language='python', executed_at=timestamp,
        description='作者资料、固定版本及本地代码核验；没有新样本质量测量或模型训练。', **extra))
    sources.append(value)
    return value


source('local', '本地 v7 清单与网页抽取器', 'training/data/processed/padding-fixed-web-350m-v7-20260925/manifest.json',
       tables_used=['training/prepare_web_data.py', 'training/download_ultrafineweb.py'],
       filters=['历史计数：单字过滤前'],
       metric_definitions=['web share = web rows / all training rows; not a quality metric'])
source('old-audit', '历史来源标签抽查，AI 初审', 'training/SOURCE_TEACHER_AUDIT_R2.md',
       tables_used=['training/SOURCE_TEACHER_AUDIT.md'],
       filters=['来源配额抽文档，非新候选数据集的独立随机样本'])
ids = {'openbmb/Ultra-FineWeb':'ultra', 'epfml/FineWeb2-HQ':'hq',
       'HuggingFaceFW/fineweb-2':'fw2', 'opencsg/Fineweb-Edu-Chinese-V2.1':'edu',
       'opencsg/Fineweb-Edu-Chinese-V2.2':'edu22', 'opencsg/Fineweb-Edu-Chinese-V3':'edu3',
       'CASIA-LM/ChineseWebText2.0':'cwt', 'BAAI/CCI3-HQ':'cci',
       'BAAI/CCI4.0-M2-Base-v1':'cci4', 'HuggingFaceFW/finewiki':'wiki'}
for repo, key in ids.items():
    entry = inventory[repo]
    source(key, repo, 'https://huggingface.co/datasets/'+repo,
           revision=entry['revision'], observed_at=entry['observed_at'],
           source_notes=('公开数据卡与 API 元数据；原始 README 下载 HTTP 401，未访问数据。' if entry.get('error')
                         else '固定版本数据卡及 SHA256 保存在研究附件。'),
           access_issue=entry.get('error', 'none'))
source('comparison', '基于作者资料的候选用途比较', 'training/reviews/corpus-source-review-20261004/evidence/index.json',
       tables_used=list(inventory), source_notes='Qualitative candidate table, not a measured quality ranking. No candidate-size chart because units and tasks are incomparable. One zero-baseline single-series bar describes the existing corpus by source only, not its quality. Technical structure: summary, evidence, scope/methods, uncertainty, experiment, open questions. Portable HTML; no publication.')

title = 'Chinese Corpus Review'
manifest = dict(version=1, title=title, generatedAt=timestamp, sources=sources, blocks=[], tables=[], charts=[])


def section(key, body, source_id=None):
    block = dict(id=key, type='markdown', body=body)
    if source_id:
        block['sourceId'] = source_id
    manifest['blocks'].append(block)


section('title', '# '+title)
section('summary', '## 有值得试的候选，但还没有证明更好的替代库\n\n建议先比较 **现用 Ultra-FineWeb、它自身的高分子集、FineWeb2-HQ 中文子集**。Chinese Fineweb Edu V2.1 高分档可作为授权条件确认后的候选；FineWiki 更适合作为百科来源补充。\n\n这是截至 2026-10-04 的作者资料与本地实现调研，优先级是我的研究判断，不是测得的模型排名。我们需要的是抽取后可靠的切分标签；上游的知识性评分和大模型测验成绩只能提供间接证据。')
section('baseline', '## 现用网页数据已筛选，升级版本不会带来新数据\n\n历史清单有 396,266,210 条训练样本，其中网页 350,000,000 条，约 88.32%；这是单字过滤前的计数。下载固定版本与本次上游 main 的 SHA 一致。当前抽取器只读 content/source，执行本地清理，但未利用上游 score 做进一步筛选。\n\n因此，先检查同一语料不同分数段的标签质量，比直接换整库成本更低；分数越高是否越适合切分仍须验证。', 'local')
source_labels = {'web':'Ultra-FineWeb 网页', 'wikipedia':'维基百科',
                 'synthetic_multistyle':'合成语料', 'news':'CLUE TNEWS',
                 'academic':'CLUE CSL', 'encyclopedia':'CLUE CMRC2018', 'dialogue':'CLUE C3'}
mix_rows = [dict(source=source_labels[key], rows=count, share=count/namespace['total'])
            for key, count in sorted(namespace['counts'].items(), key=lambda item: -item[1])]
manifest['charts'].append(dict(id='existing-source-mix', title='历史训练样本来源占比（单字过滤前）',
                              type='bar', dataset='source_mix', sourceId='local', valueFormat='percent',
                              encodings=dict(x=dict(field='source', type='nominal', label='语料来源'),
                                             y=dict(field='share', type='quantitative', label='训练行占比')),
                              rationale='Seven source categories, zero-baseline single-series bar. Historical manifest rows divided by all training rows; not quality or semantic-domain classification. Candidate comparison remains a table.'))
manifest['blocks'].append(dict(id='source-mix-chart', type='chart', chartId='existing-source-mix'))
section('mix-interpretation', '网页构成现有训练样本的主要部分，因此这次筛选决策影响面大。图中只是采集得到的样本配比，不能当成用户阅读场景分布，也不表示各来源的质量。')
section('overlap', '## 很多看似替代品，其实已是当前语料的上游\n\nUltra-FineWeb 中文来自经过再次筛选的 Chinese Fineweb-edu-v2，包含 CCI3、SkyPile、WanJuan 等，作者还直接链接 ChineseWebText2.0。它已经是筛选结果，换回这些原始来源不天然更干净，也不能假定新增覆盖。数据卡要求同时检查底层来源许可。见 [作者数据卡](https://huggingface.co/datasets/openbmb/Ultra-FineWeb)。', 'ultra')
section('candidates', '## 首先比较独立筛选方法，再考虑补充来源\n\n下表按用途排列，全部尚未在本项目完成新抽样或对照训练。“高质量”沿用作者的筛选定义，不表示切分标签已验证。文档数、文本大小与带向量的磁盘大小不可直接比较，所以没有制作规模排名图。')
rows = [
    dict(candidate='Ultra-FineWeb 高分子集', role='低成本内部对照', reason='现有 score 可作分层筛选', limit='当前已筛选；更高分可能改变文体，不保证标签更准'),
    dict(candidate='FineWeb2-HQ · cmn_Hani', role='首个外部候选', reason='不同筛选管线；有质量分数和网页溯源字段', limit='知识性不等于切分合理性；须跨库去重、核对标点'),
    dict(candidate='Chinese Fineweb Edu V2.1 · 4–5 分档', role='条件性候选', reason='可集中抽取讲解性自然段', limit='上游重叠；题目/公式仍需过滤；商用许可待明确'),
    dict(candidate='FineWiki · 中文', role='百科补充／抽取方式对照', reason='渲染 HTML 抽取，可按页面与版本追溯', limit='不是新知识范围；已有维基须按页面去重；仍含表格公式'),
    dict(candidate='ChineseWebText2.0 / CCI3-HQ', role='备用对照', reason='均有专门的质量筛选方法', limit='与现有上游有重叠；CCI3-HQ 有访问门槛'),
    dict(candidate='Fineweb-Edu V3 / CCI4.0-M2', role='不优先替换主语料', reason='已变为指令数据或重写混合语料', limit='偏离自然中文切分监督；版本新不等于更适配'),
]
comparison = next(x for x in sources if x['id'] == 'comparison')
manifest['tables'].append(dict(id='candidates', title='候选语料与用途', dataset='candidates', source=comparison,
                              density='comfortable', columns=[dict(field=k, label=v) for k,v in
                              [('candidate','候选'),('role','用途'),('reason','值得比较的原因'),('limit','主要限制')]]))
manifest['blocks'].append(dict(id='candidate-table', type='table', tableId='candidates'))
section('hq', '## FineWeb2-HQ 值得先试，但其优势尚未在切分任务成立\n\n作者保留每种语言质量分数最高的 10% 文档；中文配置 cmn_Hani 约 5,421 万篇，有 quality_score、URL、日期和原抓取定位信息。作者的 784G 磁盘规模包含向量，抽样应仅读文本、分数和溯源列。\n\n它的验证任务是大模型预训练及知识测验，没有与本项目使用的 Ultra-FineWeb 做断点预测对照。其 ODC-By 许可还附带 Common Crawl 使用条款。见 [数据卡与论文入口](https://huggingface.co/datasets/epfml/FineWeb2-HQ)。', 'hq')
section('punctuation', '## 标点保真是本任务特有的接入检查\n\nFineWeb2 的版本记录曾修复全角标点被转换为半角的问题。FineWeb2-HQ 的说明不足以确认所有中文文本采用的基础处理版本，不能据此判它受影响，也不能略过检查。先核对实际字符和段落边界，避免语料接入时改变我们依赖的标签。见 [基础库版本记录](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)。', 'fw2')
section('edu', '## Edu 高分档有吸引力，但不能只看 Apache 标签\n\nV2.1 的 4–5 分档公布约 1,779 万行、70 GB，适合尝试筛出自然说明段落。它的英文、中文 token 总量相差一个数量级，因此本报告不引用该 token 数。\n\n页面顶部标 Apache-2.0，正文同时写明社区许可及商用需获得许可；用途与条款需先明确。教育价值高的习题或数学排版不一定能产生好断点。见 [V2.1 作者说明](https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1)。', 'edu')
section('edu3', '## 更高版本号没有消除任务差异\n\nFineweb-Edu-Chinese-V3 是中文、英文指令微调样本，不是可直接替换的自然网页文档库。暂不把它列为主候选。见 [V3 数据卡](https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V3)。', 'edu3')
section('wiki', '## FineWiki 更适合检查百科抽取质量\n\n作者从渲染后的维基 HTML 提取，中文约 130 万页面，并保留页面标识、版本和修改时间；许可列出 CC BY-SA 4.0/GFDL。对本项目可比较抽取后的自然段，但仍应排除公式、表格等结构。已有维基数据意味着必须按页面而非仅文本哈希排除验证／测试重复。见 [FineWiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki)。', 'wiki')
section('cwt', '## ChineseWebText2.0 可作备用筛选对照\n\n它提供质量分数，也带自动预测的内容分类。可按分数分层抽样，但分类不是可靠的人工领域标签，不用于恢复领域权重。其清理规则还涉及繁体及英文比例，若希望覆盖繁体中文，需要单独核查覆盖损失。它已出现在 Ultra-FineWeb 的上游链接中。见 [作者数据卡](https://huggingface.co/datasets/CASIA-LM/ChineseWebText2.0)。', 'cwt')
section('cci', '## CCI3-HQ 暂列备用，未访问受限文件\n\n公开说明介绍了比基础 CCI3 更严格的质量筛选。但本次原始 README 下载返回 401，未获取受限样本或确认完整授权条款；不把独立的标注基准许可套用到该语料。见 [CCI3-HQ](https://huggingface.co/datasets/BAAI/CCI3-HQ)。', 'cci')
section('historical', '## 旧抽查不能用来给新候选排名\n\n两轮历史抽查共 1,000 篇不同文档，AI 初审记录为 973 可用、15 不适合、12 不确定。这是现有语料内部按来源配额抽文档的结果；不是全库随机误标率，更不是本轮新候选比较。文档可用也不保证每个抽出的切点都可接受。', 'old-audit')
section('definitions', '## 对比要同时衡量文本损坏与标签错误\n\n本次只核对资料与代码，没有下载新候选正文、做语义抽样或启动训练。后续应分别记录：\n\n- 文本损坏：HTML/OCR 残留、拼接、缺失与排版损坏。\n- 标签问题：恢复原上下文后，标记处不构成合理的阅读边界。\n- 不确定：存在歧义、缺上下文或可接受多个位置；不能强行计作错误。\n- 覆盖和产出：抽取后保留的样本量、长度与标点类型分布、独立文档数、重复贡献。\n\n不同库的分数不可直接同阈值比较。高知识性、低毒性、低困惑度均不是正确切分的充分条件。')
section('experiment', '## 建议先做同口径抽查，再决定是否替换\n\n1. 为现用数据、其高分子集与 FineWeb2-HQ 固定版本；从多个分片随机抽文档，保留完整上下文、分数、URL／文档 ID 和原始位置。Edu 在许可明确后加入。\n2. 每组先审查约 1,000 篇独立文档；用同一清理、抽取和两侧至少两字规则。分别看文档等权、实际训练行口径，防止长文贡献掩盖问题。对关键缺陷和不确定案例做人工复核。\n3. 在候选间及现有验证／测试数据之间做文档和近重复排查。除总体差异，也按句长、短边长度、标点类型对齐比较；保持来源标签只表示来源。\n4. 若抽查有收益，再用等量保留样本、相同长度构成／更新预算、初始化与超参数做 smoke。关闭来源和位置权重；重复随机种子评估波动。使用固定验证集，并增加独立人工确认的阅读边界验证集；测试集留待最终方案。\n\n本次没有执行以上实验，全量训练继续暂停。')
section('questions', '## 尚需实测的三个问题\n\n更高文档分数能否减少错误老师，还是只偏向教材式文字？新筛选能否改善双字短边和句中边界，同时保留自然网页文体？消除模板重复后，还能产出多少独立有效样本？\n\n这三项比“语料更大、版本更新、大模型榜单更高”更能决定是否值得换源。')

datasets = {'candidates':rows, 'source_mix':mix_rows}
database = OUT/'evidence.sqlite'
with sqlite3.connect(database) as connection:
    connection.row_factory = sqlite3.Row
    for name, data in datasets.items():
        keys = list(data[0])
        connection.execute('DROP TABLE IF EXISTS '+name)
        connection.execute('CREATE TABLE '+name+' ('+','.join('"'+k+'"' for k in keys)+')')
        connection.executemany('INSERT INTO '+name+' VALUES ('+','.join('?' for _ in keys)+')',
                               [tuple(row[k] for k in keys) for row in data])
        datasets[name] = [dict(row) for row in connection.execute('SELECT * FROM '+name)]
        assert datasets[name] == data
    total = connection.execute('SELECT SUM(rows), SUM(share) FROM source_mix').fetchone()
    assert total[0] == namespace['total'] and abs(total[1]-1) < 1e-12
for source_id, dataset in [('local','source_mix'), ('comparison','candidates')]:
    entry = next(s for s in sources if s['id'] == source_id)
    entry['query']['original_path'] = entry['path']
    entry['path'] = 'training/artifacts/corpus-source-review-20261004/evidence.sqlite'
    entry['query'].update(engine='SQLite', language='sql', sql='SELECT * FROM '+dataset)

payload = dict(surface='report', manifest=manifest,
               snapshot=dict(version=1, generatedAt=timestamp, status='ready', datasets=datasets),
               sources=sources)
save(OUT/'artifact.json', payload)
save(REVIEW/'checks.json', dict(readme_hashes_verified=8, gated_readmes=2,
                              notebook_code_cells_executed=execution_count,
                              execution_mode='sequential Python standard library; no Jupyter kernel',
                              notebook_structure_checked=True,
                              local_revision_matches_observed_upstream=True,
                              current_extractor_columns=['content','source'],
                              no_new_quality_sample=True, no_training_started=True,
                              manifest_sha256=hashlib.sha256(namespace['manifest_path'].read_bytes()).hexdigest()))
print(json.dumps(dict(artifact=str(OUT/'artifact.json'), notebook=str(REVIEW/'research.ipynb'), checks='passed'), ensure_ascii=False))
