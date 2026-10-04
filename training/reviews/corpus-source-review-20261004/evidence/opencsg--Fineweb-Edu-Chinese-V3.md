---
language:
- zh
- en
license: other
task_categories:
- text-generation
- question-answering
tags:
- education
- textbook
- sft
- synthetic
- reasoning
- fineweb
- opencsg
size_categories:
- 100K<n<1M
pretty_name: Fineweb-Edu-Chinese-V3
---

# Fineweb-Edu-Chinese-V3

<div align="center">
  <a href="#chinese">中文</a> | <a href="#english">English</a>
</div>

<div align="center">

[OpenCSG 社区](https://opencsg.com/datasets) | [数据集许可协议](./OpenCSG数据集许可协议.md)

</div>

<a id="chinese"></a>

## 数据集简介

**Fineweb-Edu-Chinese-V3** 是 OpenCSG 面向学科知识问答、教材理解和推理型指令微调场景构建的高质量中英双语教育 SFT 数据集，也是 Fineweb-Edu-Chinese 系列的最新版本。

该版本包含 **18.81 万条 SFT 样本**，来自 **100,442 篇**高质量图书、教材、学科文献与技术长文，覆盖计算机、自然科学、社科人文、法学、经济五大学科方向，并同步提供 **Messages**、**Messages-no-system**、**Alpaca** 三种训练格式。三种格式是同一批问答对的不同导出视图，训练时应按模型模板选择其中一种，而不是简单相加作为独立数据规模。

V3 是 Fineweb-Edu-Chinese 系列的一次**数据源与构造范式的整体切换**。V1.0 至 V2.3 均以大规模中文网页语料为基础：通过打分器筛选出具备教育属性的网页文本，再由大模型生成问答。V3 不再从网页出发，而是直接以**正式出版的图书、教材与学科文献**为源，每一条样本都由一条 **8 阶段文档理解管线**逐本构建：先做版面解析与章节重建，再抽取"知识点"（Knowledge Point）作为最小可测试单元，然后经过问题草拟、原文证据检索、难度精炼、跨章节合并、题型转换，最后导出为训练格式。

这条链路的目标不是产出更多样本，而是让每条问答都能追溯到源文档中的具体章节与实体，并覆盖完整的推导过程而非孤立的概念复述。

本页保留高层次说明，聚焦数据集价值、版本升级、构建方法、使用方式与风险边界。

---

## 核心价值

面向学科知识的中文 SFT 数据长期存在三个实际问题：高质量教材与专著难以转化为可训练格式；网页合成问答缺乏可追溯的证据支撑；生成结果偏向浅层概念复述，缺少推导过程与结构化表达。

本数据集重点提升三类能力：

- **教材级知识密度**：数据源为正式出版的图书、教材与学科文献，而非网页抓取内容，知识点具备体系性与准确性基础。
- **有据可依的问答构造**：问题与答案由源文档中显式抽取的实体（定理、关键公式、命题、表格）驱动，构建阶段引入原文证据检索环节，降低自由发挥式回答的比例。
- **推理型长答案**：答案普遍包含分步推导、符号与单位定义、假设与适用范围说明，适合训练需要展示完整解题过程的模型。

对社区用户而言，本数据集提供了一批可直接用于学科知识 SFT 的公开数据。对产业场景而言，它更适合作为教育助手、学科问答、专业培训和科研辅助类模型的数据底座之一。

---

## 版本演进

| 版本号 | 核心定位 | 数据规模 | 关键特性与改进 | 当前状态 |
| --- | --- | --- | --- | --- |
| **V1.0** | 概念验证 | 约 9000 万条，约 300GB | 初代 Chinese Fineweb Edu 语料；BERT 打分模型；MinHash 去重；数据源包括 CCI2、SkyPile、Tele-AI | 已弃用 |
| **V2.0** | 规模化扩展 | 约 1.88 亿条，约 420B tokens | 升级至 OpenCSG csg-wukong-enterprise V2 打分器；扩展 Industry2、wanjuan1.0、wudao 等数据源 | 已弃用 |
| **V2.1** | 预训练精选 | 总计约 1.5T tokens | 按分数分层组织；新增 map-cc、opencsg-cc；支持灵活预训练和课程学习 | 推荐用于预训练 |
| **V2.2** | SFT 与对齐 | 约 143.7 万条高质量问答 | 将高质量教育语料转化为 SFT 问答数据；提供纯 QA 与上下文版本 | 历史 SFT 版本 |
| **V2.3** | 更高纯度的 SFT 数据 | 23.04 万条 QA pairs | 升级 V2.2 的源文本选择和生成逻辑；强化证据对齐、质量过滤和多格式导出 | 历史 SFT 版本 |
| **V3** | **文档级学科推理数据** | **18.81 万条样本 / 10.04 万篇源文档** | **数据源由网页语料切换为正式出版图书与教材；8 阶段文档理解管线；知识点驱动构造；新增选择题与表格题型；单样本文本量提升至 V2.3 的 2.6 倍** | **推荐用于 SFT** |

> **关于跨版本规模的可比性**：V1.0 至 V2.1 是**预训练语料**，规模以条数、GB 或 tokens 计量；V2.2 起系列转向 **SFT 问答数据**，规模以 QA pairs 计量。因此表中"约 9000 万条"与"18.81 万条"并非同一量纲，跨阶段直接比较条数没有意义。**仅 V2.2、V2.3、V3 之间的样本数具有可比性。**

系列的演进可以分为两个阶段：

- **预训练语料阶段（V1.0 → V2.1）**：目标是从海量中文 Web 内容中筛出更具教育价值的文本。改进集中在打分器（BERT → csg-wukong-enterprise V2）、数据源扩展与分数分层组织上。
- **SFT 数据阶段（V2.2 → V3）**：目标转为构造可直接训练的问答数据。V2.2 完成了从语料到问答的形态转换；V2.3 收紧了源文本筛选门槛；**V3 则更换了数据源本身**——从中文网页语料切换为正式出版的图书与教材。

换句话说，V1.0 至 V2.3 是同一条技术路线的逐步收紧，数据源始终是网页语料，改进集中在"如何筛得更准"；V3 改变的是这条路线的起点，重心从"筛选"转向"文档理解"。

### 系列定位与生态

Fineweb-Edu-Chinese 系列是全球下载量排名前三的中文数据集之一，累计下载超百万次，已在学术与产业两端形成规模化使用：

- **学术**：被斯坦福大学、清华大学、中国人民大学高瓴人工智能学院、上海人工智能实验室、北京智源研究院等 20 余家机构的论文引用，累计被 100 余篇学术论文引用，出现在 NeurIPS、ACL、EMNLP、ICLR 等国际会议及 Nature 子刊、JMLR 等期刊中；合作机构还包括鹏城实验室、西南电子技术研究所、西班牙国家级超算中心（Barcelona Supercomputing Center）与 Mozilla Data Collective。
- **产业**：支撑 Llama3-Chinese、DeepSeek 等模型训练，并被中国移动、中国联通、英伟达（NVIDIA）、苹果公司（Apple Inc.）、OPPO、美团、阿里巴巴、蚂蚁集团、面壁智能（ModelBest）、Krafton 等企业采用。
- **生态**：系列累计数据体量达 2.42TB、覆盖 9.57 亿条高质量文本，已孵化出 10 余个垂直领域微调模型。

> 以上为 Fineweb-Edu-Chinese **系列**截至 V2.3 的累计生态数据，用于说明本系列的定位与沿革，并非 V3 单个版本的使用统计。

系列的一贯理念是让中文大模型不只是"读到更多中文"，而是能够"学到更好的中文"。V3 在此基础上再进一步：不只是"学到更好的中文"，而是**学会完整的推导与解题过程**。

---

## V3 相比 V2.3 的变化

### 数据来源与构造范式

V2.3 的核心挑战是"如何从海量网页中筛出适合生成 SFT 样本的文本"——它训练了一个中文源文本分类打分器，从约 2.3T 语料中排序选择。这条路线的上限受制于网页本身：教育类网页的知识往往是碎片化的、缺少推导过程的，且与文章排版、导航栏、广告混杂。

V3 直接绕开了这个问题：源文档本身就是成体系的图书与教材，知识密度和准确性由出版流程保证。管线的重点也随之从"筛选"转向"理解"——如何把一本 PDF 教材正确地解析为章节结构、识别其中的定理与公式、并围绕它们构造出能覆盖完整推导链的问题。

### 关键指标对比

| 维度 | V2.3 | V3 |
| --- | --- | --- |
| 数据来源 | 中文网页语料（约 2.3T 候选） | 正式出版图书、教材、学科文献、技术长文 |
| 源单位 | 网页片段 | **100,442 篇完整文档** |
| 筛选/构造核心 | Yuan-embedding 打分器筛选 + GPT-4.1 mini 生成 | **8 阶段文档理解管线 + 知识点驱动构造** |
| 样本数 | 230,400 | 188,148 |
| 单样本平均体量 | 1,691 字节 | **4,384 字节（2.6×）** |
| 文本总量（Alpaca 格式） | 约 390 MB | **约 825 MB（2.1×）** |
| 题型 | General QA | **General QA / Table QA / Single Choice / Multiple Choice** |
| 答案风格 | 段落式解释 | **分步推导，含符号定义、单位、假设与适用范围** |
| 训练格式 | Messages / Messages-no-sys / Alpaca | 同左 |
| 语言 | 中文为主 | 中英双语 |

### 如何理解规模变化

样本条数在本系列中已连续两个版本下降：V2.2 的 143.7 万 → V2.3 的 23.04 万 → V3 的 18.81 万。这是系列的一贯取向——V2.3 发布时即说明，其规模小于 V2.2「并不是因为数据能力下降，而是因为筛选标准更严格」。V3 延续了同一逻辑，但下降幅度小得多（-18%），且伴随单样本体量的大幅上升。

V3 的单条样本平均为 4,384 字节，是 V2.3（1,691 字节）的 **2.6 倍**，因此整体文本量反而是 V2.3 的 **2.1 倍**。换言之，V3 的条数下降并不意味着数据量减少，而是同样的文本预算被分配到了更少、更长、信息更完整的样本上。

差异来自样本形态：V2.3 的问答多为概念解释型的段落式回答；V3 的问题通常包含背景设定、符号与单位表格、多个分小问，答案则是分步推导过程。实测 V3 样本的问题平均 2,320 字符、答案平均 1,722 字符，P90 分别达到 4,321 和 3,343 字符。

因此，**如果你的目标是训练模型输出简洁的知识性回答，V2.3 仍然是合适的选择；如果目标是让模型展示完整的解题与推导过程，V3 更契合**。两者数据源不重叠（网页语料 vs. 出版文献），语言侧重也不同（V2.3 以中文为主，V3 为中英双语），因此完全可以混合使用以兼顾两类能力。

### 新增能力

- **客观题型**：V3 新增 Single Choice（29,443 条）与 Multiple Choice（24,054 条），合计占 28.4%，可直接用于训练或评测选择题作答能力，这是 V2.3 不具备的。
- **表格理解**：Table QA（3,910 条）来自源文档中的真实表格，而非合成数据。
- **文档级溯源**：每条样本携带 `source_paper` 与所属章节信息，可回溯到具体出版物，便于做数据审计与领域切分。
- **学科可控性**：16 个构建域按学科独立配置抽取范式与人设，支持按学科方向做采样或分层训练。

---

## 数据构建流程

原始文档经过 8 个阶段的管线处理，逐本构建为 SFT 样本。各阶段职责如下：

| 阶段 | 名称 | 职责 |
| --- | --- | --- |
| **S1** | Preprocess | 版面解析与清洗，将 PDF/文档转为结构化 Markdown，分离正文、公式、表格与图片 |
| **S1.5** | ChunkConnect | 按章节标题重建文档层级，将正文切分为带 `heading` 的语义块 |
| **S2** | Editor | 抽取知识点（KP）：识别定理、关键公式、命题、表格等实体，产出带 `section_heading`、`original_entity`、`knowledge_point_summary`、`question_types` 的原子/宏观知识点 |
| **S3** | Query Generator | 基于知识点与源文档生成问题草稿 `query_draft` |
| **S4** | Retriever | 回到原文检索支撑证据，为每个问题补齐 `core_answer` 与 `background_evidence` |
| **S5** | Refiner | 提升问题难度与完整性，产出结构化的 `question` / `final_answer` |
| **S6** | Merger | 跨章节合并知识点，筛选并生成综合性"超级问题"，附带质量评分 |
| **S7** | Converter | 转换为最终题型：General QA / Table QA / Single Choice / Multiple Choice |
| **S8** | SFT Formatter | 按文档切分 train/val，导出 Messages、Messages-no-system、Alpaca 三种格式 |

其中 **S2 的知识点抽取**是整条链路的核心设计。它遵循"整体完整性原则"（Holistic Integrity Principle）：当一个命题连同它的前置假设与关键公式共同构成一个自洽的推导闭环时，会被合并为单个**宏观知识点**而不被拆散；不构成这种闭环的实体则作为**原子知识点**处理。这一设计使得下游问题能够覆盖完整的推导链，而不是退化为孤立的公式辨认题。

管线为每个学科方向配置了独立的抽取范式、few-shot 示例与领域人设 system prompt，共覆盖 16 个构建域。

---

## 数据规格与仓库组织

仓库以 **16 个 `.tar.gz` 包**组织，每个包对应一个数据来源。解压后顶层为**文档目录**，每个文档目录内包含该文档的全部训练格式与统计文件。

| 文件 | 说明 |
| --- | --- |
| `train_messages.jsonl` / `val_messages.jsonl` | Messages 格式，含领域人设 system prompt |
| `train_messages_no_sys.jsonl` / `val_messages_no_sys.jsonl` | Messages 格式，不含 system |
| `train_alpaca.jsonl` / `val_alpaca.jsonl` | Alpaca 格式 |
| `stats.json` | 该文档的样本统计 |
| `_dataset_meta.json` | 包级汇总（位于包顶层） |





---

## 数据统计

### 总览

| 指标 | 数值 |
| --- | --- |
| 数据包数 | 16 |
| 源文档数 | 100,442 |
| 样本总数 | 188,148 |
| train / val | 87,946 / 100,202 |
| 压缩后体积 | 271 MB |
| 解压后体积 | 约 3.0 GB |

### 分包明细

| 数据包 | 类别 | 学科方向 | 文档数 | train | val | 合计 | 压缩后 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `csdn_article_sft.tar.gz` | 技术长文 | 计算机 | 44,759 | 75,006 | 44,766 | 119,772 | 117 MB |
| `disciplinary_knowledge_repository_law_sft.tar.gz` | 学科知识库 | 法学 | 18,584 | 398 | 18,588 | 18,986 | 42 MB |
| `disciplinary_knowledge_repository_social_sft.tar.gz` | 学科知识库 | 社科 | 10,232 | 533 | 10,233 | 10,766 | 29 MB |
| `高质量文档结构化数据_第一批_NA_sft.tar.gz` | 结构化文档 | 自然科学 | 6,835 | 244 | 6,666 | 6,910 | 20 MB |
| `disciplinary_knowledge_repository_CS_sft.tar.gz` | 学科知识库 | 计算机 | 2,065 | 4,400 | 2,125 | 6,525 | 7.7 MB |
| `高质量文档结构化数据_第一批_SO_sft.tar.gz` | 结构化文档 | 社科人文 | 4,391 | 351 | 4,341 | 4,692 | 14 MB |
| `电子工业出版社_sft.tar.gz` | 出版社教材 | 多学科 | 1,095 | 3,005 | 1,135 | 4,140 | 3.9 MB |
| `高质量文档结构化数据_第一批_CS_sft.tar.gz` | 结构化文档 | 计算机 | 3,823 | 235 | 3,825 | 4,060 | 12 MB |
| `北京大学出版社电子教材_sft.tar.gz` | 出版社教材 | 多学科 | 948 | 2,820 | 999 | 3,819 | 5.6 MB |
| `高质量文档结构化数据_第二批_CS_sft.tar.gz` | 结构化文档 | 计算机 | 3,706 | 284 | 3,522 | 3,806 | 12 MB |
| `disciplinary_knowledge_repository_economics_sft.tar.gz` | 学科知识库 | 经济 | 1,834 | 251 | 1,837 | 2,088 | 4.8 MB |
| `机械工业出版社_sft.tar.gz` | 出版社教材 | 多学科 | 1,572 | 113 | 1,573 | 1,686 | 4.8 MB |
| `清华大学出版社_sft.tar.gz` | 出版社教材 | 多学科 | 277 | 22 | 277 | 299 | 898 KB |
| `高质量文档结构化数据_第二批_NA_sft.tar.gz` | 结构化文档 | 自然科学 | 204 | 22 | 193 | 215 | 707 KB |
| `科学出版社_sft.tar.gz` | 出版社教材 | 多学科 | 37 | 154 | 42 | 196 | 165 KB |
| `人民邮电出版社_sft.tar.gz` | 出版社教材 | 多学科 | 80 | 108 | 80 | 188 | 222 KB |

### 按来源类别汇总

| 类别 | 包数 | 文档数 | 样本数 | 占比 |
| --- | --- | --- | --- | --- |
| 技术长文 | 1 | 44,759 | 119,772 | 63.7% |
| 学科知识库 | 4 | 32,715 | 38,365 | 20.4% |
| 结构化文档 | 5 | 18,959 | 19,683 | 10.5% |
| 出版社教材 | 6 | 4,009 | 10,328 | 5.5% |

### 题型分布

| 题型 | 样本数 | 占比 |
| --- | --- | --- |
| General QA | 130,741 | 69.5% |
| Single Choice | 29,443 | 15.6% |
| Multiple Choice | 24,054 | 12.8% |
| Table QA | 3,910 | 2.1% |

---

## Schema

### Messages

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are an expert engineering educator with deep knowledge of mechanics of materials, structural mechanics, electrical circuit analysis, sensors and instrumentation, mechanical design, and applied mathematics for engineering. Provide rigorous, step-by-step answers grounded in physical principles and engineering design standards."
    },
    { "role": "user", "content": "用户问题" },
    { "role": "assistant", "content": "助手答案" }
  ],
  "metadata": {
    "domain": "cbs",
    "question_type": "General QA",
    "source_paper": "29838-设计构成",
    "quality_score": 0.0,
    "images": []
  }
}
```

### Messages-no-system

```json
{
  "messages": [
    { "role": "user", "content": "用户问题" },
    { "role": "assistant", "content": "助手答案" }
  ],
  "metadata": { "...": "同上" }
}
```

### Alpaca

```json
{
  "instruction": "Answer the following cbs question with detailed derivations and explanations.",
  "input": "用户问题",
  "output": "助手答案",
  "metadata": { "...": "同上" }
}
```

### metadata 字段

| 字段 | 含义 |
| --- | --- |
| `domain` | 构建域标识，共 16 个（如 `disciplinary_cs`、`highqua1_na`、`cbs`、`csdn_article_b1`） |
| `question_type` | 题型：`General QA` / `Table QA` / `Single Choice` / `Multiple Choice` |
| `source_paper` | 源文档标识 |
| `quality_score` | 质量分字段。当前公开快照中该字段普遍为 `0.0`，质量控制主要发生在 S6 筛选阶段而非记录级打分，**不建议将其作为过滤依据** |
| `images` | 源文档中与该样本所属章节关联的图片路径。**图片文件本身不包含在本仓库中**，该字段仅作溯源标记，本数据集为纯文本 SFT 数据 |

### stats.json 字段

| 字段 | 含义 |
| --- | --- |
| `domain` | 该文档的构建域 |
| `total_samples` / `train_samples` / `val_samples` | 样本总数 / train / val |
| `papers_succeeded` / `papers_failed` | 构建成功 / 失败的篇数 |
| `by_question_type` | 题型分布 |
| `by_source_paper` | 按源文档的样本分布 |
| `mean_quality_score` | 平均质量分 |

---

## 真实样例

以下样例节选自 `北京大学出版社电子教材` 包，保留原始内容。

**Question**

> **Background**
>
> Engineering problem. The relationship between contrast and harmony is fundamental in color design, influencing both emotional expression and visual stability.
>
> **Variables and parameters.**
>
> | Symbol | Definition | Unit |
> |---|---|---|
> | C | Contrast level | dimensionless |
> | H | Harmony level | dimensionless |
> | E | Emotional expression | dimensionless |

**Answer**

> 1\) (a) Contrast and harmony are interdependent aspects in color design. Contrasting colors enhance emotional expression, often evoking excitement and vibrancy in a visual composition.
>
> 2\) (b) Harmonious arrangements provide stability and unity in visual communication by reducing the intensity of contrasts, creating a cohesive aesthetic. For example, using analogous colors can create a calming effect.
>
> 3\) (c) To achieve a balance between contrast and harmony, a designer could use complementary colors for focal points while employing analogous colors for background elements.

该样例体现了典型结构：问题包含背景设定、符号与单位表格、分小问设问；答案按 `1) (a)`、`2) (b)` 的编号逐问作答。

---

## 快速开始

### 下载与解压

```bash
# 通过 Hugging Face CLI 下载
pip install -U huggingface_hub
hf download opencsg/Fineweb-Edu-Chinese-V3 --repo-type dataset --local-dir ./Fineweb-Edu-Chinese-V3

# 只下载指定包
hf download opencsg/Fineweb-Edu-Chinese-V3 --repo-type dataset \
  --include "北京大学出版社电子教材_sft.tar.gz" --local-dir ./Fineweb-Edu-Chinese-V3

# 或使用 git（需要 git-lfs）
git lfs install
git clone https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V3

# 解压单个包
tar -xzf 北京大学出版社电子教材_sft.tar.gz -C ./data/
```

Python 方式：

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "opencsg/Fineweb-Edu-Chinese-V3",
    repo_type="dataset",
    local_dir="./Fineweb-Edu-Chinese-V3",
)
```

### 加载数据

```python
import glob
from datasets import load_dataset

# 递归 glob 兼容单层与三层两种目录布局
train_files = glob.glob("./data/**/train_messages.jsonl", recursive=True)
val_files   = glob.glob("./data/**/val_messages.jsonl",   recursive=True)

ds = load_dataset(
    "json",
    data_files={"train": train_files, "validation": val_files},
)
print(ds["train"][0]["messages"])
```

### 重新切分（推荐）

由于原始切分按文档进行、train/val 比例偏离常规，建议合并后自行切分：

```python
import glob
from datasets import load_dataset

files = glob.glob("./data/**/*_messages.jsonl", recursive=True)
full = load_dataset("json", data_files=files, split="train")
split = full.train_test_split(test_size=0.02, seed=42)
```

### 接入训练

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("你的基座模型")
text = tokenizer.apply_chat_template(sample["messages"], tokenize=False)
```

### 三种格式的选择

- **Messages（含 system）**：带学科人设 system prompt，推荐用于需要稳定领域角色的 chat-style SFT。
- **Messages-no-system**：不带 system 的纯对话，适合已有自定义 system prompt 的训练流程。
- **Alpaca**：`instruction` / `input` / `output` 结构，兼容 LLaMA-Factory 等 Alpaca 系训练框架。

---

## 适用场景

- 学科知识问答与教育助手模型的监督微调
- 需要展示完整推导过程的推理型模型训练
- 中英双语学科问答能力构建
- 教材理解、专业培训与科研辅助类模型
- 文档理解管线、知识点抽取与合成数据质量研究

---

## 限制与风险边界

本数据集为基于真实出版文档、通过大模型管线构造的**合成问答数据**。尽管构建过程中引入了原文证据检索与质量筛选环节，样本仍可能包含事实错误、推导缺陷、遗漏或表达偏差，**不应被视为事实权威来源或专业意见**。


将本数据集用于医疗、法律、金融、教育评价等高风险场景前，需进行额外的领域专家评审、模型评测、安全评测与合规审查。使用者也应结合自身产品形态、部署地区和下游任务要求，评估数据使用带来的偏差、版权、隐私和安全风险。

---

## 许可说明

使用本数据集需要遵循 [OpenCSG 数据集许可协议](./OpenCSG数据集许可协议.md)。仓库 metadata 中的 `license: other` 表示本数据集采用平台预设列表之外的许可协议，实际许可条款以该协议为准。

本数据集可按 OpenCSG 数据集许可协议申请商业用途。若计划将本数据集，或基于本数据集训练、增强的模型、系统、Agent、API 服务和商业产品用于商业场景，请发送邮件至 lorraineg@opencsg.com 获取许可。

---

## Citation

```bibtex
@dataset{opencsg_fineweb_edu_chinese_v3_2026,
  title        = {Fineweb-Edu-Chinese-V3: A Document-Grounded Bilingual Educational Instruction Dataset},
  author       = {OpenCSG},
  year         = {2026},
  url          = {https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V3},
  note         = {OpenCSG dataset repository}
}
```

---
---

<a id="english"></a>

# Fineweb-Edu-Chinese-V3

## Dataset Overview

**Fineweb-Edu-Chinese-V3** is a high-quality bilingual (Chinese/English) educational SFT dataset built by OpenCSG for disciplinary knowledge QA, textbook comprehension, and reasoning-oriented instruction tuning. It is the latest release in the Fineweb-Edu-Chinese series.

This release contains **188,148 SFT samples** derived from **100,442 documents** — books, textbooks, disciplinary literature, and long-form technical articles — spanning five major areas: computer science, natural sciences, social sciences and humanities, law, and economics. The same QA pairs are exported in three training formats: **Messages**, **Messages-no-system**, and **Alpaca**. These are alternative views of one QA set; select the format matching your model template rather than summing them as independent data volume.

V3 marks a **shift in both data source and construction paradigm** for the series. V1.0 through V2.3 were all built on large-scale Chinese web corpora: a scorer selected web text with educational properties, and an LLM then generated QA from it. V3 starts from **formally published books, textbooks, and disciplinary literature** instead, and constructs every sample document-by-document through an **8-stage document understanding pipeline**: layout parsing and section reconstruction, knowledge point extraction as the minimal testable unit, query drafting, source-evidence retrieval, difficulty refinement, cross-section merging, question-type conversion, and format export.

The goal of this chain is not volume but traceability — each QA pair maps back to a specific section and entity in its source document, and covers a complete derivation rather than an isolated concept restatement.

---

## Core Value

Disciplinary SFT data faces three persistent problems: high-quality textbooks and monographs are hard to convert into trainable formats; web-synthesized QA lacks traceable evidence; and generated output tends toward shallow concept restatement without derivation or structured presentation.

This dataset targets three capabilities:

- **Textbook-grade knowledge density**: sources are formally published books, textbooks, and disciplinary literature rather than scraped web pages, giving knowledge points a systematic and accurate foundation.
- **Evidence-grounded QA construction**: questions and answers are driven by entities explicitly extracted from source documents (theorems, key equations, propositions, tables), with a source-retrieval stage that reduces free-form hallucinated answers.
- **Reasoning-style long answers**: answers typically include step-by-step derivations, symbol and unit definitions, and statements of assumptions and validity ranges — suitable for training models that must show complete solution processes.

---

## Version Evolution

| Version | Positioning | Scale | Key Features and Improvements | Status |
| --- | --- | --- | --- | --- |
| **V1.0** | Proof of concept | ~90M records, ~300GB | First-generation Chinese Fineweb Edu corpus; BERT scorer; MinHash dedup; sources include CCI2, SkyPile, Tele-AI | Deprecated |
| **V2.0** | Scale-up | ~188M records, ~420B tokens | Upgraded to OpenCSG csg-wukong-enterprise V2 scorer; added Industry2, wanjuan1.0, wudao | Deprecated |
| **V2.1** | Pretraining selection | ~1.5T tokens total | Score-stratified organization; added map-cc, opencsg-cc; supports flexible pretraining and curriculum learning | Recommended for pretraining |
| **V2.2** | SFT and alignment | ~1.437M QA pairs | Converted high-quality educational corpus into SFT QA data; pure-QA and with-context variants | Legacy SFT release |
| **V2.3** | Higher-purity SFT data | 230.4K QA pairs | Upgraded V2.2's source selection and generation logic; strengthened evidence alignment, quality filtering, multi-format export | Legacy SFT release |
| **V3** | **Document-level disciplinary reasoning data** | **188.1K samples / 100.4K source documents** | **Source switched from web corpora to published books and textbooks; 8-stage document understanding pipeline; knowledge-point-driven construction; adds multiple-choice and table question types; 2.6× larger per-sample text than V2.3** | **Recommended for SFT** |

> **On cross-version comparability**: V1.0 through V2.1 are **pretraining corpora**, measured in records, GB, or tokens; from V2.2 onward the series shifted to **SFT QA data**, measured in QA pairs. "~90M records" and "188.1K samples" are therefore not the same unit, and comparing counts across these stages is meaningless. **Only V2.2, V2.3, and V3 have directly comparable sample counts.**

The series evolved in two stages:

- **Pretraining corpus stage (V1.0 → V2.1)**: selecting educationally valuable text from large-scale Chinese web content. Improvements centered on the scorer (BERT → csg-wukong-enterprise V2), source expansion, and score-stratified organization.
- **SFT data stage (V2.2 → V3)**: constructing directly trainable QA data. V2.2 completed the corpus-to-QA transformation; V2.3 tightened source-selection thresholds; **V3 replaced the data source itself** — from Chinese web corpora to formally published books and textbooks.

In short, V1.0 through V2.3 tightened one technical route whose source was always web text, with improvements focused on *selecting more precisely*; V3 changes the starting point of that route, shifting the center of gravity from *selection* to *document understanding*.

### Series Positioning and Ecosystem

The Fineweb-Edu-Chinese series is among the top three most-downloaded Chinese datasets worldwide, with over one million cumulative downloads and established use in both academia and industry:

- **Academia**: cited in papers from over 20 institutions including Stanford University, Tsinghua University, Renmin University's Gaoling School of Artificial Intelligence, Shanghai AI Laboratory, and BAAI; cited by 100+ academic papers across NeurIPS, ACL, EMNLP, ICLR, as well as Nature-family journals and JMLR. Collaborating institutions include Pengcheng Laboratory, the Southwest Institute of Electronic Technology, the Barcelona Supercomputing Center, and Mozilla Data Collective.
- **Industry**: supports the training of models such as Llama3-Chinese and DeepSeek, and has been adopted by China Mobile, China Unicom, NVIDIA, Apple Inc., OPPO, Meituan, Alibaba, Ant Group, ModelBest, and Krafton.
- **Ecosystem**: 2.42TB of cumulative data covering 957 million high-quality texts, with 10+ vertical-domain fine-tuned models incubated on top of it.

> The figures above are cumulative ecosystem statistics for the Fineweb-Edu-Chinese **series** as of V2.3, provided to explain the series' positioning and lineage. They are not usage statistics for V3 alone.

The series' guiding principle has been that Chinese LLMs should not merely "read more Chinese" but "learn better Chinese." V3 extends this one step further: beyond learning better Chinese, models should **learn complete derivation and problem-solving processes**.

---

## What Changed in V3 vs. V2.3

### Source and Construction Paradigm

V2.3's central challenge was "how to select, from a vast web corpus, text suitable for generating SFT samples" — it trained a Chinese source-text classifier and ranked candidates from roughly 2.3T of corpus. The ceiling of that route is set by web pages themselves: educational web content is often fragmentary, lacks derivation steps, and is interleaved with layout, navigation bars, and ads.

V3 sidesteps the problem: the source documents are already systematic books and textbooks, with knowledge density and accuracy underwritten by the publishing process. The pipeline's emphasis correspondingly shifts from *selection* to *understanding* — how to correctly parse a PDF textbook into section structure, identify its theorems and equations, and build questions around them that cover complete derivation chains.

### Key Metrics

| Dimension | V2.3 | V3 |
| --- | --- | --- |
| Data source | Chinese web corpora (~2.3T candidates) | Published books, textbooks, disciplinary literature, technical articles |
| Source unit | Web page fragment | **100,442 complete documents** |
| Core of selection/construction | Yuan-embedding scorer + GPT-4.1 mini generation | **8-stage document understanding pipeline + knowledge-point-driven construction** |
| Samples | 230,400 | 188,148 |
| Avg. bytes per sample | 1,691 | **4,384 (2.6×)** |
| Total text (Alpaca format) | ~390 MB | **~825 MB (2.1×)** |
| Question types | General QA | **General QA / Table QA / Single Choice / Multiple Choice** |
| Answer style | Paragraph-form explanation | **Step-by-step derivation with symbol definitions, units, assumptions, validity ranges** |
| Training formats | Messages / Messages-no-sys / Alpaca | Same |
| Language | Primarily Chinese | Bilingual (Chinese/English) |

### Understanding the Scale Change

Sample counts have now declined across two consecutive releases: 1.437M in V2.2 → 230.4K in V2.3 → 188.1K in V3. This reflects a consistent orientation in the series — when V2.3 was released, its smaller size relative to V2.2 was explained as "not a decline in data capability, but the result of stricter selection criteria." V3 follows the same logic, with a far smaller decline (-18%) and a substantial increase in per-sample volume.

V3 averages 4,384 bytes per sample versus V2.3's 1,691 — a **2.6× increase** — so total text volume is in fact **2.1× that of V2.3**. The lower count does not mean less data; the same text budget is allocated across fewer, longer, more complete samples.

The difference is in sample shape. V2.3's QA pairs are mostly paragraph-form conceptual explanations; V3's questions typically include a background setup, a symbol-and-unit table, and several sub-questions, with answers written as step-by-step derivations. Measured over V3 samples, questions average 2,320 characters and answers 1,722 characters, with P90 values of 4,321 and 3,343 respectively.

Accordingly: **if your goal is to train concise knowledge-style answers, V2.3 remains the appropriate choice; if the goal is for the model to show complete solution and derivation processes, V3 fits better.** The two do not overlap in source data (web corpora vs. published literature) and differ in language emphasis (V2.3 primarily Chinese, V3 bilingual), so they can readily be mixed to cover both capabilities.

### New Capabilities

- **Objective question types**: V3 adds Single Choice (29,443) and Multiple Choice (24,054), together 28.4% of the dataset — usable directly for training or evaluating multiple-choice answering, which V2.3 did not support.
- **Table comprehension**: Table QA (3,910) is derived from real tables in the source documents rather than synthesized.
- **Document-level provenance**: every sample carries `source_paper` and section information, traceable back to a specific publication for data auditing and domain partitioning.
- **Disciplinary controllability**: 16 construction domains configure extraction paradigms and personas independently per discipline, supporting per-discipline sampling or stratified training.

---

## Data Construction Pipeline

| Stage | Name | Responsibility |
| --- | --- | --- |
| **S1** | Preprocess | Layout parsing and cleaning; converts PDFs/documents to structured Markdown, separating body text, equations, tables, and images |
| **S1.5** | ChunkConnect | Reconstructs document hierarchy by section headings; splits body text into semantic chunks carrying a `heading` |
| **S2** | Editor | Extracts knowledge points (KPs): identifies theorems, key equations, propositions, tables, producing atomic/macro KPs with `section_heading`, `original_entity`, `knowledge_point_summary`, `question_types` |
| **S3** | Query Generator | Generates `query_draft` from KPs and the source document |
| **S4** | Retriever | Returns to the source text to retrieve supporting evidence, filling in `core_answer` and `background_evidence` |
| **S5** | Refiner | Raises difficulty and completeness, producing structured `question` / `final_answer` |
| **S6** | Merger | Merges KPs across sections, curating comprehensive "super-questions" with quality scoring |
| **S7** | Converter | Converts to final question types: General QA / Table QA / Single Choice / Multiple Choice |
| **S8** | SFT Formatter | Splits train/val per document; exports Messages, Messages-no-system, and Alpaca formats |

**S2's knowledge point extraction** is the core design of this chain. It follows a *Holistic Integrity Principle*: when a proposition together with its prerequisite assumptions and key equations forms a self-contained derivation, they are merged into a single **macro knowledge point** rather than being split; entities not forming such a closed block are handled as **atomic knowledge points**. This lets downstream questions cover complete derivation chains instead of degenerating into isolated formula-recognition items.

The pipeline configures independent extraction paradigms, few-shot examples, and domain-specific system prompts for each discipline, covering 16 construction domains in total.

---

## Data Specification and Repository Organization

The repository is organized as **16 `.tar.gz` packages**, one per data source. After extraction, the top level consists of **document directories**, each containing all training formats and statistics for that document.

| File | Description |
| --- | --- |
| `train_messages.jsonl` / `val_messages.jsonl` | Messages format with a domain-specific system prompt |
| `train_messages_no_sys.jsonl` / `val_messages_no_sys.jsonl` | Messages format without system |
| `train_alpaca.jsonl` / `val_alpaca.jsonl` | Alpaca format |
| `stats.json` | Per-document sample statistics |
| `_dataset_meta.json` | Package-level summary (at package root) |


---

## Statistics

### Overview

| Metric | Value |
| --- | --- |
| Packages | 16 |
| Source documents | 100,442 |
| Total samples | 188,148 |
| train / val | 87,946 / 100,202 |
| Compressed size | 271 MB |
| Uncompressed size | ~3.0 GB |

### Per-package Breakdown

| Package | Category | Discipline | Docs | train | val | Total | Compressed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `csdn_article_sft.tar.gz` | Technical articles | CS | 44,759 | 75,006 | 44,766 | 119,772 | 117 MB |
| `disciplinary_knowledge_repository_law_sft.tar.gz` | Knowledge repository | Law | 18,584 | 398 | 18,588 | 18,986 | 42 MB |
| `disciplinary_knowledge_repository_social_sft.tar.gz` | Knowledge repository | Social sciences | 10,232 | 533 | 10,233 | 10,766 | 29 MB |
| `高质量文档结构化数据_第一批_NA_sft.tar.gz` | Structured documents | Natural sciences | 6,835 | 244 | 6,666 | 6,910 | 20 MB |
| `disciplinary_knowledge_repository_CS_sft.tar.gz` | Knowledge repository | CS | 2,065 | 4,400 | 2,125 | 6,525 | 7.7 MB |
| `高质量文档结构化数据_第一批_SO_sft.tar.gz` | Structured documents | Social sciences & humanities | 4,391 | 351 | 4,341 | 4,692 | 14 MB |
| `电子工业出版社_sft.tar.gz` | Publisher textbooks | Multidisciplinary | 1,095 | 3,005 | 1,135 | 4,140 | 3.9 MB |
| `高质量文档结构化数据_第一批_CS_sft.tar.gz` | Structured documents | CS | 3,823 | 235 | 3,825 | 4,060 | 12 MB |
| `北京大学出版社电子教材_sft.tar.gz` | Publisher textbooks | Multidisciplinary | 948 | 2,820 | 999 | 3,819 | 5.6 MB |
| `高质量文档结构化数据_第二批_CS_sft.tar.gz` | Structured documents | CS | 3,706 | 284 | 3,522 | 3,806 | 12 MB |
| `disciplinary_knowledge_repository_economics_sft.tar.gz` | Knowledge repository | Economics | 1,834 | 251 | 1,837 | 2,088 | 4.8 MB |
| `机械工业出版社_sft.tar.gz` | Publisher textbooks | Multidisciplinary | 1,572 | 113 | 1,573 | 1,686 | 4.8 MB |
| `清华大学出版社_sft.tar.gz` | Publisher textbooks | Multidisciplinary | 277 | 22 | 277 | 299 | 898 KB |
| `高质量文档结构化数据_第二批_NA_sft.tar.gz` | Structured documents | Natural sciences | 204 | 22 | 193 | 215 | 707 KB |
| `科学出版社_sft.tar.gz` | Publisher textbooks | Multidisciplinary | 37 | 154 | 42 | 196 | 165 KB |
| `人民邮电出版社_sft.tar.gz` | Publisher textbooks | Multidisciplinary | 80 | 108 | 80 | 188 | 222 KB |

### By Source Category

| Category | Packages | Docs | Samples | Share |
| --- | --- | --- | --- | --- |
| Technical articles | 1 | 44,759 | 119,772 | 63.7% |
| Knowledge repository | 4 | 32,715 | 38,365 | 20.4% |
| Structured documents | 5 | 18,959 | 19,683 | 10.5% |
| Publisher textbooks | 6 | 4,009 | 10,328 | 5.5% |

### Question Type Distribution

| Question Type | Samples | Share |
| --- | --- | --- |
| General QA | 130,741 | 69.5% |
| Single Choice | 29,443 | 15.6% |
| Multiple Choice | 24,054 | 12.8% |
| Table QA | 3,910 | 2.1% |

---

## Schema

See the Chinese section above for the Messages / Messages-no-system / Alpaca JSON examples. Common `metadata` fields:

| Field | Meaning |
| --- | --- |
| `domain` | Construction domain, 16 in total (e.g. `disciplinary_cs`, `highqua1_na`, `cbs`, `csdn_article_b1`) |
| `question_type` | `General QA` / `Table QA` / `Single Choice` / `Multiple Choice` |
| `source_paper` | Source document identifier |
| `quality_score` | Quality score field. In the current snapshot this is generally `0.0`; quality control happens at the S6 curation stage rather than as per-record scoring, so **it should not be used as a filter** |
| `images` | Image paths associated with the section the sample came from. **Image files are not included in this repository**; the field is a provenance marker only — this is a text-only SFT dataset |

---

## Quick Start

```bash
pip install -U huggingface_hub
hf download opencsg/Fineweb-Edu-Chinese-V3 --repo-type dataset --local-dir ./Fineweb-Edu-Chinese-V3

tar -xzf 北京大学出版社电子教材_sft.tar.gz -C ./data/
```

```python
import glob
from datasets import load_dataset

# Recursive glob handles both single-level and three-level layouts
train_files = glob.glob("./data/**/train_messages.jsonl", recursive=True)
val_files   = glob.glob("./data/**/val_messages.jsonl",   recursive=True)

ds = load_dataset("json", data_files={"train": train_files, "validation": val_files})
print(ds["train"][0]["messages"])
```

Because the original split is per-document and deviates from the usual ratio, re-splitting is recommended:

```python
files = glob.glob("./data/**/*_messages.jsonl", recursive=True)
full = load_dataset("json", data_files=files, split="train")
split = full.train_test_split(test_size=0.02, seed=42)
```

---

## Use Cases

- Supervised fine-tuning for disciplinary knowledge QA and educational assistants
- Training reasoning models that must show complete derivations
- Building bilingual (Chinese/English) disciplinary QA capability
- Textbook comprehension, professional training, and research-assistant models
- Research on document understanding pipelines, knowledge point extraction, and synthetic data quality

---

## Limitations and Risk Boundaries

This is **synthetic QA data** constructed from real published documents via an LLM pipeline. Although source-evidence retrieval and quality filtering are part of the construction process, samples may still contain factual errors, flawed derivations, omissions, or biased phrasing, and **should not be treated as an authoritative source or professional advice**.


Before using this dataset in high-risk scenarios such as healthcare, law, finance, or educational assessment, conduct additional domain-expert review, model evaluation, safety evaluation, and compliance review. Users should also assess bias, copyright, privacy, and security risks in light of their own product form, deployment region, and downstream task requirements.

---

## License

Use of this dataset is subject to the [OpenCSG Dataset License Agreement](./OpenCSG数据集许可协议.md). The `license: other` field in the repository metadata indicates that this dataset uses a license outside the platform's preset list; the agreement itself governs the actual terms.

Commercial use may be requested under the OpenCSG Dataset License Agreement. If you plan to use this dataset — or models, systems, agents, API services, and commercial products trained or augmented with it — in commercial scenarios, please email lorraineg@opencsg.com to obtain a license.
