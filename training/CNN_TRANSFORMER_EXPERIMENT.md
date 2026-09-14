# CNN / Transformer 架构对比实验记录

本次受控 smoke 中，CNN 在完整 validation、完整 test、领域宏平均以及全部五个领域的测试准确率上均领先。
Test 准确率为 **CNN 68.5956%，Transformer 60.8484%**，差 **7.7473 个百分点**。
Transformer 未达到替换 CNN 的目标。决定结束本次实验，继续使用正式 CNN 路线，并清理实验专用代码和生成产物。

实验于 2026-09-14 17:13:50 CST 完成；2026-09-14 归档。本记录在删除原始文件前，从最终
`comparison.json`、两组 `metrics.json`、配置和源代码核对生成，结果与两组原始报告一致。

## 目标与比较范围

检验双向全局注意力能否在相近参数量下改善字符间断点选择，特别是突破 CNN 的局部感受野。
数据准备、字符级输入、断点标签、训练目标和评估方式保持一致，仅替换编码器架构。
两组均从头训练，共享 embedding 和 gap head 的初始权重；本次 CNN 对照没有继承正式大训练的权重。

输入标准为 `unicode-context-v1`，使用现有 8,192 项字符词表和现有文本规范化规则。
没有 BPE、预训练 Transformer、另行生成的语料或输入截断。比较的是模型原始断点分数，
没有叠加浏览器后端的长度平衡或列表保护规则。

## 架构与共同训练设置

| 设置 | CNN | Transformer |
| --- | --- | --- |
| 编码器 | 6 个残差块、12 层卷积 | 3 个 TransformerEncoderLayer |
| 通道数 | 192 | 192 |
| 上下文 | 核宽 3、步长/膨胀率 1；字符特征覆盖 25 字，gap 覆盖 26 字 | 全样本双向注意力，仅屏蔽 padding |
| 注意力与前馈层 | — | 4 个 head，前馈宽度 768，GELU，Pre-LN 和最终 LayerNorm |
| 位置编码 | — | 正弦位置编码，embedding 乘 sqrt(192) |
| 参数量 | 3,052,423 | 3,055,681（多 0.1067%） |

两组使用同一个相邻字符 gap head：拼接左特征、右特征、右减左、左右逐项乘积，
经 Linear(768→192)、GELU(tanh)、Linear(192→1) 输出各 gap 的 logit。
Transformer 各层独立初始化，dropout 为 0。

- 随机种子：2026090405；共享 embedding 初始化使用 seed + 1，gap head 使用 seed + 2。
- 每组 2 个 epoch，每个 epoch 3,977 个训练 batch，总计 7,954 次更新。
- AdamW：固定学习率 0.0003，weight decay 0.0001，foreach=True；无 warmup 或衰减计划。
- 损失：带样本权重的 gap 交叉熵，以 batch 样本数作分母并除以共同 normalizer。
  两组复用同一套位置权重、领域权重及 normalizer；domain-weight power 为 0.65。
- 梯度裁剪 1.0；每批最多 512 条、8,192 个 padding 后 token；相同长度分桶、顺序和 batch 计划。
- 每轮完整 validation；选择分数为 0.5 × 总准确率 + 0.5 × 五领域宏平均准确率。
  两组均选择 epoch 2，之后各自评估一次完整 test。
- CPU、PyTorch 2.8.0；每组 1 个计算线程、1 个 interop 线程、nice 10；先 CNN 后 Transformer。
  测量期间正式大训练并行运行，因此计时受系统负载影响。

## 数据与控制变量核验

直接读取正式语料 manifest 中固定抽取的四个分片，索引为 12、65、152、311。
每 epoch **742,270 条样本、22,754,955 个字符 token**，最长训练样本 1,775 token。
两组完整 validation **727,578 条**、完整 test **766,489 条**；验证和测试没有再次抽样。

| 领域 | 训练 / epoch | Validation | Test |
| --- | ---: | ---: | ---: |
| news | 646 | 3,916 | 3,994 |
| academic | 656 | 4,843 | 4,748 |
| encyclopedia | 1,229 | 6,973 | 6,888 |
| dialogue | 1,376 | 8,215 | 8,185 |
| wikipedia | 489,829 | 703,631 | 742,674 |
| synthetic_multistyle | 248,534 | 0 | 0 |

最终配对核验 `passed=true`。重新核对了两组相同的训练/验证/测试标识、词表、样本权重、
共享初始权重、源代码标识、优化器设置以及每轮实际 batch 顺序、样本数和更新预算。
CNN 与 Transformer 的参数量差异低于 1%。

## 最终指标

准确率是最高分 gap 恢复被隐藏标点位置的比例；宏平均给予五个领域相同权重。
MRR 是目标 gap 排名倒数的平均值，越高越好；平均字符误差和 loss 越低越好。

| 指标 | CNN | Transformer |
| --- | ---: | ---: |
| 参数量 | 3,052,423 | 3,055,681 |
| 最佳 epoch | 2 | 2 |
| Validation 准确率 | 68.5063% | 60.9544% |
| Validation 领域宏平均准确率 | 64.5189% | 56.1617% |
| Validation 选择分数 | 0.665126 | 0.585581 |
| Test 准确率 | 68.5956% | 60.8484% |
| Test 领域宏平均准确率 | 64.8830% | 56.6071% |
| Test loss | 1.021301 | 1.331418 |
| Test MRR | 0.797565 | 0.734328 |
| Test 平均边界误差（字符） | 2.512574 | 2.943535 |
| Test 误差不超过 1 字的比例 | 71.5336% | 65.2577% |
| Test 误差不超过 2 字的比例 | 77.5506% | 72.3405% |
| 两个 epoch 训练耗时（分钟） | 22.347 | 27.928 |
| 两个 epoch 验证耗时（分钟） | 8.673 | 10.564 |
| 完整 Test 推理耗时（分钟） | 4.431 | 5.379 |

## 分领域准确率

| 领域 | CNN Validation | Transformer Validation | CNN Test | Transformer Test |
| --- | ---: | ---: | ---: | ---: |
| academic | 69.3578% | 60.6442% | 70.0505% | 61.8787% |
| dialogue | 69.2392% | 61.3268% | 70.0550% | 61.9914% |
| encyclopedia | 63.9610% | 57.1060% | 65.3600% | 56.9832% |
| news | 51.4045% | 40.6282% | 50.2504% | 41.2118% |
| wikipedia | 68.6321% | 61.1035% | 68.6989% | 60.9706% |

## 每轮结果

| 模型 | Epoch | Train loss | Validation loss | Validation 准确率 | Validation 宏平均 | 选择分数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN | 1 | 1.439496 | 1.003770 | 68.4964% | 62.9126% | 0.657045 |
| CNN | 2 | 0.809163 | 1.023323 | 68.5063% | 64.5189% | 0.665126 |
| Transformer | 1 | 1.848168 | 1.330789 | 60.2463% | 54.5190% | 0.573826 |
| Transformer | 2 | 1.193507 | 1.327257 | 60.9544% | 56.1617% | 0.585581 |

CNN 的第二轮整体 validation 准确率变化很小，宏平均提升，因此仍由共同选择规则选中 epoch 2。
两轮中 Transformer 的训练 loss 均高于 CNN。

## 结论与局限

当前数据、参数规模和训练预算下，CNN 的全部已报告质量指标优于 Transformer，五个领域的
validation 和 test 准确率也均领先；保留 CNN 有直接实验依据。Transformer 的训练与测试
耗时在本次观测中也更长，但并行系统负载下的计时不能作为隔离环境的速度基准。

结论限于一个随机种子、四个训练分片、两个 epoch 和这一套共同超参数。
实验没有为各架构分别寻优，也没有运行全量语料的 Transformer；不能据此判定 Transformer
在所有规模和训练配置下都会失败。准确率衡量标点代理的恢复能力，不等同于人工阅读切分质量。

可能解释是 CNN 的局部结构与当前任务更匹配，或 Transformer 在该预算和优化设置下学习不足；
本次没有消融实验验证这些原因。全局上下文可见并不自动保证当前预算下更高的准确率。

## 实现预检

预检在 2,048-token 输入上验证了有限 loss 和前向/反向运行；输出 2,047 个 gap。
训练与评估实际采用 PyTorch SDPA 的 CPU flash attention 路径；关闭了会显式展开注意力矩阵的
评估 MHA fast path。预检单次前向/反向约 0.994 秒，评估约 0.080 秒，只用于确认实现可运行。
预检 checkpoint 停在 epoch 1 的首个 batch（128 条样本），不计入上面的正式配对结果。

## 数据和结果指纹

以下 SHA-256 来自实验完成时的冻结记录；原始报告和 checkpoint 已按清理决定删除。

| 对象 | SHA-256 |
| --- | --- |
| 原始 comparison.json | `dd7f19c5edb6f6ceec1342f7ea2ba789c267e424ff5d42a475654b0ca6bad9e9` |
| 训练 manifest | `c406e50b443f025ae59d51b75d7e238120e611b721c982741c943e920a79c8f9` |
| 数据 summary | `7df6a900add2dd2a8bc4487720d0f5b4243f75b81e34e295166ed6d1d6a53062` |
| 词表 | `625ec1bb79f24c47948a2f2cfdaf7515335b911b9694f33cd14e85b2e433aac2` |
| 完整 validation 文件 | `46a39bf230fa7830c4fc5d8896d818f18e0f634dd5c1a7d8d66a48b31d895da0` |
| 完整 test 文件 | `f7ae76d1d2c91904cd54e945f1154764544dd601e2f037f2ab314a2b4df28f53` |
| Epoch 1 实际批次计划 | `bbcc9be104d6218b0181f8c046d452e9a75a58742e19b84c9ecb35aad7fb6561` |
| Epoch 2 实际批次计划 | `e249447965f24c9b27b7dbd5525c28a37d702a02fc4df4cfceda5659b92ea26b` |

| 分片索引 | 原始共享数据路径 | SHA-256 |
| ---: | --- | --- |
| 12 | `training/data/processed/unicode-context-192ch-12conv-20260913-wiki/train-012.jsonl` | `95b30716ebb3222fa5fbaa82de16c3e8d95f43f1881dc02795d618edb1ece5fa` |
| 65 | `training/data/processed/unicode-context-192ch-12conv-20260913-wiki/train-065.jsonl` | `00fe9dbf50f2a8d7a3a83369a0aadfcf71ff2a90124b84edccd5b5663aa9ec7b` |
| 152 | `training/data/processed/unicode-context-192ch-12conv-20260913-wiki/train-152.jsonl` | `c4ff8dde4c06c6f6c585d544ce212087f41b2697835e903fec94121c74cc79f6` |
| 311 | `training/data/processed/unicode-context-192ch-12conv-20260913-combined/synthetic-train-055.jsonl` | `429b2a5880d05249b411e7554be2de1a22f2c2655f6f3ad29bde3504870df81c` |

共享初始参数的 SHA-256：

| 参数 | SHA-256 |
| --- | --- |
| `embedding.weight` | `cfe03ce69c7c608e43bcbd048797e99762ef54b4a38978b7b81505da08c7af01` |
| `boundary_hidden.weight` | `621f41bc8aec9692b131148e90062ce306f0742f8a4e3b3eab4207e83c07b9c2` |
| `boundary_hidden.bias` | `0ef22309e52310a326f05d1cd931cefb206c2c964191e1447bbc03d72032cf78` |
| `boundary_output.weight` | `d8234f80a211b0a05960ec9a03079b2f8fde75013cae62e4fe189885e321afb2` |
| `boundary_output.bias` | `00a3904c58bc098665eb3e4c0451b5dc1717841a8e39ef75c922cbfeb7928e69` |

实验源代码指纹（路径相对于 `training/`；专用代码现已删除）：

| 源文件 | SHA-256 |
| --- | --- |
| `architecture_experiment/compare.py` | `24ffbe2c3a40fe4faccbb78e4355c8daed08573b35ec6638defbff808d40cd93` |
| `architecture_experiment/data.py` | `8f64b07cca05d1e9a0a1e9e12c70c55a13f661fd6ceed7dc0007020ff2ac66ac` |
| `architecture_experiment/model.py` | `49bbd6cc6b371009589bd31521199fccee0c96dbf76dc916e540dd262c9b254b` |
| `architecture_experiment/run.py` | `ce544475f6423e4f8186392a900fa699591ec8eeae76198deac17dd793087ceb` |
| `architecture_experiment/test_experiment.py` | `035f5c40d56b48a145add3f1f7f3ddff12253924ae9c130735b15ea5b069c539` |
| `architecture_experiment/train.py` | `fac10a8c687ad8f357e4fa92b52a6a799a6a7cc56bc90f343647b0a7885a7431` |
| `text-policy.json` | `115ffe35795c35182e3dcbf33cdd14849fb30174ca66ae0a330384ee014d548f` |
| `text_policy.py` | `53bdb21ffb33af23bed2d9773f3ac08be46af2d94ffdb2c1cc692fb0c763c0eb` |
| `train_sharded.py` | `4af7b48ebdd15d0f7d350b8e629008bd9cf7be409cd823b23247c6179ea81847` |
| `train_smoke.py` | `673545fd73957ee9f43a2f4a8f6ad86a7b0b2072a9c0471535db0b739403cd54` |

## 清理范围

保留本 Markdown 实验记录，并由 [训练说明](README.md) 链接。
删除以下三个目录中的 30 个文件，合计 159,884,807 字节（约 159.9 MB）：

- `training/architecture_experiment/`：模型、数据适配、运行、训练、配对核验和测试脚本，以及原实验 README。
- `training/artifacts/architecture-pair-smoke-20260914/`：两组 checkpoint、日志、状态、配置和原始比较报告。
- `training/artifacts/architecture-transformer-preflight-20260914/`：预检 checkpoint、状态、配置和注意力性能记录。

同时移除 package.json 中三个 `model:architecture*` 命令和原运行说明。
共享数据分片、完整 holdout、词表、PyTorch 依赖、正式 CNN 训练与恢复工具、已接入后端的模型保留。
因此本记录保留了实验设计、主要结果和核验指纹，不包含可直接恢复该实验的 checkpoint 或专用运行代码。
