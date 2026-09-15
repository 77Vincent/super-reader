# 默认递归 softmax

2026-09-15：正式后端默认采用 **75% 门槛、原始 logits 缓存、递归 softmax**。
使用现有正式 16 层、192 通道 epoch 1 模型，位置加权已移除。

## 运行方式

1. 标点预分句后，超过 12 个视觉单位的原始子句才运行 CNN。
   每个最多 256-token 的固定窗口评分一次，相邻窗口共享一个 token。
2. 缓存合并后的全部 gap logits。在当前片段内部的全部 gap 上计算 softmax。
3. 只有概率严格大于 75%、符合原始词语/数量/标点保护、两侧都有视觉内容的点可选。
   按原始 logit 取最高分，同分取靠前点；过滤保护点后不再次归一化。
4. 对超过 12 个视觉单位的左右子句继续处理。只重算各自的 softmax，不重新推理。
   没有合格切点即停止。源文本、Unicode 偏移和原始保护范围始终保留。

使用显式栈处理递归。长输入始终沿用最初固定窗口的上下文，子句切开不会改变窗口。
缓存仅在当前原始子句的处理期间有效。

```js
chunker.process(texts); // Worker 和 demo 的正式默认入口

// 离线对比入口：
chunker.process(texts, { minConfidence: 0.75, scoringStrategy: "fixed" });
chunker.process(texts, { minConfidence: 0.75, scoringStrategy: "recursive-model" });
```

`scoringStrategy` 默认是 `recursive-softmax`；`fixed` 保留整句概率，
`recursive-model` 对子句重新运行 CNN。`minConfidence: 0` 可关闭置信度门槛。

## Demo 结果

`｜` 为模型新增切点。加粗文本保持独立 DOM 文本节点。

> 中文天然不使用空格切分语意块，而仅用标点断句。因此｜过长的的句子｜会破坏阅读体验。请感受本文里断句的出现｜是否让你的阅读更轻松。

> **切分阅读**使用神经网络，在长句中｜找到符合人类习惯的断句点，把长句切成更短的语意块｜提升阅读效率。

> 即便是完全符合语法的｜通顺的｜但没有任何标点｜断句的句子模型依然能够找到恰当的切分点。

三段、四个文本节点，共 **8 个新增切点、6 次 CNN 调用、122 个累计输入 token**。
实际 Worker 的无选项请求包含此回归检查；长句测试检查原始窗口只运行一次、右侧索引
和补充 Unicode 字符的 UTF-16 偏移。

203 项测试通过。Chrome 152 中，从 demo 的真实 DOM 文本节点经正式 inference service
和 Worker 发起无选项请求，冷、热两次结果均与上述切点一致；首次含模型加载约 405 ms。
本地验证记录位于 `training/artifacts/recursive-softmax-backend-20260915/chrome-worker-verification.json`。

这批文本没有人工完整切分标签，第三段仍存在不自然的切分。候选范围缩小可以提高
相对概率，模型尚无“不切分”类别；75% 不等于真实正确率。
[原始输入的全量置信度验证](CONFIDENCE_THRESHOLD.md)不能直接用于递归后续切点。
