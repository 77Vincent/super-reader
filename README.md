# Super Reader

用于梳理核心流程的最小 Chrome 中文阅读辅助扩展。只有工具栏开关，作用于当前网页；没有 popup、快捷键、全局开关或样式配置。刷新页面后默认关闭。

## 运行流程

```text
点击工具栏按钮
  → background.js 按需注入 content.js 和 content.css
  → content.js 收集当前网页的中文文本节点
  → 每批一个不超过 128 个 UTF-16 单位的字符串
  → background.js 转发到隐藏的 inference.html
  → inference-service.js 交给专用 Worker
  → inference-worker.js 调用 chunker.js / model-backend.js
  → 返回分隔位置，content.js 同步插入空标记，content.css 画线
```

每批从请求到 DOM 插入完成期间禁止切换。批次之间留出 50ms 调度间隔，允许关闭；关闭会清空后续批次并移除当前页分隔线。其他页面的开关独立。推理服务设有 5 秒超时保护，失败会结束处理并允许重试；这些数值限制工作量和等待，不是浏览器墙钟耗时的硬保证。

扩展始终在浏览器本地计算。隐藏页面和 Worker 按需创建并共享复用，关闭一个页面的阅读效果不会销毁共享模型。

## 职责边界

| 文件 | 职责 |
| --- | --- |
| `manifest.json` | 权限、工具栏按钮、后台入口 |
| `src/background.js` | 当前页按钮、按需注入、转发推理和显示状态 |
| `src/content.js` | 文本采集、批次开关锁、结果映射和清理 |
| `src/content.css` | 固定红色竖线，所有显示样式都在这里 |
| `src/inference.html` | 隐藏页面，承载推理服务 |
| `src/inference-service.js` | 请求 ID、Worker 生命周期、超时与错误返回 |
| `src/inference-worker.js` | 检查批次大小，调用算法并返回分隔位置 |
| `src/chunker.js` | 标点分句、词语保护、模型评分与递归切分 |
| `src/model-backend.js` | 纯 JavaScript CNN 数值计算 |
| `src/boundary-model-data.js` | 导出的模型结构、词表和权重 |
| `training/` | 独立的离线数据准备、训练和导出工具 |

消息只传文本和分隔位置，推理层没有 DOM 或显示配置。之后恢复样式选择时，只增加展示层的配置更新，不接入分块、队列或 Worker 流程。

## 当前范围

骨架只处理开启时已有的普通 DOM 文本节点，保留标签与原文，跳过代码、表单、隐藏和可编辑区域。每个文本节点独立处理，过长节点从后向前拆成有界输入，批次之间不额外画线；跨标签或跨批次的模型上下文暂不合并。

动态内容监听、可见区域调度、结果缓存和 Shadow DOM 适配暂时移除，后续可以分别加回。现有模型、分块算法与训练工具保留。

## 安装和验证

1. 在 `chrome://extensions/` 开启开发者模式，加载本目录。
2. 打开普通网页，点击工具栏中的 Super Reader 按钮。`ON` 表示开启，`…` 表示当前批次处理中，`ERR` 表示失败，悬停可看错误。
3. 再次点击关闭。使用本地文件时，需要开启扩展的“允许访问文件网址”。

`demo.html` 是普通测试文章，使用真实扩展按钮操作。`test/browser-fixture.html` 是浏览器测试页，用一个按钮模拟工具栏，并复用真实 content、推理服务和 Worker。

```bash
npm test
python3 -m http.server 8765 --bind 127.0.0.1
```

随后访问 `http://127.0.0.1:8765/test/browser-fixture.html`。

离线训练见 [`training/README.md`](training/README.md)。
