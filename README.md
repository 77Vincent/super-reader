# Super Reader

用于梳理核心流程的最小 Chrome 中文阅读辅助扩展。只有工具栏开关，作用于当前网页；刷新页面后默认关闭。直接加载源码即可运行，不需要安装依赖或构建。

## 核心流程

```text
开启 → 锁定 toggle → read() → process(texts) → write(snapshot, results) → remember(written) → 解锁
滚动 / 缩放 / 内容变化 → 重置 200ms 防抖 → 等待当前任务完成且变化稳定 → 读取最新画面
关闭 → 停止监听和等待中的刷新 → clear() → reset()
失败 → 记录 error → 停止监听、关闭并清理 → 解锁，等待用户重试
```

一次处理的单位是任务开始时与可视区域相交、尚未处理的合格文本节点，每个节点保留完整字符串。`read()` 同步固定文本及其 DOM 映射；`process()` 将全部文本作为一个请求交给 Worker；全部结果返回后，`write()` 一次同步写入分隔标记。锁覆盖读取、推理、写入的全过程，处理中点击开关无效。

没有按字符数切分的任务批次、批次间延迟、应用层任务队列或用户取消机制。视口用于选择节点，不裁剪字符，也不限制字符串长度；单个跨多屏的长节点会被完整处理，因此数据量和运行时间没有硬保证。推理请求设有 5 秒超时，超时会终止 Worker 并报错。失败后保留错误信息、停止处理，用户可再次点击重试。

页面滚动、内部滚动容器滚动、窗口及 visual viewport 的滚动和尺寸变化都会请求刷新。相关文本增删、修改和可见性属性变化也使用同一条刷新路径。连续变化重置 200ms 防抖定时器；每页只有一个运行中的任务和一个 `refreshPending` 布尔标志，只有任务完成且定时器到期后才开始下一次读取。中间画面不排队，当前任务不取消。关闭或失败时移除监听、清除定时器和待刷新标志。已捕获的快照不随页面变化。此处“一次操作”指调度和锁的单位，不代表网页 DOM 被冻结：页面自行改变的源文本不会写入旧结果。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `src/app/reader.js` | `createReader()`：协调 read/process/write，管理 enabled、busy、error、toggle 和防抖刷新 |
| `src/frontend/read.js` | `read()` 跳过无关 DOM 分支，保留至少部分可见的完整文本节点，生成固定快照和 DOM 映射 |
| `src/frontend/dom-tree.js` | 遍历普通 DOM 和嵌套的 open Shadow DOM；跨 slot 和 shadow host 查找显示层级中的父元素 |
| `src/frontend/changes.js` | 观察文档及已发现 shadow root 的内容变化，报告刷新需求；隔离插件自己的同步写入，释放已移除根的监听 |
| `src/frontend/viewport.js` | `readViewport(document)` 捕获可视区域边界；`watchViewport(document, onChange)` 监听变化并返回取消监听函数 |
| `src/frontend/visibility.js` | `createVisibilityFilter(document, viewport)` 提供 `shouldSkipSubtree(element)` 和 `getVisibleArea(node)`；负责分支排除、隐藏判断及祖先溢出裁剪，样式缓存仅用于本次读取 |
| `src/frontend/processed-text.js` | 独立记录已处理文本节点及其当前值、父节点；由 reader 在写入成功后记录、关闭或失败时重置 |
| `src/frontend/write.js` | 校验源节点、插入标记及 shadow root 样式，返回成功处理的文本片段；`clearMarkers()` 清理，包括已脱离文档的根 |
| `src/backend/chunker.js` | `process(texts)` 返回每段文本的 UTF-16 分隔位置；负责标点分句、词语保护和模型驱动的递归切分 |
| `src/backend/inference.js` | 纯 JavaScript CNN 数值计算，只接受模型数据输入 |
| `src/boundary-model-data.js` | 导出的模型结构、词表和权重 |
| `src/inference-worker.js` | Worker 入口，将一个完整文本数组交给 backend，返回一份完整结果 |
| `src/inference-service.js` | 管理标准 Worker 生命周期、请求与响应对应、超时及错误 |
| `src/start-reader.js` | 每页的通用装配入口：将 DOM 工具和适配器注入 reader，再连接控件 |
| `src/start-inference.js` | 推理宿主的通用装配入口：创建服务、连接宿主消息 |
| `src/platform/chrome/background.js` | Chrome 入口：工具栏、脚本注入、状态显示、消息转发和隐藏页面创建 |
| `src/platform/chrome/content.js` | 将 Chrome 消息转换成 process、publishState、toggle/status 调用 |
| `src/platform/chrome/inference.js` | 提供 Worker 地址，将 Chrome 消息连接到推理服务 |
| `src/platform/interfaces.js` | 适配器的 JSDoc 接口说明，无运行时代码 |
| `src/inference.html` | Chrome 隐藏页面，承载推理服务 |
| `src/content.css` | 分隔线的固定样式 |
| `manifest.json` | Chrome 权限、工具栏和后台入口 |
| `test/` | 单元测试、集成测试与浏览器测试页 |
| `training/` | 独立的离线训练和导出工具 |

每次切分先排除词语与数量保护范围内的候选位置，再选择模型原始评分最高的位置；同分时保留第一个候选。切分层仍使用 8 视觉单位的递归停止阈值，以及长句评分时的 256-token 窗口；这些计算细节不拆分视口请求，也不影响 toggle 的锁。

## 模块如何连接

核心只使用标准 DOM、Worker 和 JavaScript，不调用扩展 API。Chrome 调用集中在 `src/platform/chrome/`。普通脚本通过 `globalThis.SuperReader` 暴露接口，后台按以下顺序加载页面代码：

```text
frontend/dom-tree.js → frontend/viewport.js → frontend/visibility.js → frontend/processed-text.js
  → frontend/read.js → frontend/write.js → frontend/changes.js → app/reader.js
  → platform/chrome/content.js → start-reader.js
```

`start-reader.js` 注入依赖：

```js
const adapter = SuperReader.createReaderAdapter();
const changes = SuperReader.createDOMChanges(document);
const reader = SuperReader.createReader({
  read: () => SuperReader.read(changes.observeRoot),
  process: adapter.process,
  write: (snapshot, results) => changes.mutate(() =>
    SuperReader.write(snapshot, results, adapter.markerStyleUrl)),
  clear: () => changes.mutate(SuperReader.clearMarkers),
  remember: SuperReader.rememberProcessedText,
  reset: SuperReader.clearProcessedText,
  watch(onChange) {
    const stopDOM = changes.watch(onChange);
    const stopViewport = SuperReader.watchViewport(document, onChange);
    return () => { stopViewport(); stopDOM(); };
  },
  publishState: adapter.publishState,
});
adapter.connect(reader);
```

阅读器对外只有 `status()` 和 `toggle()`。适配器发布状态、连接控件、转发整个 `texts` 数组并提供分隔线样式地址；DOM 引用始终留在页面里。后台依据阅读器的状态通知更新按钮，避免滞后的 toggle 回复覆盖新状态。Shadow DOM 中的标记加载同一个 `src/content.css`，该文件在 manifest 中声明为可访问资源，前端无需调用 Chrome API。

Chrome 的推理消息路径：

```text
页面适配器 → Chrome 后台 → inference.html → inference-service → Worker → backend
```

隐藏页面和 Worker 按需创建，多个页面共享模型；服务使用请求 ID 匹配各页的结果。关闭一页只清理该页。Worker 超时或崩溃会拒绝共享 Worker 上尚未完成的请求，后续请求可创建新 Worker。

普通网页测试页提供另一套适配器，复用相同的 reader、DOM、启动入口和推理实现。以后支持其他浏览器时，可替换适配器和安装配置；当前仅实现 Chrome 适配。`jsconfig.json` 与 JSDoc 用于编辑器提示，无需构建。

## DOM 数据接口

`read()` 返回固定快照：

```js
{
  texts: ["需要处理的可见中文"],
  sources: [{ node, parent, text, start, end }],
  viewport: { left, top, right, bottom }
}
```

`sources` 与 `texts` 一一对应。`text` 保存读取时完整的节点值，`start` 固定为 `0`，`end` 为完整字符串的 UTF-16 长度；`texts[i]` 等于 `source.text.slice(start, end)`。`process(texts)` 返回同样顺序的 `number[][]`，每个偏移严格递增、位于对应输入内部，并保留完整的 Unicode 字符。

`read.js` 调用 `dom-tree.js`，使用 `TreeWalker` 遍历元素和文本，并递归进入 open shadow root。遇到代码、控件、可编辑区域或不能显示的分支时，直接跳过其后代；对其余文本节点，只要 `Range.getClientRects()` 的任一文本矩形与允许显示的区域相交，就保留整个节点，一节点一条输入，不再查找可见字符的起止位置。仅保留含中文的字符串；快照中的 `parent` 可以是元素或直接容纳文本的 shadow root。

`visibility.js` 判断分支是否应排除，并计算视口与祖先滚动容器的矩形交集；`viewport.js` 捕获视口尺寸。对应的独立测试位于 `test/visibility.test.js` 和 `test/viewport.test.js`，`test/dom-read.test.js` 验证分支跳过、整节点选择和快照行为。

写入前检查节点连接、父节点和完整原文。分隔位置从后往前写入，原有标签与文本保留。writer 只返回成功处理的文本片段，由 reader 调用 `remember()` 记录；没有分隔点的节点也会记录，过期快照跳过的节点不会记录。后续读取跳过未变化的已处理节点，避免重叠视口重复推理或插入重复标记；空快照不调用后端。关闭时移除标记、合并相邻文本节点并清空已处理记录，再次开启会重新处理。处理记录使用 `WeakMap`。

`changes.js` 使用 `MutationObserver` 观察普通文档和每个已发现的 open shadow root，因此 Bilibili 这类嵌套 Web Component 中后来加载的评论也能请求刷新。观察器在我们的同步写入期间暂停，写入前已排队的页面变化仍会保留；推理期间持续观察。Shadow root 内部的滚动和 slot 分配变化也会请求刷新。根被移除时释放其观察和事件监听，关闭时释放全部监听。

当前处理当前文档的普通 DOM 和 open Shadow DOM；不进入 closed shadow root 或 iframe，不判断遮挡层、CSS 蒙版或任意旋转后的裁剪形状。若页面仅在已有 host 上调用 `attachShadow()`，且没有其他可观察变化，新根会在下次滚动或刷新读取时发现。不能仅凭父元素在屏幕外就跳过整个分支，因为其定位后代仍可能可见；`visibility:hidden` 也可能被后代覆盖。初次发现节点的耗时仍受文档规模和浏览器布局成本影响。

## 安装和验证

1. 在 `chrome://extensions/` 开启开发者模式，加载本目录。
2. 打开普通网页，点击 Super Reader 工具栏按钮。`…` 表示整屏处理中，`ON` 表示完成并开启，`ERR` 表示失败，悬停可看错误。
3. 处理完成后再次点击关闭。使用本地文件时，需要开启扩展的“允许访问文件网址”。

```bash
npm test
python3 -m http.server 8765 --bind 127.0.0.1
```

- `demo.html`：普通测试文章，使用真实扩展按钮操作。
- `http://127.0.0.1:8765/test/browser-fixture.html`：普通网页适配器，运行真实 Worker 和模型，显示请求次数、输入长度和状态变化。
- `http://127.0.0.1:8765/test/dom-read-fixture.html`：点击 **Run layout tests**，验证真实浏览器中的长节点、视口边界、内部滚动刷新、处理中缩放及标记写入清理。
- `http://127.0.0.1:8765/test/shadow-dom-fixture.html`：点击 **Run shadow DOM tests**，验证嵌套评论、slot、宿主裁剪、动态加载和修改、推理期间的更新、内部滚动、样式和完整清理。

离线训练见 [`training/README.md`](training/README.md)。
