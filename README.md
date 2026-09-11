# Super Reader

用于梳理核心流程的最小 Chrome 中文阅读辅助扩展。只有一个全局工具栏开关，保存在本地，刷新页面或重启浏览器后仍保留。页面在获得焦点时应用开关，后台标签页不会因开关变化而收到消息或开始处理。直接加载源码即可运行，不需要安装依赖或构建。

在 Chrome 中按 **Shift + Option + R**（macOS；其他平台为 **Shift + Alt + R**）可切换开关。快捷键通过 manifest 的 `_execute_action` 触发同一个工具栏点击处理函数，处理中同样不响应，无需额外事件监听。重新加载扩展后生效；可在 `chrome://extensions/shortcuts` 查看或修改绑定。声明方式见 [Chrome 快捷键文档](https://developer.chrome.com/docs/extensions/reference/api/commands#action-commands)。

切回已经处理的页面时，只同步当前全局 ON/OFF 状态。状态相同就保留原有标记和处理记录，即使页面在后台期间错过了关闭再开启也无需重做。新标签页在首次聚焦时处理，当前页面刷新后重新处理。后台页面只在下次聚焦时响应，没有向所有标签页广播、遍历后台页面或后台同步任务。

## 核心流程

```text
开启 → 锁定 toggle → read() → process(texts) → write(snapshot, results) → remember(written) → 解锁
滚动 / 缩放 → 重置 200ms 防抖 → 等待当前任务完成且变化稳定 → 读取最新画面
关闭 → 停止监听和等待中的刷新 → clear() → reset()
失败 → 记录 error → 停止监听、关闭并清理 → 解锁，等待用户重试
```

一次处理的单位是任务开始时与可视区域相交、尚未处理的合格文本节点，每个节点保留完整字符串。`read()` 同步固定文本及其 DOM 映射；`process()` 将全部文本作为一个请求交给 Worker；全部结果返回后，`write()` 一次同步写入分隔标记。锁覆盖读取、推理、写入的全过程，处理中点击开关无效。

没有按字符数切分的任务批次、批次间延迟、应用层任务队列或用户取消机制。视口用于选择节点，不裁剪字符，也不限制字符串长度；单个跨多屏的长节点会被完整处理，因此数据量和运行时间没有硬保证。推理请求设有 5 秒超时，超时会终止 Worker 并报错。失败只停止当前页面，保留错误信息，不改变全局开关；切换标签页不会重试，刷新该页面，或在该页面关闭再开启后重试。

页面滚动、普通 DOM 内部滚动容器滚动、窗口及 visual viewport 的滚动和尺寸变化，以及正文容器的尺寸变化都会请求刷新。`ResizeObserver` 监听 body 尺寸，让加载完成后才出现的正文也能触发读取；初始通知和尺寸不变的通知不会重复触发。连续变化重置 200ms 防抖定时器；每页只有一个运行中的任务和一个 `refreshPending` 布尔标志，只有任务完成且定时器到期后才开始下一次读取。中间画面不排队，当前任务不取消。关闭或失败时移除监听、清除定时器和待刷新标志。已捕获的快照不随页面变化。此处“一次操作”指调度和锁的单位，不代表网页 DOM 被冻结：页面自行改变的源文本不会写入旧结果。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `src/app/reader.js` | 定义应用接口；`createReader()` 协调 read/process/write，通过 `toggle(flag)` 管理状态、开关和防抖刷新 |
| `src/frontend/read.js` | `read()` 跳过无关 DOM 分支，保留至少部分可见的完整文本节点，生成固定快照和 DOM 映射 |
| `src/frontend/dom-tree.js` | 遍历普通 DOM 和嵌套的 open Shadow DOM；跨 slot 和 shadow host 查找显示层级中的父元素 |
| `src/frontend/viewport.js` | `readViewport(document)` 捕获可视区域边界；`watchViewport(document, onChange)` 监听滚动、视口及 body 尺寸变化并返回取消监听函数 |
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
| `src/platform/chrome/background.js` | Chrome 入口：保存全局开关，聚焦及当前页面加载完成时检查状态，按需注入脚本，转发推理消息并创建隐藏页面 |
| `src/platform/chrome/content.js` | 将 Chrome 消息转换成 `reader.toggle(flag)` / `status()`，转发推理请求与状态通知 |
| `src/platform/chrome/inference.js` | 提供 Worker 地址，将 Chrome 消息连接到推理服务 |
| `src/platform/interfaces.js` | 适配器的 JSDoc 接口说明，无运行时代码 |
| `src/inference.html` | Chrome 隐藏页面，承载推理服务 |
| `src/content.css` | 分隔线的固定样式 |
| `manifest.json` | Chrome 权限、工具栏、快捷键和后台入口 |
| `icons/` | 彩色和灰色 S 图标；工具栏按全局开关切换，忙碌状态不改变外观 |
| `test/` | 单元测试、集成测试与浏览器测试页 |
| `training/` | 独立的离线训练和导出工具 |

需要切分的子句先生成一份固定的模型评分，后续拆分复用这些评分，不再次推理。在当前片段内，令 `p = 左侧视觉长度 / 总视觉长度`，以 `softmax 概率 × p × (1 - p)` 选择得分最高的允许位置。实现使用等价的 `logit + Math.log(p × (1 - p))` 比较，无需计算 softmax；没有额外权重系数。视觉长度计入汉字和数字表达式，累计长度预先计算一次，每次拆分都按当前片段重新计算比例。词语与数量保护仍排除其范围内的候选位置，加权后同分时保留第一个候选，停止阈值仍是 8 视觉单位。长子句按最多 256 个汉字的固定窗口各推理一次，相邻窗口共享一个汉字，保证每个候选边界恰好评分一次；窗口边缘不强制切分。评分只保留到本次子句处理结束。这些计算细节不拆分视口请求，也不影响 toggle 的锁。

## 模块如何连接

核心只使用标准 DOM、Worker 和 JavaScript，不调用扩展 API。Chrome 调用集中在 `src/platform/chrome/`。普通脚本通过 `globalThis.SuperReader` 暴露接口，后台按以下顺序加载页面代码：

```text
frontend/dom-tree.js → frontend/viewport.js → frontend/visibility.js → frontend/processed-text.js
  → frontend/read.js → frontend/write.js → app/reader.js
  → platform/chrome/content.js → start-reader.js
```

`start-reader.js` 注入依赖：

```js
const adapter = SuperReader.createReaderAdapter();
const reader = SuperReader.createReader({
  read: SuperReader.read,
  process: adapter.process,
  write: (snapshot, results) =>
    SuperReader.write(snapshot, results, adapter.markerStyleUrl),
  clear: SuperReader.clearMarkers,
  remember: SuperReader.rememberProcessedText,
  reset: SuperReader.clearProcessedText,
  watch: (onChange) => SuperReader.watchViewport(document, onChange),
  publishState: adapter.publishState,
});
adapter.connect(reader);
```

应用在 `app/reader.js` 中定义 `Reader` 和 `ReaderState` 接口，只提供 `status()` 和 `toggle(flag)`。`toggle(true)` 请求开启，`toggle(false)` 请求关闭。相同状态不重启；忙碌时直接忽略所有开关请求，不保存待应用设置。失败后重复 ON 不重试，收到 OFF 再 ON 或刷新页面才重试。按钮调用方负责计算目标状态，并在忙碌时忽略点击。

适配器只翻译接口：将设置消息传给 `reader.toggle(flag)`，将 `PING` 转成 `status()`，转发整个 `texts` 数组并提供分隔线样式地址。`publishState()` 仅发送通知，不再触发任何 reader 控制操作。DOM 引用始终留在页面里。当前启动代码明确加载 Chrome 适配器；其他浏览器可提供同一个应用接口的实现。

后台使用 `chrome.storage.local` 只保存 `{ enabled }`，按钮以全局开关为准，页面错误单独显示。标签页切换、浏览器窗口聚焦及当前标签页加载完成时，只查询当前窗口的活动标签页；`PING` 检查页面是否已有 reader，随后通过 `APPLY_SETTING` 传递目标布尔值。reader 负责忽略相同状态和忙碌时的开关请求。若聚焦时 reader 仍在处理，当前同步请求会被忽略；任务完成后不会补做同步，下次空闲时的聚焦检查再应用全局状态。当前页面忙碌时，按钮点击不响应；其他后台页面不会被查询。

后台同时只做一次活动页检查。检查期间若再次聚焦或页面刷新完成，只设一个 `checkPending` 标志；当前检查结束后重新查询最新活动页和当前文档，不保存中间标签页队列。旧文档的延迟状态回复不再让新文档错过检查。

自动处理新网站需要 manifest 中的 HTTP、HTTPS 和本地文件访问权限，替代原来只覆盖用户点击页面的 `activeTab`；修改 manifest 后需要重新加载扩展。本地文件还需打开“允许访问文件网址”。浏览器自身页面等受限页面不注入。相关权限要求见 [Chrome 内容脚本文档](https://developer.chrome.com/docs/extensions/develop/concepts/content-scripts#inject-programmatically)。

Shadow DOM 中的标记加载同一个 `src/content.css`，该文件在 manifest 中声明为可访问资源，前端无需调用 Chrome API。

Chrome 的推理消息路径：

```text
页面适配器 → Chrome 后台 → inference.html → inference-service → Worker → backend
```

隐藏页面和 Worker 按需创建，多个页面共享模型；服务使用请求 ID 匹配各页的结果。全局关闭立即清理当前页面，后台页面保留原样，直到下次聚焦再清理。已经开始的任务仍会完成，原有滚动和缩放处理逻辑不变。Worker 超时或崩溃会拒绝共享 Worker 上尚未完成的请求，后续请求可创建新 Worker。

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

当前 writer 用 `splitText()` 拆分原始文本节点。若页面保留原节点引用，并在渲染后把该节点的 `nodeValue` 替换成整段新文字，旧的后续片段仍会留在 DOM 中，造成新旧内容混合。处理前的快照检查无法防止这种后续更新；解决它需要调整标记渲染方式。

`processed-text.js` 的记录是弱引用，但 writer 为关闭时清理而保存的标记 `Set` 和 shadow 样式 `Map` 是强引用。已脱离页面的标记和其关联子树仍可能保留到关闭或刷新页面；大量替换内容的长会话需要关注这部分内存。

当前没有 DOM 内容变化观察器。新增正文或评论若改变 body 尺寸，会通过 `ResizeObserver` 请求处理；不改变 body 尺寸的文本、属性或 slot 分配变化仍需等下一次滚动、缩放或重新开启。只在 shadow root 内传播的滚动事件也不会触发刷新。读取和写入直接调用 DOM 工具；关闭或失败时会断开尺寸观察器。

当前处理当前文档的普通 DOM 和 open Shadow DOM；不进入 closed shadow root 或 iframe，不判断遮挡层、CSS 蒙版或任意旋转后的裁剪形状。新 shadow root 在下次读取时发现。不能仅凭父元素在屏幕外就跳过整个分支，因为其定位后代仍可能可见；`visibility:hidden` 也可能被后代覆盖。初次发现节点的耗时仍受文档规模和浏览器布局成本影响。

## 安装和验证

1. 在 `chrome://extensions/` 开启开发者模式，加载本目录。
2. 打开普通网页，点击 Super Reader 工具栏按钮或按 Shift + Option + R（macOS）。开启时显示彩色 S 图标，关闭时显示灰色版本，不显示 `ON` 标签。处理中保持相同外观和标题，点击不响应、不排队；`ERR` 表示失败，悬停可看错误。图标通过 [Chrome action API](https://developer.chrome.com/docs/extensions/reference/api/action#icon) 切换。
3. 切到其他普通网页会沿用全局开关；切回已处理页面不会重复处理。刷新当前页面也会沿用开关。
4. 处理完成后再次点击全局关闭。后台标签页不会收到通知，下次聚焦时才清理。使用本地文件时，需要开启扩展的“允许访问文件网址”。

```bash
npm test
python3 -m http.server 8765 --bind 127.0.0.1
```

- `demo.html`：普通测试文章，使用真实扩展按钮操作。
- `http://127.0.0.1:8765/test/browser-fixture.html`：普通网页适配器，运行真实 Worker 和模型，显示请求次数、输入长度和状态变化。
- `http://127.0.0.1:8765/test/dom-read-fixture.html`：点击 **Run layout tests**，验证真实浏览器中的长节点、视口边界、内部滚动刷新、处理中缩放、加载完成后出现的正文及标记写入清理。
- `http://127.0.0.1:8765/test/shadow-dom-fixture.html`：点击 **Run shadow DOM tests**，验证嵌套评论、slot、宿主裁剪、视口事件触发后的内容更新、样式和完整清理；确认不改变 body 尺寸的内容变化与 shadow root 内部滚动本身不触发处理。

离线训练见 [`training/README.md`](training/README.md)。
