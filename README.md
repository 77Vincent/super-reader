# 好读

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

页面滚动、普通 DOM 内部滚动容器滚动、窗口及 visual viewport 的滚动和尺寸变化，以及正文容器的尺寸变化都会请求刷新。`ResizeObserver` 监听 body 尺寸；`MutationObserver` 监听文本和子节点变化，捕捉尺寸不变的内容替换。两者都只请求重新读取当前可视区域，未变化的已处理文本仍会跳过。连续变化重置 200ms 防抖定时器；每页只有一个运行中的任务和一个 `refreshPending` 布尔标志，只有任务完成且定时器到期后才开始下一次读取。中间画面不排队，当前任务不取消。关闭或失败时移除监听、清除定时器和待刷新标志。已捕获的快照不随页面变化。此处“一次操作”指调度和锁的单位，不代表网页 DOM 被冻结：页面自行改变的源文本不会写入旧结果。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `src/app/reader.js` | 定义应用接口；`createReader()` 协调 read/process/write，通过 `toggle(flag)` 管理状态、开关和防抖刷新 |
| `src/frontend/read.js` | `read()` 跳过无关 DOM 分支，保留至少部分可见的完整文本节点，生成固定快照和 DOM 映射 |
| `src/frontend/dom-tree.js` | 遍历普通 DOM 和嵌套的 open Shadow DOM；跨 slot 和 shadow host 查找显示层级中的父元素 |
| `src/frontend/viewport.js` | `readViewport(document)` 捕获可视区域边界；`watchViewport(document, onChange)` 监听滚动、视口及 body 尺寸变化并返回取消监听函数 |
| `src/frontend/content-changes.js` | 观察文档及读取时发现的 open shadow root 的文本和子节点变化；请求刷新，并在同步写入标记时暂停观察 |
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
| `src/content.css` | 分隔线样式：高度 `1em`、宽度 `calc(1em / 6)`，按所在文字的字号等比例缩放 |
| `manifest.json` | Chrome 权限、工具栏、快捷键和后台入口 |
| `icons/` | 彩色和灰色 S 图标；工具栏按全局开关切换，忙碌状态不改变外观 |
| `test/` | 单元测试、集成测试与浏览器测试页 |
| `training/` | 独立的离线训练和导出工具 |

超过 12 个视觉单位的子句才调用模型；视觉长度计入汉字、数字表达式和 Latin 单词。长子句按最多 256 个 Unicode token 的固定窗口各推理一次，相邻窗口共享一个 token，保证每个候选边界恰好评分一次。后续递归保留这些原始 logits，只在当前子句内部的全部 gap 上重算 softmax，默认只允许概率严格大于 **50%** 的位置切分。模型切点在原文中紧邻的左右字符都必须是汉字，且须通过浏览器分词的词内保护；符合条件的位置直接按模型原始 logit 排序，同分时取当前片段内靠前的位置。过滤受保护的位置不会再次归一化概率。

每次切开后，对仍超过 12 个视觉单位的左右子句继续处理；置信度不足就停止。当前模型按每条样本一个目标断点训练，softmax 是相对概率；递归缩小候选范围可能提高分数，50% 不是实际正确率保证。`src/backend/chunker.js` 中的 `MIN_SPLIT_CONFIDENCE` 是运行默认值；`chunkText`、`chunkTextByClause`、`buildVisualChunks` 和 `process` 无需传参即可使用递归 softmax。评分只保留到本次原始子句处理结束；这些计算不拆分视口请求，也不影响 toggle 的锁。

当前算法、demo 结果及离线对比入口见 [递归 softmax](training/RECURSIVE_SOFTMAX.md)。

## 模块如何连接

核心只使用标准 DOM、Worker 和 JavaScript，不调用扩展 API。Chrome 调用集中在 `src/platform/chrome/`。普通脚本通过 `globalThis.SuperReader` 暴露接口，后台按以下顺序加载页面代码：

```text
frontend/dom-tree.js → frontend/viewport.js → frontend/visibility.js → frontend/processed-text.js
  → frontend/read.js → frontend/write.js → frontend/content-changes.js → app/reader.js
  → platform/chrome/content.js → start-reader.js
```

`start-reader.js` 注入依赖：

```js
const adapter = SuperReader.createReaderAdapter();
const changes = SuperReader.createContentChanges(document, SuperReader.markDirtySources);
const reader = SuperReader.createReader({
  read: () => {
    changes.withoutObservation(() => {
      const { forgotten, preserved } = SuperReader.removeStaleMarkers();
      SuperReader.forgetProcessedText(forgotten);
      SuperReader.rememberProcessedText(preserved);
    });
    return SuperReader.read(changes.observeRoot);
  },
  process: adapter.process,
  write: (snapshot, results) => changes.withoutObservation(() =>
    SuperReader.write(snapshot, results, adapter.markerStyleUrl)),
  clear: SuperReader.clearMarkers,
  remember: SuperReader.rememberProcessedText,
  reset: SuperReader.clearProcessedText,
  watch(onChange) {
    const stopContent = changes.watch(onChange);
    const stopViewport = SuperReader.watchViewport(document, onChange);
    return () => { stopContent(); stopViewport(); };
  },
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

写入前检查节点连接、父节点和完整原文。分隔位置从后往前写入，原有标签与文本保留。writer 只返回成功处理的文本片段，由 reader 调用 `remember()` 记录；没有分隔点的节点也会记录，过期快照跳过的节点不会记录。后续读取跳过未变化的已处理节点，避免重叠视口重复推理或插入重复标记；空快照不调用后端。关闭时按原始文本节点分组撤销写入并清空已处理记录，再次开启会重新处理。处理记录使用 `WeakMap`。

writer 使用现有 `markedSources` 集合保存每个原始文本节点的切分记录：`sequence[0]` 是原节点 A，其余文本是衍生片段，记录同时包含分隔标记及切分后的文本值。页面对成员的文字写入会标记 dirty，即使赋值与当前片段相同；单独移走再放回文本成员也会留下移动记录。插件自己的同步写入不参与标脏。

自动刷新和关闭共用按组清理规则：整组文字未改且文本成员完整、有序时，`unwrite` 把原文恢复到同一个 A，再删除衍生片段和 divider；A 被改写且衍生片段未被改动时，`discard` 保留 A 的新值，只删除旧衍生内容。A 被删除且剩余片段仍完整、未被改动时，也丢弃旧片段，不重新插回 A。只有 divider 被移除或移动时，仍可还原完整文本组。任何路径都不对父元素调用 `normalize()`，同一父元素下的其他原始文本节点保持独立。

兼容约定是：页面写入 A 的新值代表整条原始文本的新完整内容，空字符串代表整条清空；仅修改原节点前半段的页面不适用此约定。若页面修改了衍生片段、独立移动了文本成员或插入节点打断了组内结构，则只移除该组 divider，保留现存文字及节点身份。装配层将这些仍连接的冲突文本记为已处理，暂停插线，直到其文字或父节点再次变化；关闭后重新开启也会重新评估。`removeStaleMarkers()` 返回待清除处理记录的 `forgotten` 和需要暂停的 `preserved`，装配层先清除、再记住；`read()` 本身仍然只读。行内 `splitText()` 仍会改变网页 DOM，这些规则不保证兼容所有基于子节点位置的页面操作。

`processed-text.js` 的记录是弱引用，writer 保存的切分记录、标记和 shadow 样式则是强引用。过期切分记录和标记在下一次读取前释放；shadow 样式保留到关闭或刷新页面。

`content-changes.js` 观察文档及已发现的 open shadow root 的 `childList` 和 `characterData`。先把页面变化交给 `markDirtySources`，已有切分组的变化会请求清理，包括 ASCII 片段的相同值写入；其余变化仍按中文内容过滤。含中文的文本新增、移除、替换和修改、旧值含中文但新值被清空、页面自行移除分隔标记、新增带 open shadow root 的宿主都会请求读取。未涉及已有分组的脚本、样式、代码、控件、可编辑区域及纯数字计时变化不触发内容刷新。不观察属性：尺寸不变的样式、可见性或 slot 分配变化仍需等下一次视口事件。只在 shadow root 内传播的滚动事件也不会触发刷新。

装配层在同步写入及清理期间暂停内容观察，操作前先处理已经排队的页面变化，操作后立即恢复观察，避免标记造成反馈循环；页面随后自行修改 DOM 仍会被捕捉。关闭或失败时，先消费尚未送达的页面变化，再断开观察器并按组清理，避免立即关闭时遗漏 dirty 状态。内容观察器会释放脱离文档的 shadow root。

当前处理当前文档的普通 DOM 和 open Shadow DOM；不进入 closed shadow root 或 iframe，不判断遮挡层、CSS 蒙版或任意旋转后的裁剪形状。新 shadow root 在下次读取时发现；如果已存在的宿主稍后才调用 `attachShadow()`，且没有已观察到的 DOM 或尺寸变化，则仍需等下一次视口事件。不能仅凭父元素在屏幕外就跳过整个分支，因为其定位后代仍可能可见；`visibility:hidden` 也可能被后代覆盖。初次发现节点的耗时仍受文档规模和浏览器布局成本影响。

## 安装和验证

1. 在 `chrome://extensions/` 开启开发者模式，加载本目录。
2. 打开普通网页，点击好读工具栏按钮或按 Shift + Option + R（macOS）。开启时显示彩色 S 图标，关闭时显示灰色版本，不显示 `ON` 标签。处理中保持相同外观和标题，点击不响应、不排队；`ERR` 表示失败，悬停可看错误。图标通过 [Chrome action API](https://developer.chrome.com/docs/extensions/reference/api/action#icon) 切换。
3. 切到其他普通网页会沿用全局开关；切回已处理页面不会重复处理。刷新当前页面也会沿用开关。
4. 处理完成后再次点击全局关闭。后台标签页不会收到通知，下次聚焦时才清理。使用本地文件时，需要开启扩展的“允许访问文件网址”。

`npm test` 需要 Node.js 和 Python 3.9+（含 pip）。Parquet 语料测试在缺少依赖时会自动安装固定版本 `pyarrow==21.0.0` 到忽略提交的 `training/.deps/`，首次安装需要联网；后续运行复用已有依赖。测试自行生成微型语料，不需要下载训练数据，也不会因缺少 PyArrow 而跳过测试。

```bash
npm test
python3 -m http.server 8765 --bind 127.0.0.1
```

- `http://127.0.0.1:8765/test/browser-fixture.html`：普通网页适配器，运行真实 Worker 和模型，显示请求次数、输入长度和状态变化。
- `http://127.0.0.1:8765/test/dom-read-fixture.html`：点击 **Run layout tests**，验证真实浏览器中的长节点、视口边界、内部滚动刷新、处理中缩放、加载完成后出现的正文及标记写入清理。
- `http://127.0.0.1:8765/test/shadow-dom-fixture.html`：点击 **Run shadow DOM tests**，验证同尺寸标题替换后自动恢复标记、晚到及更新的嵌套评论、处理中合并更新、避免标记反馈循环、slot、宿主裁剪、样式和完整清理。

离线训练见 [`training/README.md`](training/README.md)。
