# Chrome Web Store 首发材料

基于当前 `manifest.json` 的 0.1.0 版；首发 Chrome Web Store，Edge 商店地址暂留空。发布材料与实际提交进度分别记录，不能把本地构建成功当成商店发布成功。

## 当前进度（2026-09-30）

- `npm test`：254 项通过，无失败或跳过。
- `npm run release:extension`：生成 `dist/haodu-0.1.0.zip`，27 个运行文件，约 14 MB；`unzip -t` 校验通过。
- ZIP SHA-256：`3768328404b0298ff9cc8ac88efc03d1a6c5435ed482c1376c2bc3b388ed5207`。
- 已核对源代码：模型与运行脚本均随包提供，开关写入 `chrome.storage.local`，没有向外部服务发送正文的代码。
- 公开隐私政策与支持链接已具备，440 × 280 小宣传图已准备。
- 尚未上传或提交审核：用户已手动打开开发者后台并确认登录，但 Chrome 仍拒绝浏览器工具读取或截图，返回 `The extensions gallery cannot be scripted.`。这不是登录问题；当前工具无法代为操作这个商店页面。需要手动创建项目、上传 ZIP 并填写首次上架资料。账号注册及验证状态无法从受限页面确认。
- 现有商店预览图是官网静态演示。另提供不含阅读脚本的真实扩展截图用例页，实际扩展截图和最终安装验证仍待完成。

## 名称与摘要

名称：好读 - 中文辅助阅读

摘要（与 manifest 一致）：按人类习惯，将长句拆分为短句，让阅读更轻松。

语言：简体中文。选择与阅读辅助最接近的商店分类。

## 详细介绍

让中文更好读。

中文长句缺少词语之间的空格，阅读时有时难以找到停顿。好读使用浏览器内运行的微型神经网络，为网页长句添加细竖向分隔线，把文字分成更容易阅读的语意块。

- 保留原文：添加视觉分隔，不改写、不总结网页内容。
- 本地运行：模型随扩展提供，网页正文不上传到外部 AI 服务。
- 按需处理：从当前可视区域内的合格中文文本开始，滚动后处理新出现的内容。
- 随时切换：通过工具栏按钮或快捷键开启、关闭；开关设置保存在本机。

安装后，将好读固定到工具栏，打开包含中文正文的普通网页并点击图标。红色竖线图标表示开启，灰色表示关闭。也可以按 Alt + Shift + R 切换；macOS 使用 Option + Shift + R。

分隔由模型自动判断，可能出现不理想的切分。浏览器内部页面、商店等受限页面无法运行扩展；本地 HTML 文件需另外开启“允许访问文件网址”。

## 单一用途

在用户开启后，为网页中的中文长句添加视觉分隔线，帮助用户按较短的语意块阅读，保留原有正文。

## 权限解释

| 字段 | 可填写的说明 |
| --- | --- |
| `scripting` | 向当前符合条件的网页注入阅读脚本和分隔线样式，并在关闭时移除阅读标记。 |
| `offscreen` | 在本地隐藏扩展页面运行模型 Worker，使多个网页共享模型，避免在每个页面重复加载权重。 |
| `storage` | 仅在 `chrome.storage.local` 保存全局开启或关闭状态，不保存网页正文、浏览历史或推理结果。 |
| 网站访问权限 | 开启后，在用户切换到普通 HTTP/HTTPS 网页时自动应用阅读辅助，因此需要访问这些页面；`activeTab` 不能覆盖未再次点击工具栏的新网页。本地文件只有在用户另行允许后才处理。所有推理留在本机。 |
| 远程代码 | 不使用远程代码。脚本、样式、模型词表和权重均随上传的 ZIP 打包，没有外部脚本加载或云端推理。 |

## 数据处理与审核说明

扩展读取网页文本用于当前本地推理，不向开发者或第三方传输正文、URL、历史记录或使用统计。除开关设置外，没有持久化的用户数据。商店隐私字段应与网站 `/privacy/` 一致；按提交时表单定义填写，不把“本地读取正文”描述成“完全不接触用户数据”。

网站地址：<https://github.com/77Vincent/super-reader>。正式官网 `https://haodu.site/` 尚未公开上线，暂不填写不可访问的网址。

隐私政策地址：<https://github.com/77Vincent/super-reader/blob/main/site/content/privacy.md>。该文件已在公开仓库中可访问，与站点使用同一份政策；官网上线后可改为 `https://haodu.site/privacy/`。

支持地址：<https://github.com/77Vincent/super-reader/issues>

审核人员无需账号、密码或付费。安装后打开普通中文网页，点击工具栏图标开启，再次点击关闭。产品官网包含静态演示并主动禁止扩展处理，不能用官网验证真实模型注入；应使用普通中文内容页面。

## 图片材料

- 128 × 128 图标：`icons/on-128.png`，已包含在 ZIP。
- 440 × 280 小宣传图：[store-promo-440x280.jpg](assets/store-promo-440x280.jpg)，对应可编辑源文件 [store-promo-440x280.svg](assets/store-promo-440x280.svg)。
- 首张预览图：[store-preview-1280x800.jpg](assets/store-preview-1280x800.jpg)，1280 × 800，已保存，可用于商店 Overview 上方的图片展示区。按照 2026-09-29 用户选定的截图内容，保留标题、本地神经网络简介、两个按钮及第一张纸张示例卡片。
  - 图片说明：官网阅读辅助预览。该图展示官网的静态演示，不声称是此时运行扩展模型得到的结果。
  - 用户原附件为 1706 × 894 的内存截图，没有磁盘原文件。当前版本取自本地页面的同一组内容，按商店尺寸重新排版并截图，不是原附件的逐像素副本。
- 真实扩展截图用例：[store-reading-preview.html](assets/store-reading-preview.html)。该页没有阅读脚本，也没有人工插入的正文分隔线。通过本地 HTTP 服务打开后，开启待发布扩展，确认分隔线实际出现，再按 1280 × 800 截图。不能将未处理页面或人工标记页面描述为扩展运行结果。

```sh
python3 -m http.server 8096 --bind 127.0.0.1 --directory release/assets
# 打开 http://127.0.0.1:8096/store-reading-preview.html
```

![好读官网阅读辅助预览](assets/store-preview-1280x800.jpg)

## 官方要求

- [上传包准备](https://developer.chrome.com/docs/webstore/prepare)：ZIP 根目录必须直接包含 manifest。
- [隐私字段与权限解释](https://developer.chrome.com/docs/webstore/cws-dashboard-privacy)。
- [图片规格](https://developer.chrome.com/docs/webstore/images)。
- [提交发布](https://developer.chrome.com/docs/webstore/publish/)：在开发者后台上传 ZIP、补齐材料并提交审核。
- [官方 API](https://developer.chrome.com/docs/webstore/api/reference/rest)：V2 的上传接口针对已有项目，没有首次创建项目或填写商店文案、隐私表单的接口。[API 使用说明](https://developer.chrome.com/docs/webstore/using-api)仍要求在开发者后台完成商店详情与隐私字段；API 无法替代首次上架的全部后台步骤。
