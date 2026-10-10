# 好读静态入口页

独立的 Hugo + [Doks](https://github.com/thuliteio/doks) 项目。包含产品首页、隐私政策和 404 页面，不生成 docs、blog、分类、标签、RSS 或搜索索引。内容在构建时生成静态 HTML，主题的 JavaScript 用于明暗切换、移动端导航等交互。

主题通过官方 npm 包 `@thulite/doks-core` 安装，按 [Doks 官方安装方式](https://getdoks.org/docs/start-here/installation/) 和 Thulite 模块挂载配置接入。主题、构建依赖和锁文件都位于 `site/`，与插件的根目录 npm 项目独立。

配置参考官方 Doks starter，其 MIT 许可保留在 `LICENSE-Doks.txt`。

默认品牌文字标志为「好｜读」，由 `layouts/_partials/brand-wordmark.html` 统一生成。导航栏和移动端菜单通过本地 `header/header.html` 覆盖使用它；该文件保留 Doks 1.9.3 的导航模板，仅替换这两处品牌标题，升级主题时需同步检查。品牌分隔线与正文共用颜色及随字号缩放的比例，固定显示，不受阅读辅助开关影响。站点名称、页面元数据中的品牌名仍为「好读」。

网站 favicon 与插件图标统一为透明底的红色粗竖线，颜色为 `#ff1744`；在仓库根目录运行 `npm run icons:generate` 可同时重新生成 SVG、插件各尺寸 PNG 和 `static/favicon.png`。

首页的切分示例是静态演示，不加载模型或运行推理。在 `content/_index.md` 的 `reader-demo` shortcode 内用 `｜` 指定示意断点；构建时生成分隔线。安装入口位于顶部中央，使用 `btn-primary`；阅读辅助开关固定在视口底部中央，使用带实色底和阴影的 `btn-outline-primary`，适配手机底部安全区域；悬浮按钮脱离文档流，不额外增加页面底部或页脚高度。开关统一控制简介及所有演示段落中的分隔线，主标题和品牌分隔线始终显示。卡片保留自定义的 [CSS Scan #32](https://getcssscan.com/css-box-shadow-examples) 纸张阴影，间距使用 Bootstrap 的 `p-3 p-md-4`。没有 JavaScript 时仍显示静态示例，开关隐藏。

演示引用可用 `{{< reader-demo source="罗素《幸福之路》" >}}` 指定来源，模板自动添加 `-- ` 前缀。来源在卡片内渲染为正文之后的独立段落，使用 Bootstrap 的 `text-end mb-0` 靠右对齐；不指定 `source` 时不生成来源段落。

首页主标题 `headline` 和副标题 `lead` 也用 `｜` 指定固定断点，构建时由 `layouts/_partials/reader-copy.html` 转为与演示相同的红色分隔线，不显示字面量竖线，也不运行模型。每个语意块优先保持完整，分隔线随前一块一起换行，避免单独落在行首。`title` 和 `seo.title` 保留不带分隔标记的纯文本，供页面元信息使用。

官网与扩展的分隔线均以桌面正文的 18px 字号、3px 线宽为基准：高度 `1em`、宽度 `calc(1em / 6)`，间距和垂直偏移也使用 `em`。标题和小字按同一比例缩放；使用实色伪元素绘制，避免边框宽度取整改变细线比例。

站点布局和样式优先使用 Bootstrap 预定义类，包括栅格、间距、排版、背景和按钮；标题语意块使用 `d-inline-block mw-100`，正文使用 `lh-lg text-break`。自定义 CSS 的例外是阅读分隔线绘制规则、演示卡片的纸张纹理与阴影，以及悬浮开关的底色、层级和安全间距，均保留在 `assets/scss/common/_custom.scss`。纹理用内嵌 SVG 生成单色颗粒，以 200px 方块平铺；SVG 中的 `opacity='0.05'` 控制颗粒强度，不影响文字透明度。浅色模式使用暖白纸面和 multiply 纹理，深色模式跟随 Bootstrap 的 `--bs-tertiary-bg`，用 screen 混合淡色颗粒及更适合深色背景的阴影；文字继承主题配色。后续样式调整也优先组合现有类，避免在这些例外之外新增项目 CSS 或内联样式。

全站通过 `layouts/_partials/head/custom-head.html` 在 HTML 的 `<head>` 中输出 `<meta name="super-reader" content="off">`。即使扩展全局开启，也会跳过这些页面，不运行切分推理，避免影响静态演示开关；其他网站仍按全局设置处理。此标记不依赖域名或页面 JavaScript，部署后同样生效。已安装旧版开发扩展时，重新加载扩展及页面后该规则生效。

## 环境与本地预览

- Hugo **Extended 0.158.0 或更新版本**（Doks 使用 SCSS）。
- Node.js **24.13.0 或更新版本**，推荐 Node.js 24 LTS。

以下命令在仓库根目录运行：

```sh
npm --prefix site ci
npm --prefix site run dev
```

开发预览通过 `--renderToMemory` 在内存中提供页面和资源，与生产构建的 `public/` 分开。`config/development/hugo.toml` 同时将开发资源的编译缓存放到已忽略的 `resources/development/`，避免生产构建的 `--cleanDestinationDir` 或 `--gc` 删除预览正在使用的文件。可以在预览运行时执行构建。

打开终端输出的本地地址（默认 `http://localhost:1313/`）。如果端口已占用：

```sh
npm --prefix site run dev -- --port 1315
```

## 构建

```sh
npm --prefix site run build
```

将 `site/public/` 部署到静态托管服务即可。`public/`、`node_modules/`、`resources/`、`hugo_stats.json` 和 Hugo 锁文件均已忽略；`package-lock.json` 应提交，保证干净检出后可通过 `npm ci` 安装相同依赖。

正式发布推荐从仓库根目录运行：

```sh
npm run release:site -- --base-url https://正式域名/
```

该入口必须指定真实 HTTPS 地址，输出到独立的 `dist/site/`，并检查隐私页、扩展跳过标记、sitemap 和 robots 地址。它不会修改开发配置。域名确定后再生成最终产物；部署只上传 `dist/site/` 的内容。详细步骤见 [`../release/README.md`](../release/README.md)。

尚未确定域名时运行 `npm run release:site:preview`，生成 `dist/site-preview/`，用本地静态服务器预览即可。预览产物禁用抓取并标记 noindex；正式产物在指定真实域名后重新生成。

## 发布前配置

- `config/_default/hugo.toml` 的 `baseURL`：把 `https://example.org/` 替换为正式地址，包含协议、部署子路径（如有）和末尾斜杠。
- `config/_default/params.toml` 的 `chromeWebStoreURL`：首页安装按钮的固定地址。目前使用含 `REPLACE_WITH_EXTENSION_ID` 的 Chrome 商店占位链接；发布后替换为好读的完整商店详情页链接，再重新构建、部署网站。
- `content/_index.md`：维护首页标题、SEO 标题、描述和正文。

首页安装入口统一显示“现在安装”，所有浏览器都直接打开配置的商店详情页，不检测浏览器，也不依赖 JavaScript。

也可以在构建时指定地址：

```sh
npm --prefix site run build -- --baseURL https://example.org/super-reader/
```

页面使用 Doks/Thulite 的 SEO 集成生成元信息和 sitemap，保留自定义的 `robots.txt`。404 页面设有 `noindex`。子路径部署时，爬虫读取域名根目录的 `/robots.txt`，需在根配置中添加对应 sitemap 地址。

## 维护

| 路径 | 用途 |
| --- | --- |
| `config/_default/` | Hugo、Doks、SEO、导航和 npm 模块挂载配置 |
| `config/postcss.config.js` | 官方 PostCSS 配置，压缩生产 CSS 并保留动态主题样式 |
| `content/_index.md` | 产品首页 |
| `content/privacy.md`、`layouts/legal.html` | 隐私政策和简单的正文布局 |
| `layouts/home.html` | 使用 Doks 基础模板和 Bootstrap 栅格的产品首页 |
| `layouts/_partials/` | 项目扩展入口、站点图标、404 的 noindex |
| `assets/scss/common/` | 保留阅读分隔线和卡片纸张阴影，其余使用 Bootstrap 预定义类 |
| `package.json`、`package-lock.json` | 固定版本的主题依赖和构建命令 |

没有复制官方示例的 docs、blog 等页面。隐私政策按本项目实际的数据流编写。文档搜索、侧栏、版本切换等功能在 `params.toml` 中关闭；页脚包含隐私政策入口。

升级主题时更新依赖与锁文件并重新构建，检查首页、404、明暗切换和移动端菜单即可。不修改 `node_modules` 中的主题源码。

当前 Doks 1.9.3 及其上游依赖在 Hugo 0.164.0 下仍会输出 `LanguageCode` 和 LibSass 的弃用提示，正常构建成功。将来升级 Hugo 时需同时检查主题的兼容性。
