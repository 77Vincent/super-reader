# 好读静态入口页

独立的 Hugo + [Doks](https://github.com/thuliteio/doks) 项目。仅保留产品首页和 404 页面，不生成 docs、blog、分类、标签、RSS 或搜索索引。内容在构建时生成静态 HTML，主题的 JavaScript 用于明暗切换、移动端导航等交互。

主题通过官方 npm 包 `@thulite/doks-core` 安装，按 [Doks 官方安装方式](https://getdoks.org/docs/start-here/installation/) 和 Thulite 模块挂载配置接入。主题、构建依赖和锁文件都位于 `site/`，与插件的根目录 npm 项目独立。

配置参考官方 Doks starter，其 MIT 许可保留在 `LICENSE-Doks.txt`。

首页的切分示例是静态演示，不加载模型或运行推理。在 `content/_index.md` 的 `reader-demo` shortcode 内用 `｜` 指定示意断点；构建时生成分隔线，按钮只控制这些标记的显示和隐藏。开关位于纸张卡片外的右上方，卡片使用纯色纸面和 [CSS Scan #32](https://getcssscan.com/css-box-shadow-examples) 阴影，不设边框或圆角；深色模式仅调整纸面底色。没有 JavaScript 时仍显示静态示例。

首页主标题 `headline` 和副标题 `lead` 也用 `｜` 指定固定断点，构建时由 `layouts/_partials/reader-copy.html` 转为与演示相同的红色分隔线，不显示字面量竖线，也不运行模型。每个语意块优先保持完整，分隔线随前一块一起换行，避免单独落在行首。演示按钮只控制演示段落，主标题和副标题始终显示分隔线。`title` 和 `seo.title` 保留不带分隔标记的纯文本，供页面元信息使用。

官网与扩展的分隔线均以桌面正文的 18px 字号、3px 线宽为基准：高度 `1em`、宽度 `calc(1em / 6)`，间距和垂直偏移也使用 `em`。标题和小字按同一比例缩放；使用实色伪元素绘制，避免边框宽度取整改变细线比例。

首页用 `<meta name="super-reader" content="off">` 让本项目的扩展跳过这个页面，避免真实扩展影响演示开关。已安装旧版开发扩展时，重新加载扩展及页面后该规则生效。

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

## 发布前配置

- `config/_default/hugo.toml` 的 `baseURL`：把 `https://example.org/` 替换为正式地址，包含协议、部署子路径（如有）和末尾斜杠。
- `config/_default/params.toml` 的 `chromeWebStoreURL`、`edgeAddonsURL`：分别填写好读的 Chrome、Edge 商店详情页链接。尚未上架时保持为空，首页按钮指向 GitHub 的项目与安装说明。
- `content/_index.md`：维护首页标题、SEO 标题、描述和正文。

首页始终只有一个安装入口，由少量本地 JavaScript 根据浏览器调整文案与链接：桌面 Chrome 使用 Chrome 商店；桌面 Edge 优先使用 Edge 商店，未配置时使用 Chrome 商店（Edge 支持从其他商店安装扩展）。没有可用商店链接、移动端或未识别的浏览器显示安装说明。识别优先使用 `userAgentData.brands`，再使用 User-Agent，不发送检测请求。禁用 JavaScript 时仍有可用链接，按 Chrome 商店、Edge 商店、项目说明的顺序选择。

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
| `content/_index.md` | 唯一的产品内容页 |
| `layouts/home.html` | 使用 Doks 基础模板和 Bootstrap 栅格的产品首页 |
| `layouts/_partials/` | 项目扩展入口、站点图标、404 的 noindex |
| `assets/scss/common/` | 小范围样式调整，保留 Doks 默认配色和排版 |
| `package.json`、`package-lock.json` | 固定版本的主题依赖和构建命令 |

没有复制官方示例的 docs、blog、privacy 等页面。文档搜索、侧栏、版本切换等功能在 `params.toml` 中关闭；导航仅链接首页各部分及 GitHub。

升级主题时更新依赖与锁文件并重新构建，检查首页、404、明暗切换和移动端菜单即可。不修改 `node_modules` 中的主题源码。

当前 Doks 1.9.3 及其上游依赖在 Hugo 0.164.0 下仍会输出 `LanguageCode` 和 LibSass 的弃用提示，正常构建成功。将来升级 Hugo 时需同时检查主题的兼容性。
