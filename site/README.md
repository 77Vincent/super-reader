# Super Reader 静态入口页

独立的 Hugo 项目。页面内容、配置、模板、样式和静态资源均位于本目录，不依赖扩展的构建和模型文件。暂未选择或安装主题，`layouts/` 与 `assets/` 提供最小的自定义页面。

## 本地预览

安装 Hugo **0.158.0 或更新版本**，不要求 Extended 版本。以下命令在仓库根目录运行：

```sh
hugo server --source site --bind 127.0.0.1
```

浏览器打开终端输出的本地地址（默认 `http://localhost:1313/`）。修改 Markdown、模板或样式后会自动更新。

## 构建

```sh
hugo build --source site --cleanDestinationDir --minify
```

输出位于 `site/public/`，将该目录部署到静态托管服务即可。生成的 HTML 包含正文，不需要 JavaScript 渲染。`public/`、`resources/` 和 Hugo 锁文件已在本目录的 `.gitignore` 中排除。

## 发布前配置

编辑 `site/hugo.toml`：

- `baseURL`：把 `https://example.org/` 替换为实际站点地址，包含协议和末尾斜杠。若部署在子路径中，也需要写入子路径。
- `params.chromeWebStoreURL`：填写实际商店详情页链接。未填写时，首页按钮指向 GitHub 的项目与安装方法。
- `title`、`params.description`：网站名称和默认描述；首页自己的标题、描述在 `content/_index.md` 中维护。

也可以在构建时指定部署地址，例如：

```sh
hugo build --source site --cleanDestinationDir --minify --baseURL https://example.org/super-reader/
```

Canonical、Open Graph 和 sitemap 使用实际构建时的 `baseURL`。已启用 Hugo 的 sitemap 生成，并提供带 sitemap 地址的 `robots.txt`。404 页面设置了 `noindex`。子路径部署时，爬虫读取的是域名根目录的 `/robots.txt`，需在托管站点的根配置中加入对应 sitemap 地址。

此项目不包含部署凭据或自动发布流程。

## 目录

| 路径 | 用途 |
| --- | --- |
| `hugo.toml` | 站点配置 |
| `content/_index.md` | 首页文案 |
| `archetypes/default.md` | 新内容的默认模板 |
| `layouts/` | 基础页面、首页、普通页面、404 和元信息模板 |
| `assets/css/main.css` | 最小样式，构建时压缩并生成文件指纹 |
| `static/favicon.png` | 复用插件图标的站点图标 |

以后选择主题时，注意项目中的 `layouts/` 优先于主题模板；应删除或调整对应的最小模板，让主题接管渲染。

采用 Hugo 官方的[目录结构](https://gohugo.io/getting-started/directory-structure/)、[基础模板与页面模板](https://gohugo.io/templates/types/)、[sitemap](https://gohugo.io/templates/sitemap/) 和 [robots.txt](https://gohugo.io/templates/robots/) 配置方式。
