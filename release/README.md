# 好读发布

网站与插件独立发布。网站只上传静态页面；插件只上传运行文件。不要把仓库根目录或整个 `dist/` 上传到任意一方。

当前准备方向：Chrome Web Store 首发；站点优先考虑中国大陆的访问体验，正式域名和托管地域待确定。此目录记录准备步骤，不表示已经上线或通过商店审核。

## 1. 构建插件

环境：Node.js 24.13+ 和 `zip`（macOS 自带；Ubuntu 可安装 `zip`）。无需 npm 依赖、Python 或任何训练数据。

```sh
npm run release:extension
```

产物：

- `dist/extension/`：加载已解压的扩展程序时选择这个目录。
- `dist/haodu-0.1.0.zip`：提交商店的 ZIP，名称取自 manifest 版本。
- `dist/haodu-0.1.0.zip.sha256`：ZIP 校验值。
- `dist/extension-release.json`：版本、体积、每个文件的 SHA-256 清单，不放入 ZIP。

`scripts/build-extension.mjs` 明确列出发布文件，拒绝符号链接，检查 JS 语法及 manifest 引用，并固定文件时间和权限。ZIP 不含顶层 `extension/` 目录，也不含 training、site、测试或开发配置。新增运行文件时，需要同时更新打包清单。

构建直接复制当前 `src/boundary-model-data.js`，不会重新导出或替换正在使用的模型。后续升级时同步修改根 `package.json` 和 `manifest.json` 的版本，再构建新的 ZIP。

## 2. 构建站点

已验证工具：Hugo Extended 0.164.0、Node.js 25.9.0；主题依赖由 `site/package-lock.json` 锁定。Node.js 24.13+ 也在项目声明的支持范围内。

域名尚未确定时，先构建可本地预览的完整静态文件：

```sh
npm --prefix site ci
npm run release:site:preview
python3 -m http.server 8088 --bind 127.0.0.1 --directory dist/site-preview
```

访问 `http://127.0.0.1:8088/`。`dist/site-preview/` 使用站点根路径、noindex 和禁止抓取的 robots，只用于检查产物，不能作为最终 SEO 发布包。正式域名确定后重新运行以下命令：

```sh
npm --prefix site ci
npm run release:site -- --base-url https://正式域名/
```

CI 也可以通过 `SITE_BASE_URL` 传入地址。正式地址必须包含 HTTPS；若有子路径，包含子路径。发布入口会拒绝缺失地址及常见占位域名。`site/config/_default/hugo.toml` 的本地默认占位地址不用于正式发布。

将 **`dist/site/` 内的文件**上传到静态网站根目录，保留 CSS、JS、字体与图片路径结构。不要上传 `site/` 源码目录或插件 ZIP。首页为 `index.html`，错误页为 `404.html`，隐私页为 `privacy/index.html`。

托管服务要支持目录首页（`/privacy/` → `privacy/index.html`）和真正的 404 状态，不要配置所有路径回退到首页。为 HTML、CSS、JS、SVG、PNG、字体设置正确的 Content-Type。HTTPS 证书必须覆盖正式域名。

HTML 建议 `Cache-Control: no-cache`；带内容指纹的 CSS/JS 可以长缓存；favicon 等无指纹资源使用短缓存。此站点所有构建资源均由本站提供，部署不依赖浏览器访问 npm、GitHub 或外部字体 CDN。

## 3. 中国访问优先的托管选择

- 已有备案域名：可使用阿里云 OSS 大陆地域静态托管，按需要配合大陆 CDN。
- 暂无备案：可考虑 OSS 香港地域和自有域名，先做大陆多网络访问测试；它仍是跨境访问，不能保证各运营商的延迟和稳定性。
- 自有域名、HTTPS、目录首页和错误页配置，在上传文件后仍需配置并实测。具体控制台/CLI部署步骤待域名和地域确定后补齐。

依据：[阿里云自定义域名与备案要求](https://www.alibabacloud.com/help/en/oss/user-guide/access-buckets-via-custom-domain-names)。普通 Cloudflare 托管不等于大陆网络；其[中国网络需企业套餐另购](https://developers.cloudflare.com/china-network/get-started/)。

## 4. 上架顺序

1. 确定正式域名、托管平台及地域；部署站点，确认首页、`/privacy/`、404、CSS/JS、明暗切换、演示按钮均正常。
2. 用 `dist/extension/` 做最终安装验证，避免与旧的开发版同时启用。检查开关、页面刷新、滚动、跨标签页和本地推理；官网应保持静态演示。
3. 在 Chrome Web Store 开发者后台创建项目，上传 ZIP，参考 [商店材料](chrome-web-store.md) 填写文案、权限、隐私页和支持链接，并补齐真实截图与宣传图。
4. 获得商店项目 ID 后，保留它作为后续升级的同一项目；不要把本地开发扩展的 ID 当成商店 ID。
5. 商店详情页可公开访问后，填写 `site/config/_default/params.toml` 的 `chromeWebStoreURL` 并重新部署站点。`edgeAddonsURL` 暂留空，现有逻辑让桌面 Edge 使用同一个 Chrome 商店链接。

首次提交需要开发者账号及后台要求的账号验证。网站先上线、商店审核后再更新安装按钮，中间阶段按钮继续链接到项目安装说明，不填写虚构的商店地址。

## 尚需提供或完成

- 正式域名、选定的托管服务及部署账号配置。
- Chrome Web Store 开发者账号和上架材料最终确认。
- 小宣传图及真实扩展使用截图。
- 上线后的域名访问、HTTPS 与商店安装验证。

本地构建、ZIP 校验和单元测试不等于商店审核通过。
