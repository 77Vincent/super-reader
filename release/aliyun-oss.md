# haodu.site 阿里云静态站点部署

本站统一使用阿里云：域名为 `haodu.site`，网站由上海 OSS 静态托管。默认发布入口为 `npm run deploy:site`。

目标域名为 `https://haodu.site/`。2026-09-29 已将生产构建的 21 个静态文件（共 286,494 字节）通过官方 ossutil 2.4.0 上传至 OSS Bucket `haodu-site`，并逐个通过 ETag 校验。Bucket 位于华东 2（上海，`cn-shanghai`），使用标准存储、本地冗余 LRS，静态托管配置已保存。文件部署已完成；Bucket 当前仍为私有，域名尚未公开上线。中国内地 Bucket 需要域名完成 ICP 备案后才能绑定上线。

已领取并成功签发 1 张零元 DigiCert 个人测试证书（基础版、RSA 2048），覆盖 `haodu.site` 和 `www.haodu.site`，有效期为 2026-09-29 至 2026-12-28。联系人由用户填写，`_dnsauth.haodu.site` 的 TXT 验证记录已添加并通过验证。证书尚未部署到 OSS，签发完成不代表站点已支持 HTTPS。免费证书没有自动续期，需要在到期前重新申请并部署。

## 发布文件

在仓库根目录运行：

```sh
npm run release:site -- --base-url https://haodu.site/
```

仅上传 `dist/site/` 内的文件，保持目录结构。不要上传整个仓库、插件或训练文件。`site/config/production/hugo.toml` 同时记录了正式站点地址；单独运行 `release:site` 时仍需显式指定域名，`deploy:site` 已固定使用 `https://haodu.site/`。

## API 发布

使用 [ossutil 2.0](https://help.aliyun.com/zh/oss/developer-reference/ossutil-overview/)；当前 Mac 的安装位置为 `~/.local/bin/ossutil`，下载包已按官方 SHA-256 校验。其他环境按官方文档安装 ossutil 2.x 到 PATH，或通过 `--ossutil /path/to/ossutil` 指定。

已创建专用 RAM 用户 `haodu-site-deploy`，仅绑定自定义策略 `HaoduSiteDeploy`（`v1`）。云端策略及用户授权列表均已核对，内容与 [aliyun-site-policy.json](aliyun-site-policy.json) 一致：只允许对 `haodu-site/*` 调用 `PutObject` 和 `GetObject`，用于上传和校验，不授予删除、修改权限配置或访问其他 Bucket 的权限。

本机已配置 `~/.ossutilconfig` 中的 `haodu-site` profile，文件权限为 `600`。凭证保存在仓库外，不写入仓库或聊天；本机直接运行下面的发布命令即可。

首次配置其他机器时，使用交互式向导填写部署用户的 AccessKey ID、AccessKey Secret、地域 `cn-shanghai`；Endpoint 可留空：

```sh
~/.local/bin/ossutil config --profile haodu-site
chmod 600 ~/.ossutilconfig
```

运行发布预览（不需要凭证，也不调用 OSS）：

```sh
npm run deploy:site -- --dry-run
```

执行实际上传（本机凭证已配置）：

```sh
npm run deploy:site -- --profile haodu-site
```

CI 可通过密钥存储注入 `OSS_ACCESS_KEY_ID`、`OSS_ACCESS_KEY_SECRET`；使用 STS 时还需 `OSS_SESSION_TOKEN`，此时运行 `npm run deploy:site` 即可。不要把密钥直接放进命令行参数。

发布脚本重新构建生产站点，只上传 `dist/site/` 内容，保持路径并设置 MIME 和缓存头。资源先上传、HTML 后上传、首页最后上传，每次上传后用 `HeadObject` 的 `If-Match` 校验本次单次 `PutObject` 生成的 ETag；失败立即停止。它保留云端旧资源，避免影响仍使用旧页面的访客。该命令只负责文件发布，不会自动更改 Bucket 权限、域名解析或证书配置，也不能代替备案后的网站访问验证。

## OSS 配置

| 配置 | 值 |
| --- | --- |
| Bucket | `haodu-site` |
| 地域 | 华东 2（上海，`cn-shanghai`） |
| Endpoint | `oss-cn-shanghai.aliyuncs.com` |
| 用途 | 仅存放本站公开静态文件的独立 Bucket |
| 存储类型 | 标准存储 |
| 冗余类型 | 本地冗余 LRS |
| 当前访问权限 | 私有，阻止公共访问已开启；备案通过并准备上线时再启用公共读 |
| 默认首页 | `index.html` |
| 子目录首页 | 开通，使 `/privacy/` 返回 `privacy/index.html` |
| 文件 404 规则 | Redirect，使 `/privacy` 跳转到 `/privacy/` |
| 默认 404 页 | `404.html` |
| 错误文档响应码 | 404 |
| 自定义域名 | `haodu.site`，尚未绑定 |

默认首页、子目录首页、文件 404 规则、错误页与响应码已在控制台保存。备案通过并准备上线时，只对这个专门存放公开网站文件的 Bucket 关闭阻止公共访问，再设置公共读，不修改账号级别或其他 Bucket 的访问限制。

按控制台提供的目标值，为 `haodu.site` 添加 CNAME 解析，并绑定覆盖该域名的 HTTPS 证书。OSS 默认域名不能替代正常的网站访问域名。域名解析或证书尚未就绪时，不能算完成上线。

上传时保留正确的 Content-Type。HTML 使用 `Cache-Control: no-cache`，带内容指纹的 CSS/JS 可使用 `public, max-age=31536000, immutable`；其他文件使用短缓存。更新时先上传资源，最后上传 HTML，避免页面引用尚未上传的文件。

## 上线验证

- `https://haodu.site/` 与 `/privacy/` 正常返回 HTML，HTTPS 证书有效。
- `/privacy` 跳转到 `/privacy/`；不存在的路径显示项目错误页，并返回 HTTP 404。
- 页面引用的 CSS、JS、字体和图标都能从本站加载。
- 安装入口、阅读辅助开关、明暗模式和手机布局正常。
- sitemap 与 robots 引用 `https://haodu.site/`，首页和隐私页没有预览环境的 noindex。

## 备案与后续上线

通过阿里云提交首次 ICP 备案前，需要先购买或关联符合备案条件的中国内地云产品。仅购买域名或创建大陆 OSS Bucket 不会取得备案服务码；OSS 本身也不在可直接申请备案服务码的云产品列表中。这里要求先具备接入资源，不要求网站先公开上线。首次备案期间的网站关闭要求应按备案流程执行。具体条件见 [备案服务器检查](https://help.aliyun.com/zh/icp-filing/basic-icp-service/user-guide/icp-filing-server-access-information-check) 和 [备案期间对网站访问的影响](https://help.aliyun.com/zh/icp-filing/the-influence-of-the-record-during-the-site-visit)。

完成 ICP 备案后，使用现有上海 Bucket 绑定 `haodu.site`，按控制台提供的 CNAME 目标配置解析，部署有效证书并开放公共读，随后完成上述上线验证。若未来更换地域，需要新建 Bucket 并迁移；现有 Bucket 的地域不能直接修改。

参考：[OSS 静态网站托管](https://help.aliyun.com/zh/oss/user-guide/hosting-static-websites)、[绑定自定义域名](https://help.aliyun.com/zh/oss/user-guide/access-buckets-via-custom-domain-names)、[Bucket 地域不能修改](https://help.aliyun.com/zh/oss/user-guide/0015-00000227)。
