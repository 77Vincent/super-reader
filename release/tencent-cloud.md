# haodu.site 腾讯云部署

2026-09-29 决定迁移到腾讯云。域名注册与续费继续留在阿里云，其余目标为腾讯云：SCF 承载站点、腾讯云办理首次 ICP 备案、DNSPod 解析、腾讯云加速与 HTTPS。站点仍然是 Hugo 生成的静态文件，不增加数据库或业务后端。

## 当前进度

- 已完成腾讯云 API 发布脚本、SCF 静态服务及本地生产打包验证。`npm run deploy:site` 现在以腾讯云为目标。
- 已通过 API 创建上海 `ap-shanghai/default/haodu-site` Web 函数并上传站点。云端 24 个文件（21 个静态文件、清单和 2 个运行文件）逐个校验一致；线上 21 个静态文件的字节、SHA-256、Content-Type，以及首页、隐私页、目录跳转和 404 均已验证。
- 用户充值 10 元后，费用中心实测余额为 `9.98` 元。已领取 0 元个人高级版试用套餐 `pkg-iUPCpgOn`，上海地域所有命名空间共享，生效时间为 2026-09-29 23:39:15，到期时间为 2026-12-29 23:39:14（北京时间），未开启自动续费。110 元备案包尚未购买。
- SCF 服务授权已完成。CAM 用户 `haodu-site-deploy` 已创建并绑定 `HaoduSiteScfDeploy`，本机专用凭证已配置，API 更新验证成功；首次创建的临时权限已撤回。用户创建时产生但未配置到本机的旧密钥已禁用，保留本机正在使用的密钥。
- 测试端点为 `https://1316174341-1kyobpv1zo.ap-shanghai.tencentscf.com`（默认流量别名）。实测首页返回 200 和正确内容，但平台附加 `Content-Disposition: attachment`，浏览器会下载 HTML，不能作为可正常浏览的预览站点。该端点仅用于部署验证，未接入生产域名。
- ICP 备案、DNSPod、加速域名、HTTPS 与阿里云注册商处的 NS 切换均尚未完成。`haodu.site` 尚未公开上线。
- 原阿里云私有 Bucket `haodu-site` 的静态文件与部署凭证仍保留。旧发布入口改为 `npm run deploy:site:oss`；迁移成功前不删除旧资源。

## 为什么由 SCF 承载

[腾讯云 SCF 备案文档](https://cloud.tencent.com/document/product/583/45477)明确支持网站托管于腾讯云大陆 Serverless 的场景。单独 COS 不是首次备案所需的合格资源；“仅买 SCF 包而网站只托管 COS”的适用性没有得到确认。因此使用 SCF 实际提供本站文件，避免将备案包当作与网站托管无关的服务码。

已在登录后的[购买页面](https://buy.cloud.tencent.com/scf)核实备案资源包为一次性 **110 元 / 5 年**，包含 5000 万次事件函数调用和 40 万 GBs。5 年是资源包有效期，不是备案有效期；用量耗尽也会失效。本站采用 Web 函数，调用次数与响应流量需按其规则另计，该包的事件函数调用额度不能承诺抵扣 Web 函数调用。参考[资源包说明](https://cloud.tencent.com/document/product/583/61679)和[SCF 定价](https://cloud.tencent.com/document/product/583/12281)。

初始函数使用 Node.js 20.19、128 MB、10 秒请求超时，无预置实例、不自动创建 CLS 日志主题、关闭函数主动访问外网。函数 URL 的入站访问需另外配置；关闭主动外网访问不替代函数 URL 鉴权。备案包、流量与后续加速费用分别计费。

## 免费试用的范围

已领取的[新客试用](https://cloud.tencent.com/document/product/583/12282)按月提供事件函数和 Web 函数各 100 万次调用、100 万 GBs 资源使用量、2 GB 外网出流量。**Web 函数响应流量不在免费额度内**，不能把页面访问承诺为完全免费。超额或试用结束后按量计费；未开启自动续费不代表停止按量计费。试用套餐也不能替代首次 ICP 备案要求的 110 元备案资源包。

## 构建与 API 发布

仅构建、查看发布目标（不需要凭证，不调用云 API）：

```sh
npm run deploy:site -- --dry-run
```

产物：

- `dist/site/`：正式域名 `https://haodu.site/` 的静态文件。
- `dist/tencent-site/`：静态文件、SHA-256 清单与 SCF 启动文件。
- `dist/haodu-site-scf.zip`：可直接通过腾讯云 API 上传的完整代码包。
- `dist/tencent-site-release.json`：包校验值与文件清单，不含密钥。

打包只读取重新构建的 `dist/site/`，拒绝隐藏项、符号链接和未支持的文件类型。不会上传仓库源码、插件包或训练数据。SCF 只提供清单内的文件，支持 `/privacy` 到 `/privacy/` 的跳转、目录首页、真正的 404、HEAD、ETag 与内容缓存。代码和清单自身不对网站访客公开。

发布使用官方 [TCCLI](https://cloud.tencent.com/document/product/440/34011)。本机安装位置为 `~/.local/share/haodu-tencent-tools/bin/tccli`；其他机器可安装到 PATH，或用 `--tccli /path/to/tccli` 指定。

使用专用 CAM 用户并关联 [tencent-site-policy.json](tencent-site-policy.json)，只允许更新、读取及校验本站上海函数。策略中的主账号 UIN 是本站部署账号；迁移到其他账号时必须替换。此策略不包含创建、购买、删除、其他函数或 DNS 管理权限。

首次创建接口实测还检查 `scf:CreateFunction` 对 `qcs::scf:sh:uin/100003835317:function/*` 的权限，不能用仅指向待创建函数的策略完成。本次只在创建期间临时授予该权限并设置半小时到期条件，创建后已从当前策略移除。需要重建时由管理员执行，或重新提供短期创建权限，不能将其长期留在日常部署策略里。

在仓库外配置 `haodu-site` profile；凭证文件 `~/.tccli/haodu-site.credential` 设为 `600`。不要在聊天或命令行参数中粘贴密钥。本机已配置并验证可用。

```sh
# 重建时才需要 --create，以及管理员配置的短期创建权限。
npm run deploy:site -- --create

# 日常更新。
npm run deploy:site
```

脚本只操作 `ap-shanghai/default/haodu-site`，API 上传后等待函数处于 Active。SCF 会重新打包 ZIP，导致容器 SHA-256 改变；脚本在哈希不同时通过 SDK 返回的临时 COS 地址下载云端代码，先确认下载包与 `CodeSha256` 一致，再逐个比较文件名和内容，并验证启动文件的可执行权限。临时地址不会打印，下载包验证后删除。任何鉴权、上传或校验错误都会停止。它不会修改 DNS、购买套餐或自动宣称域名上线。

## 后续云端配置

1. 已完成控制台身份验证、充值、专用 CAM 用户及本机凭证、SCF 服务授权。
2. 已通过 API 部署并验证所有文件和在线 HTTP 响应。后续用 `npm run deploy:site` 更新。
3. 购买已确认的 110 元 SCF 备案包，按照腾讯云的 Serverless 方式提交 `haodu.site` 首次备案。主办者信息、证件及人脸核验由用户本人提供和确认；备案期间的公开访问按当地管局要求执行。
4. 在 DNSPod 建立 `haodu.site` 解析区，核对并迁入现有 DNS 记录（包括仍需保留的证书验证 TXT）；准备齐全后再到阿里云注册商更换 NS。阿里云只继续承担域名注册、续费。
5. 备案通过后配置腾讯云加速与 HTTPS。需在控制台核实适用套餐与证书续期方式，再选择 CDN 或 EdgeOne；不预先订阅额外收费套餐。加速回源使用 SCF 函数 URL，保留路径并核实回源 Host、证书校验和源站鉴权。
6. 按生成的目标值配置网站解析。验证 HTTPS 首页、`/privacy/`、重定向、404、所有静态资源、安装入口与阅读辅助开关。核实证书自动续期或建立续期流程。
7. 完成备案后按要求展示真实备案号及链接。网站访问和部署验证均通过后，才清理不再使用的阿里云资源与部署用户。

SCF 新部署使用[函数 URL](https://cloud.tencent.com/document/product/583/96099)，不依赖已下线的旧 API 网关触发器。当前测试 URL 开启公网访问、开放鉴权，未开启内网访问、CORS 或旧 API 网关参数兼容；只提供本项目公开静态内容。
