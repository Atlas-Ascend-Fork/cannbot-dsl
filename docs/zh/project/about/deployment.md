# 发布与部署

文档站的目标域名是 `cannbot-dsl.gitcode.com`，正式源码仓库是 `https://gitcode.com/cann/cannbot-dsl`。域名和托管配置完成前，可以在开发机或服务器上预览构建结果。

## 生成静态文件

```bash
cd docs
npm install
npm run build
```

构建产物位于 `docs/.vitepress/dist/`。正式部署系统应将该目录作为站点根目录。

## 生产预览

```bash
cd docs
npm run preview
```

预览服务默认只监听服务器的 `127.0.0.1:4174`。远程服务器无需图形界面，可以通过 SSH 本地端口转发后在本机浏览器检查页面。

## 子路径部署

站点默认部署在域名根路径。如果临时部署在子路径，可在构建时设置：

```bash
DOCS_BASE=/preview/ npm run build
```


## 构建版本

文档源码只维护一份。开发中的站点从 `main` 分支构建，商用站点从冻结的 release 分支或 Git tag 构建。

```bash
# 开发预览
DOCS_VERSION=next npm run build

# 商用版本示例
DOCS_BASE=/v0.1.0/ DOCS_VERSION=v0.1.0 npm run build
```

两种构建使用相同的 API 页面路径；版本差异由构建来源和站点基础路径体现。旧版本发布后不得回写，后续修复应创建新的 patch 版本。
