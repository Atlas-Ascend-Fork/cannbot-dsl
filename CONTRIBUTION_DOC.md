
# 文档贡献指南

本文件说明如何修改和校验 CANNBot-DSL 文档。

## 修改范围

- API 参考统一放在 `docs/zh/api/`。
- 开发指南统一放在 `docs/zh/guide/`。
- 安装和构建说明放在 `docs/zh/install/`。
- 项目资料、样例导航和协作说明放在 `docs/project/`。

## 本地校验

```bash
cd docs/project
npm install
npm run build
```

提交前请确认新增页面已经加入对应的 `index.md` 目录索引，并检查中英文入口是否存在。
