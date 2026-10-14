# CANNBot-DSL 文档站

这是 `cannbot-dsl` 的 VitePress 文档站源码。文档内容按语言和用途组织：

```text
docs/
├── zh/                 # 中文文档
│   ├── api/            # API 参考，索引文件只保留一级标题
│   ├── guide/          # 开发指南与编程模型
│   ├── install/        # 安装、构建与预览
│   ├── project/        # 项目资料、样例和贡献指南
│   ├── CONTRIBUTION_DOC.md
│   └── README.md
├── en/                 # 英文文档（API 与 Guide 待人工翻译）
├── .vitepress/         # 站点配置与主题
└── public/             # 静态资源
```

## 本地开发

```bash
cd docs
npm install
npm run dev
```

## 构建与预览

```bash
npm run build
npm run preview
```

构建产物位于 `.vitepress/dist/`。开发预览默认使用 `DOCS_VERSION=next`；构建商用版本时同时设置 `DOCS_BASE=/vX.Y.Z/ DOCS_VERSION=vX.Y.Z`。
