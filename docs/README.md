
# CANNBot-DSL 文档目录

本目录保存 CANNBot-DSL 的中文、英文文档和文档站构建资源：

```text
docs/
├── zh/                 # 中文 API、开发指南和源码构建文档
│   ├── index.md        # 中文站点首页
│   ├── api/            # API 参考及 API 公共图片
│   │   ├── README.md   # API 手册目录
│   │   └── figures/
│   ├── guide/          # 开发指南与编程模型
│   │   └── README.md   # 开发指南目录
│   └── install/        # 源码构建文档
├── en/                 # 英文文档
├── project/            # 文档站构建工程
└── README.md            # 本目录说明和文档构建方法
```

样例源码、样例 README 和样例图片归属于仓库根目录的 `samples/`；仓库级贡献指南位于仓库根目录。

## 文档本地构建

文档站使用 VitePress 构建，构建工程位于 `docs/project/`。本地构建前请安装 Node.js 18 或更高版本，并进入构建工程目录：

```bash
cd docs/project
npm install
```

启动开发服务器，修改 Markdown 后页面会自动刷新：

```bash
npm run dev
```

默认访问 `http://localhost:5173/`。如果只需要检查正式构建结果，请执行：

```bash
npm run build
npm run preview
```

`npm run build` 会生成 `docs/project/.vitepress/dist/` 静态站点；`npm run preview` 用于预览该构建产物。提交文档前请检查构建是否成功、页面链接是否可访问，并执行：

```bash
git diff --check
```

文档正文位于 `docs/zh/` 和 `docs/en/`，VitePress 配置、主题和构建脚本位于 `docs/project/`。API 和开发指南的 `README.md` 只用于在代码仓库中提供手册目录，不作为网站侧边栏页面展示。
