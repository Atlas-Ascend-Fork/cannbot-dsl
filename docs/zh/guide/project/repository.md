# 仓库结构

项目源码按样例、测试和工程脚本组织：

```text
cannbot-dsl/
├── docs/              # 文档站源码
├── samples/           # 算子样例及各自的 README
├── test/              # 与样例对应的测试代码
├── scripts/           # CI 与合规检查脚本
├── figures/           # 文档和 README 使用的图片
├── build.sh           # 项目构建入口
└── requirements.txt   # Python 依赖
```

## `samples/`

每个子目录对应一个算子或一组紧密相关的实现。阅读样例时，应从该目录的 README 开始。

## `test/`

测试目录与 `samples/` 基本对应，用于验证实现的功能和精度。修改样例后，应运行对应的测试文件。

## `docs/`

文档站采用 VitePress 构建。源文件使用 Markdown，站点配置与主题位于 `.vitepress/`。
