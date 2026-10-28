# 仓库结构

项目源码按样例、测试和工程脚本组织：

```text
cannbot-dsl/
├── docs/
│   ├── zh/              # 中文 API、开发指南和源码构建文档
│   │   └── api/figures/ # API 公共图片
│   ├── en/              # 英文文档
│   └── project/         # 文档站构建工程
├── samples/             # 算子样例、样例导航和 README
├── test/                # 与样例对应的测试代码
├── net/native_package/  # Native 算子包构建工程
├── figures/             # 仓库根 README 使用的图片
├── build.sh             # CI 环境校验和测试调度入口
└── requirements.txt     # Python 依赖
```

## `samples/`

每个子目录对应一个算子或一组紧密相关的实现。阅读样例时，应从该目录的 README 开始；样例性能图片统一放在根目录 `figures/samples/<算子>/` 下。样例总导航位于 [`samples/`](../../../samples/)。

## `test/`

测试目录与 `samples/` 基本对应，用于验证实现的功能和精度。修改样例后，应运行对应的测试文件。

## `docs/`

文档正文位于 `zh/` 和 `en/`，文档站构建配置、主题、静态资源和 Node.js 依赖位于 `project/`。本地构建请进入 `docs/project/` 执行 npm 脚本。
