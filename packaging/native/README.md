# Native 编译与打包

通过 `cannbotdsl.aot.build` 编译算子，并生成可安装的 wheel。
[算子清单](../../catalog/README.md)定义源码、辅助模块和 Native 导出，
[网络清单](../../net/README.md)定义 DS41、K3 等构建分组。

## 构建

需要匹配的 CANN 环境、Python 及 CANNBot-DSL 0.8.0 wheel。在仓库根目录配置：

```bash
export PYTHON=/path/to/python3.12
export CANN_ENV=/path/to/Ascend/ascend-toolkit/set_env.sh
export CANNBOTDSL_WHEEL=/path/to/cannbot_dsl-0.8.0-cp312-cp312-manylinux_2_28_x86_64.whl

# 列出算子和网络分组。
bash packaging/native/run-build.sh --list

# 构建网络分组。
bash packaging/native/run-build.sh --group ds41
bash packaging/native/run-build.sh --group k3

# 构建指定算子或全部登记算子。
bash packaging/native/run-build.sh --operator mqsmla
bash packaging/native/run-build.sh
```

`--list` 和 `--help` 可直接运行。默认目标为 `dav-3510`，使用
`--target dav-2201` 指定其他受支持目标。

## 产物与安装

构建按全量、网络分组和单个算子分别保存缓存与产物：

```text
packaging/native/
├── .build/
│   ├── frameworks/<wheel-sha256>/
│   ├── groups/ds41/
│   ├── operators/mqsmla/
│   └── all/
└── output/
    ├── groups/ds41/{native,aicpu,wheels}/
    ├── operators/mqsmla/{native,aicpu,wheels}/
    └── all/{native,aicpu,wheels}/
```

`native/` 保存 AICore 二进制，`aicpu/` 保存配套 AICPU 资源，`wheels/` 保存算子 wheel。
`--work-root` 指定编译中间目录，`--output-root` 指定发布目录。

例如，安装 DS41 分组的构建产物：

```bash
python3 -m pip install packaging/native/output/groups/ds41/wheels/*.whl
```

wheel 分发名为 `cannbot-arena-net`，Python 包名为 `ops`，依赖
`cannbot-dsl>=0.8.0`。导入 `ops` 时注册 Native 资源；AICPU 资源位于 `ops/_aicpu/`。

## 源码与配置

`operator_groups.toml` 由算子和网络清单生成，源码路径相对 `samples/`。
每个导出需登记主模块及其辅助模块。选中的模块使用文件名作为包内模块名，
同一次构建中不同来源的同名模块会在编译前报错。

修改清单后，在仓库根目录执行：

```bash
python3 scripts/ci/sync_catalog.py --write
python3 scripts/ci/sync_catalog.py --check
```

默认静态配置由各算子的 `export_*` 函数定义。
`CANNBOTDSL_DS41_PROFILES=/path/to/profiles.json` 可指定 `qli`、`qsli` 和
`attn_epilogue` 的导出配置，字段见对应导出函数。全量构建包含全部登记算子
及其默认配置；二进制覆盖范围由这些静态参数决定。

运行时设置 `CANNBOTDSL_NATIVE_BINARY_MODE=require` 可要求命中 Native 二进制；
`CANNBOTDSL_NATIVE_BINARY_REPORT=/path/to/report.json` 可记录命中信息。
