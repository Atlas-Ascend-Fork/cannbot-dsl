# Native 编译与打包

通过 `cannbotdsl.aot.build` 编译算子，并生成可安装的 wheel。
[算子清单](../../catalog/README.md)定义源码、辅助模块和 Native 导出，
[网络清单](../../net/README.md)定义 DS41、K3 等构建分组。

## 构建准备

先按仓库首页完成 CANN 和 CANNBot-DSL 0.8.0 的安装。`run-build.sh` 还需要本地 DSL wheel 文件，用于创建隔离的构建环境；仅通过 pip 安装 DSL 包不会提供该文件。可使用与构建相同的 Python 下载匹配当前 Python 版本和 CPU 架构的 wheel：

```bash
python3 -m pip download --no-deps --only-binary=:all: cannbot-dsl==0.8.0 -d /path/to/dsl-wheels
```

## 构建

在仓库根目录配置 Python、CANN 环境脚本和本地 DSL wheel。以下以 Python 3.12、x86_64 为例，`CANNBOTDSL_WHEEL` 请填写实际下载文件的绝对路径：

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

`--list` 和 `--help` 可直接运行。默认目标为 `dav-3510`。

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

例如，执行 `bash packaging/native/run-build.sh --group k3` 成功后，在仓库根目录安装 K3 分组的构建产物：

```bash
python3 -m pip install --force-reinstall --no-deps packaging/native/output/groups/k3/wheels/*.whl
```

构建 `ds41` 分组时，安装路径改为 `packaging/native/output/groups/ds41/wheels/*.whl`；构建全部已登记算子时，改为 `packaging/native/output/all/wheels/*.whl`。单算子产物位于 `packaging/native/output/operators/<算子名>/wheels/`，实际路径以构建日志为准。

wheel 分发名为 `cannbot-arena-net`，Python 包名为 `ops`，依赖
`cannbot-dsl>=0.8.0,<0.9`。导入 `ops` 时注册 Native 资源；AICPU 资源位于 `ops/_aicpu/`。

安装后验证包导入与 Native 资源注册：

```bash
python3 -c 'import ops; print(ops.__file__)'
```

不要与同样提供 `ops` 的旧分发 `cannbot-arena-net-ops` 混装。算子精度及预编译二进制覆盖需通过对应样例验证。

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

## attn_epilogue 静态配置

`attn_epilogue` 的实现与导出集中在 `attn_epilogue.py`，无需 common 模块。
默认导出 28/32 核、ND/NZ 两种权重布局：NG=8 的
`T=1,72,128,256,512,1024,2048,4096,8192`，以及 NG=2/1 的 `T=1,16`，共 52 个配置。
两级混合布局可通过 `first_nz`、`second_nz` 独立指定。

例如，profiles.json 中的配置可写为：

```json
{"attn_epilogue":[{"block_num":32,"ng":2,"weight_format":"nz","tokens":[1,7,16]}]}
```

当前安装包依赖 DSL 0.8.0 系列。实现使用 RingBuffer、显式 layout 和 copy_options，
已有 Native 资源应使用当前源码及相同版本的 DSL 重新构建。
