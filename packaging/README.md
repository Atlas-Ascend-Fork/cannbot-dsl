# Native 编译与打包

## 构建

### 环境准备

按[源码构建](../docs/zh/install/source-build.md)准备并激活 Python 环境。按实际安装路径加载 CANN，再安装生成 wheel 所需的打包工具。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -m pip install 'setuptools>=70.1' wheel
```

### 编译命令

在仓库根目录运行下面的命令，编译全部已登记的 Native 算子。

```bash
python3 packaging/native/build.py
```

`--group` 与 `--operator` 互斥。可重复传入 `--operator=算子名` 选择多个算子。

| 参数 | 用途 | 示例或默认值 |
| --- | --- | --- |
| `--group` | 构建指定[网络算子组](net/README.md) | `--group=k3` |
| `--operator`、`--operators` | 构建一个或多个算子 | `--operator=mqsmla` |
| `--list` | 列出已登记的算子和组 | `--list` |
| `--soc` | 指定目标 SoC | `--soc=ascend950`（默认） |
| `-j`、`--jobs` | 设置算子编译并发上限 | 默认 `32` |
| `--no-partial` | 要求所有选中算子成功才发布 wheel | 默认允许发布部分 wheel |
| `-v`、`--verbose` | 显示 CANNBotDSL 编译日志和 kernel 进度 | `--verbose` |
| `--output-dir` | 指定 wheel 发布目录 | 默认 `packaging/native/output/` |
| `--work-dir` | 指定缓存和临时文件目录 | 默认 `packaging/native/.build/` |
| `-h`、`--help` | 查看帮助 | `--help` |

从支持 jobserver 的 GNU Make 规则调用时，构建器会复用共享的并发额度。
各算子在目标 SoC 上的可用性以实际编译验证为准。

### 获取编译日志

需要记录详细输出或排查编译报错时，可在 Bash 中用下面的命令构建 K3 组并保存日志。

```bash
set -o pipefail
python3 packaging/native/build.py --group=k3 --verbose 2>&1 | tee k3-build.log
```

## 产物

### 默认目录布局

```text
packaging/native/
├── output/                    # 已发布的 wheel
└── .build/
    ├── cache/                 # 可跨构建复用的编译缓存
    └── staging/build-*/       # 本次构建的暂存目录
        ├── aicore-*/artifacts/ # 各独立编译任务的 Native 产物
        ├── wheel-project/     # 汇总源码和 Native 资源后的 wheel 工程
        └── wheel-output/      # 待发布的 wheel
```

暂存目录在构建退出时清理。成功发布会替换输出目录中的同名 wheel。
安装后的 `ops/` 包含算子 Python 源码、`_native/` 下的 AICore 二进制与清单，以及按需生成的 `_aicpu/` 资源。

### wheel 命名与安装

| 构建结果 | wheel 文件名示例（CPython 3.12，Linux x86_64） |
| --- | --- |
| 全量或指定算子 | `cannbot_dsl_ops-0.3.0-cp312-cp312-linux_x86_64.whl` |
| K3 组 | `cannbot_dsl_ops_k3-0.3.0-cp312-cp312-linux_x86_64.whl` |
| K3 组部分成功 | `cannbot_dsl_ops_k3_partial-0.3.0-cp312-cp312-linux_x86_64.whl` |

多算子构建默认允许部分成功。部分 wheel 只包含成功算子，发布后打印失败清单并返回 0。全量或指定算子的部分包以 `cannbot_dsl_ops_partial-` 开头。全部成功时使用常规名称，全部失败时返回非零。传入 `--no-partial` 时，任何算子失败都不会发布 wheel。

在仓库根目录安装默认输出目录中的 wheel。以下以 Python 3.12、Linux x86_64 的 K3 完整包为例。其他环境或自定义 `--output-dir` 时，使用本次构建输出的 `Wheel:` 路径。

```bash
python3 -m pip install --force-reinstall --no-deps packaging/native/output/cannbot_dsl_ops_k3-0.3.0-cp312-cp312-linux_x86_64.whl
```

各 Native 分发都提供 `ops`。切换完整包、组包、部分包或从旧版迁移时，先卸载已安装的 Native 分发。

## 源码与导出

`native/operators.json` 登记导出名及相对于 `samples/` 的源码路径，[网络清单](net/network.json)定义组成员。构建器直接读取这两份清单。部分 DS41 算子可通过 `CANNBOTDSL_DS41_PROFILES` 指定导出规格 JSON，配置示例和二进制命中验证见 [AOT 与 Native 算子包](../docs/zh/guide/aot-packaging.md)。
