# AOT 与 Native 算子包

JIT 很适合开发，但部署时首次调用的编译开销往往不可接受。CANNBot-DSL 通过 `cannbotdsl.aot` 提供提前编译（AOT）和 Native 算子包发布能力：把一组算子配置提前编译成二进制，打包成 wheel 分发，运行时直接加载。

## `@export`：声明发布入口

`@export("name")` 由 `cannbotdsl.aot` 提供，用于声明算子发布入口。被装饰的函数**不接收参数**，函数体中列出需要包含在 Native 算子包中的编译配置。

```python
import cannbotdsl as cbd
from cannbotdsl.aot import export


@export("add")
def export_add():
    for dtype in (cbd.dtypes.float16, cbd.dtypes.float32):
        spec = cbd.TensorSpec((1024,), dtype)
        cbd.compile(Add().run, spec, spec, spec)
```

要点：

- 装饰器参数是算子的**发布名称**，必须匹配 `[A-Za-z0-9][A-Za-z0-9_.-]*`——以字母或数字开头，可包含字母、数字、下划线、点和连字符；`.` 和 `..` 本身不合法。
- 发布入口必须是**无参数的普通 Python 函数**，不能使用 `async def`。带参数会在装饰时直接抛 `TypeError`。
- **构建时才会执行发布入口中列出的配置**；仅导入包含该函数的模块不会触发编译。

`cannbotdsl.aot` 一共导出这些：`compile`、`compile_cache`、`TensorSpec`、`Dim`、`TensorListSpec`、`StructSpec`、`Constexpr`、`export`、`build`、`register_directory`。

函数体就是普通 Python，所以你可以用循环枚举配置组合——dtype × layout × 页大小 × mask 模式等。配置越全，覆盖越广，但构建时间和包体积也越大。

## 打包工程：`net/native_package`

仓库提供了一个现成的打包工程，通过 `cannbotdsl.aot.build` 完成 Native 编译与 wheel 打包。**算子源码直接取自 `samples/` 目录，不需要复制到别处。**

### 登记算子

在 `operator_groups.toml` 中登记算子名与源文件名：

```toml
[operators]
mqsmla = ["mixed_quant_sparse_flash_mla.py", "mixed_quant_sparse_flash_mla_metadata.py"]
qli = ["quant_lightning_indexer.py", "quant_lightning_indexer_metadata.py"]
qsli = ["quant_sparse_lightning_indexer.py", "quant_sparse_lightning_indexer_metadata.py"]

[groups]
ds41 = ["mqsmla", "qli", "qsli"]
```

规则：

- **算子名必须与源码中的 `@export(...)` 名称一致。**
- 源文件按**文件名**在 `samples/` 下递归查找，所以文件放在任意子目录都可以。
- **算子用到的辅助模块必须一并列在它的文件数组里**，例如 `matmul_streamk = ["matmul_streamk.py", "_sample_raw_vf.py", "device_properties.py"]`。漏列会导致 wheel 内缺模块，装载后 import 失败。
- 采样文件的基名必须唯一：扁平化后文件名即模块名，重名会被构建脚本直接拒绝。

### 构建

在 `net/native_package` 下执行：

```bash
./run-build.sh --list                 # 查看可用算子和组
./run-build.sh                        # 全部已登记算子
./run-build.sh --group ds41           # 一个组
./run-build.sh --operator mqsmla      # 单个算子
./run-build.sh --operator mixed_quant_sparse_flash_mla.py   # 也可以传源文件名
```

默认目标是 `dav-3510`，可通过 `--target` 切换。

构建前需要准备：

```bash
export PYTHON=/path/to/python3.12
export CANN_ENV=/path/to/Ascend/cann/set_env.sh
export CANNBOTDSL_WHEEL=/path/to/cannbotdsl-0.7.0-cp312-cp312-manylinux_2_28_x86_64.whl
./run-build.sh --group ds41
```

`run-build.sh` 会优先选择 `net/` 下修改时间最新的 `cannbotdsl-*.whl`；`CANNBOTDSL_WHEEL` 可以显式指定版本。需要包含 `cannbotdsl.aot.export/build/register_directory` 的 0.7.0 wheel。`--list` / `--help` 不需要 CANN 环境或 wheel。

### 产物布局

构建按选择类型分别保存，彼此不会覆盖：

```text
native_package/
├── .build/
│   ├── frameworks/<dsl-wheel-sha256>/
│   ├── groups/ds41/
│   ├── operators/mqsmla/
│   └── all/
└── output/
    ├── groups/ds41/{native,aicpu,wheels}/
    ├── operators/mqsmla/{native,aicpu,wheels}/
    └── all/{native,aicpu,wheels}/
```

`.build/` 是编译缓存和临时目录，`output/` 只包含当前成功发布的产物。**构建失败时不会替换已有的成功产物。**

生成的算子 wheel 依赖 `cannbotdsl>=0.7.0,<0.8`，导入时通过 `cannbotdsl.aot.register_directory` 注册 Native 资源。

### AICore 与 AICPU 两条路径

- AICore 部分通过 `cannbotdsl.aot` 导出。
- AICPU metadata 在各自独立的编译目录生成，随 wheel 安装到 `ops/_aicpu/`，运行时优先加载这些预编译文件。
- 直接运行源码时仍保留原有的 metadata 编译路径。

### 验证二进制覆盖

安装后可以用环境变量检查是否命中了预编译二进制。**这两个是 DSL 运行时读的**（`cannbotdsl/aot/runtime.py`），和上面那几个构建脚本变量不是一回事：

```bash
# 未命中时立即报错，不回退 JIT
export CANNBOTDSL_NATIVE_BINARY_MODE=require

# 保存命中记录
export CANNBOTDSL_NATIVE_BINARY_REPORT=/path/to/native-report.json
```

上线前跑一遍 `require` 模式，是确认配置覆盖完整的最快方式。

::: info 两类环境变量别混
名字都带 `CANNBOTDSL_` 前缀，但读的人不是一个：

| 变量 | 谁读 |
| --- | --- |
| `CANNBOTDSL_NATIVE_BINARY_MODE` / `NATIVE_BINARY_REPORT` | **DSL 运行时** |
| `CANNBOTDSL_WHEEL` / `CANNBOTDSL_ROOT` / `CANNBOTDSL_DS41_PROFILES` / `CANN_ENV` / `PYTHON` | **`run-build.sh` 打包脚本**，`python/cannbotdsl` 里没有读取点 |

所以在运行环境里设第二组是没用的。完整的 DSL 运行时变量清单见[编译选项与产物观察](./compiler-options.md)。
:::

::: warning
「全量包」指全部已登记算子及其默认导出配置，**不表示任意静态参数组合都有二进制**。没被 `@export` 枚举到的配置在运行时会回退到 JIT（除非设置了 `require` 模式）。
:::

## 配置枚举的取舍

AOT 的核心工作量不在构建流程，而在**决定导出哪些配置**。

| 维度 | 处理方式 |
| --- | --- |
| 变化范围大且连续的维度（batch、序列长度、token 数） | 声明成 `Dim`，一份二进制覆盖整个范围 |
| 离散且数量有限的配置（dtype、layout、页大小、mask 模式） | 在 `@export` 函数里枚举组合 |
| 影响代码结构的内层 tile 参数 | 固定成若干档，按档导出 |

仓库里 DS41 的实际做法可以作为参考：默认导出 MQSMLA 的 36 个配置、QLI 和 QSLI 各 18 个配置，覆盖 PA_BBND/TND、`mask_mode=0/3`、因果压缩比 1/2、PA 页大小 64/128；B、T1 和 K 序列长度作为动态轴。`attn_epilogue` 的 `T` 参与编译（tiling 与形状随 `T` 变化），不能声明成动态轴，因此按 28/32 核配额枚举代表性 token 数（`T=1,72,128,256,512,1024,2048,4096,8192`，共 18 个配置）。

配置集合也可以外置成 JSON，通过环境变量指定：

```bash
export CANNBOTDSL_DS41_PROFILES=/path/to/profiles.json
```

JSON 里只列出的配置会被替换，未列出的算子保留默认配置。

## JIT 还是 AOT

| 场景 | 建议 |
| --- | --- |
| 算子开发、调参 | JIT，直接调用 `@host` 函数 |
| 精度测试、CI | JIT + `clear_compile_cache()` |
| 性能基准 | JIT，但先 `cbd.compile` 预热再计时 |
| 推理服务部署 | AOT，`@export` 枚举线上会用到的配置 |
| 形状高度不可预测的场景 | AOT 覆盖主路径 + 保留 JIT 回退（不设 `require`） |

## 下一步

[当前限制、迁移与常见问题](./limitations.md)：哪些写法现在不支持，报错了先查什么。
