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

## Native 打包工程

`packaging/native/` 通过 `cannbotdsl.aot.build` 编译登记的算子，并生成 wheel。
源码位于 `samples/`，路径与辅助模块在 `catalog/operators.json` 中维护。
例如，`operators` 映射中的一个算子定义为：

```json
{
  "mixed_quant_sparse_flash_mla": {
    "source": "samples/attention/mixed_quant_sparse_flash_mla",
    "native": {
      "export": "mqsmla",
      "sources": [
        "@mixed_quant_sparse_flash_mla/source/mixed_quant_sparse_flash_mla.py",
        "@mixed_quant_sparse_flash_mla/source/mixed_quant_sparse_flash_mla_metadata.py"
      ]
    }
  }
}
```

`native.export` 与源码中的 `@export(...)` 名称一致；`native.sources` 列出主模块
及其辅助模块。`net/<网络>/network.json` 通过算子 ID 选择分组成员。
在仓库根目录生成构建配置：

```bash
python3 scripts/ci/sync_catalog.py --write
python3 scripts/ci/sync_catalog.py --check
```

生成的 `packaging/native/operator_groups.toml` 使用相对 `samples/` 的源码路径。
同一次构建中的模块使用文件名作为包内模块名，不同来源的同名模块会在编译前报错。

### 构建与安装

在仓库根目录执行：

```bash
export PYTHON=/path/to/python3.12
export CANN_ENV=/path/to/Ascend/ascend-toolkit/set_env.sh
export CANNBOTDSL_WHEEL=/path/to/cannbot_dsl-0.8.0-cp312-cp312-manylinux_2_28_x86_64.whl

bash packaging/native/run-build.sh --list
bash packaging/native/run-build.sh --group ds41
bash packaging/native/run-build.sh --operator mqsmla
python3 -m pip install packaging/native/output/groups/ds41/wheels/*.whl
```

默认目标是 `dav-3510`，可通过 `--target dav-2201` 指定其他受支持目标。
`--list` 和 `--help` 可直接运行。环境配置及产物目录见
[源码构建](../install/source-build.md)。

wheel 分发名为 `cannbot-arena-net`，Python 包名为 `ops`，依赖 `cannbot-dsl>=0.8.0`。
导入 `ops` 时通过 `cannbotdsl.aot.register_directory` 注册 Native 资源。
AICore 使用 AOT 导出；配套 AICPU metadata 资源安装在 `ops/_aicpu/`。

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

DS41 的静态配置由各算子的 `export_*` 函数定义。QLI、QSLI 等算子枚举布局、掩码、压缩比和分页大小，动态维度通过 TensorSpec 表达；`attn_epilogue` 的 token 数和核配额按静态配置枚举。具体组合以对应导出函数为准。

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
