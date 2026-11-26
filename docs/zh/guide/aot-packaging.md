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
源码位于 `samples/`，Native 导出及其辅助模块在 `packaging/native/operators.json` 中维护。
例如，清单的 `operators` 映射中的一个定义为：

```json
{
  "mixed_quant_sparse_flash_mla": {
    "export": "mqsmla",
    "sources": [
      "attention/mixed_quant_sparse_flash_mla/mixed_quant_sparse_flash_mla.py",
      "attention/mixed_quant_sparse_flash_mla/mixed_quant_sparse_flash_mla_metadata.py"
    ]
  }
}
```

`export` 与源码中的 `@export(...)` 名称一致；`sources` 列出相对于 `samples/` 的主模块
及辅助模块路径。`packaging/net/network.json` 通过算子 ID 选择分组成员。
`build.py` 自行读取和校验这两份 JSON，不需要运行 CI 配置生成器。
同一次构建中的模块使用文件名作为包内模块名，不同来源的同名模块会在编译前报错。

### 构建与安装

按[源码构建](../install/source-build.md)准备环境后，在仓库根目录执行：

```bash
# 先激活已安装 DSL、PyTorch、torch_npu 的 Python 环境。
# CANN 非默认安装时，将下列路径改为实际 set_env.sh。
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 -m pip install 'setuptools>=70.1' wheel
python3 packaging/native/build.py --list
python3 packaging/native/build.py --group ds41 --no-partial
python3 packaging/native/build.py --operator mqsmla
python3 -m pip install --force-reinstall --no-deps packaging/native/output/cannbot_dsl_ops_ds41-0.3.0-*.whl
```

构建脚本不会自动探测或 `source` CANN，也不会安装 DSL；每个新终端都需先加载上述环境。
`python3` 和 `python3 -m pip` 应使用同一已激活的 Python 环境。

Native 打包默认使用 `ascend950`，也可显式传入 `--soc=ascend950`。
`--list` 和 `--help` 可直接运行。完整构建参数和产物目录见仓库中的 `packaging/README.md`。

单算子或全量 wheel 分发名为 `cannbot-dsl-ops`；DS41、K3 分组包分别为
`cannbot-dsl-ops-ds41`、`cannbot-dsl-ops-k3`，版本均为 `0.3.0`，依赖 `cannbot-dsl>=0.8.0`。
它们都提供 Python 包 `ops`，切换分发前需先卸载另一种已安装的分发。
导入 `ops` 时通过 `cannbotdsl.aot.register_directory` 注册 Native 资源。
AICore 使用 AOT 导出；配套 AICPU metadata 资源安装在 `ops/_aicpu/`。

### 验证二进制覆盖

安装后可以用环境变量检查是否命中了预编译二进制。**这两个是 DSL 运行时读的**（`cannbotdsl/aot/runtime.py`），与算子导出阶段使用的配置变量作用不同：

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
| `CANNBOTDSL_DS41_PROFILES` | 支持该变量的 DS41 算子导出函数 |

`CANNBOTDSL_DS41_PROFILES` 在 AOT 导出阶段读取，不改变已安装 Native 二进制的配置集合；
运行时未命中是否回退 JIT，由 `CANNBOTDSL_NATIVE_BINARY_MODE` 决定。完整的 DSL 运行时变量清单见[编译选项与产物观察](./compiler-options.md)。
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

`CANNBOTDSL_DS41_PROFILES` 是可选的构建配置文件路径。支持它的算子在执行
`export_*` 时读取 JSON，用指定的 token 数、布局、核配额等配置替换默认导出集合。
它不选择 DS41 网络成员，也不加载 CANN 环境；未设置该变量时使用源码中的默认规格。

例如，只为 `engram_gate` 构建 `tokens=72`、56 核、不带 mask 的配置。
在仓库根目录且已完成上面的环境准备后执行：

```bash
mkdir -p packaging/native/.build
cat > packaging/native/.build/engram-profiles.json <<'JSON'
{"engram_gate": [{"max_blocks": 56, "tokens": [72], "has_mask": [false]}]}
JSON
CANNBOTDSL_DS41_PROFILES="$PWD/packaging/native/.build/engram-profiles.json" \
  python3 packaging/native/build.py --operator engram_gate
```

上面的环境变量只对该次构建命令生效。各算子的 JSON 字段由其导出函数定义，
不同算子不能直接套用同一组字段。部分导出函数要求指定的 JSON 包含自身的键，
例如 `attn_epilogue` 缺键时会报错；按组构建时需为相关算子准备完整配置，
不能统一假定“未列出的算子都会保留默认配置”。仅需默认配置时不设置此变量。
未导出的规格在运行时按 Native 模式回退 JIT 或报错。

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
