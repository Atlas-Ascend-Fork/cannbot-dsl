# JIT 参数与编译缓存

CANNBot-DSL 是 JIT 编译的：第一次用某组参数调用 `@host` 函数时才编译，之后复用缓存。这一节讲清楚**什么决定了「同一组参数」**，以及怎么在「专门化带来的性能」和「编译次数爆炸」之间取舍。

## 两种调用路径

### 路径一：直接调用，即时编译

```python
op = RmsNorm(dtype=dtypes.bfloat16)
op.run(x2d, gamma2d, y2d, rstd2d, float(epsilon))
```

传入真实的 NPU Tensor（例如 torch_npu 创建的 `torch.Tensor`）。框架根据实际参数编译、缓存并执行。

开发阶段用这条路径最方便。

### 路径二：提前编译，得到可复用的程序

```python
import cannbotdsl as cbd

spec = cbd.TensorSpec((1024,), cbd.dtypes.float16)
program = cbd.compile(Add().run, spec, spec, spec)
program(x, y, out)          # 之后用真实 Tensor 调用
```

`cbd.compile(host_fn, *args)` 的第一个参数是 `@host` 函数，后面按位置传参数规格。`@jit` 函数不能作为编译入口。

这条路径的价值在于：**你可以精确声明哪些维度是动态的**，从而用一份编译产物覆盖一批 shape。

## 参数规格类型

| 类型 | 描述什么 |
| --- | --- |
| [`TensorSpec`](../api/compilation/host/specs/tensor_spec.md) | 单个 Tensor 参数的 shape、dtype、stride、存储格式 |
| [`TensorListSpec`](../api/compilation/host/specs/tensor_list_spec.md) | Tensor 列表参数的长度，以及每个 Tensor 的类型和形状规则 |
| [`Dim`](../api/compilation/host/specs/dim.md) | 动态 shape / 动态 stride / 动态列表长度及其取值边界 |
| `StructSpec` | 结构化参数（配合 `@datastruct`） |

规格对象只记录输入应满足的条件，**不保存数据、不触发分配、不触发搬运或执行**。编译完成后执行程序时仍然要传真实的 NPU Tensor。

### `TensorSpec`

```python
TensorSpec(
    shape: Iterable[int | Dim],
    dtype: DType,
    *,
    stride: Iterable[int | Dim] | None = None,
    storage_format: str = "nd",   # "nd" 或 "nz"
) -> TensorSpec
```

`stride` 省略时按 shape 推导行主序紧凑 stride，单位是元素。`storage_format="nz"` 时 rank 必须为 2 或 3，且显式 stride 必须与推导出的紧凑 stride 完全一致。

### `Dim`：给动态维度命名

```python
Dim(name: str, min: int = 1, max: int | None = None, multiple_of: int = 1) -> Dim
```

```python
M = cbd.Dim("M", min=8, max=1024, multiple_of=8)
spec = cbd.TensorSpec((M,), cbd.dtypes.float32)
program = cbd.compile(copy, spec, spec)

program(x, y)    # x、y 的长度可以是 8、16、…、1024 中任意值，不触发重新编译
```

要点：

- **同一次 `compile(...)` 调用中，`name` 相同的 `Dim` 表示同一个动态长度**，并且必须声明完全相同的 `min` / `max` / `multiple_of`。
- `min` / `max` / `multiple_of` 是运行时实际长度必须满足的约束。编译器按这份声明生成一份代码；它们不保证「知道 `multiple_of=8` 就自动省略尾块处理」。尾块是否特化，仍取决于你在 Kernel 里怎么写。
- `Dim` 可以通过 `+`、`-`、`*`、`//` 与 Python 整数或其他 `Dim` 组成维度表达式；`//` 的除数必须是正 `int`。
- 维度表达式不是新的 `Dim`，没有独立的 `name` / `min` / `max`。使用 `M * 2` 这类表达式时，同一次 `compile(...)` 的 shape、stride 或列表长度里**必须至少有一个位置直接使用 `M`**——框架从该位置取得 `M` 的实际值，不会反向求解。

### 用 `Dim` 描述动态 stride

Cache / paged KV 这类场景里，Tensor 的 stride 本身是运行时契约。`flash_kda` 样例的做法：

```python
def _sequence_spec(dtype, physical_batches, length, heads, width, stride_prefix):
    stride_0 = cannbotdsl.Dim(f"{stride_prefix}_S0")
    stride_1 = cannbotdsl.Dim(f"{stride_prefix}_S1")
    stride_2 = cannbotdsl.Dim(f"{stride_prefix}_S2")
    return cannbotdsl.TensorSpec(
        (physical_batches, heads, length, width),
        dtype,
        stride=(stride_0, stride_1, stride_2, 1),
    )
```

这样一来，**不同 stride 的 cache view 不需要整理、复制，也不会触发单独编译**。

### `@datastruct`：成组传递配置

```python
@cbd.datastruct
class TileConfig:
    rows: cbd.dtypes.int32
    columns: cbd.dtypes.int32


config = TileConfig(rows=64, columns=128)
```

适用于 tile 大小、循环次数或其他需要成组传递的配置。类体仅用于字段声明，不应定义默认值、继承关系或业务方法。提前编译时对应的规格类型是 `StructSpec`。

## 什么会触发重新编译

这是实际开发中最常踩的坑。**凡是参与设备代码生成的编译期值发生变化，就会产生一份新的编译产物。**

| 变化项 | 是否重新编译 | 说明 |
| --- | :---: | --- |
| Tensor dtype | 是 | 不同 dtype 的计算路径不同 |
| Tensor rank | 是 | — |
| Tensor 的静态维度值 | 是 | 除非在 `TensorSpec` 里声明成 `Dim` |
| Tensor 的 stride 模式 | 是 | 除非声明成 `Dim` |
| `storage_format`（nd / nz） | 是 | — |
| `@kernel` 类的构造参数 | 是 | `__init__` 里算出的一切都会固化进代码 |
| Host 侧 tiling 推导出的不同结果 | 是 | tile 大小变了，代码就变了 |
| `const_expr` 条件取值 | 是 | 不同分支生成不同代码 |
| Tensor 的实际数据内容 | 否 | — |
| Kernel 的标量形参取值 | 否 | 标量形参是运行期值 |
| 声明为 `Dim` 的维度在范围内变化 | 否 | 这正是 `Dim` 的用途 |

### 症状与对策

**症状：每换一个 batch size 就卡顿几秒。**
典型的编译爆炸。把变化的维度在 `TensorSpec` 里声明成 `Dim`，并用 `cbd.compile` 提前编译一次。

**症状：首次调用很慢，之后很快。**
正常的 JIT 行为。生产部署时用 [AOT 与 Native 算子包](./aot-packaging.md)消除这个开销。

**症状：希望某个维度动态，但 kernel 里用它做了 `range_constexpr` 的上界。**
这两者冲突。`range_constexpr` 要求编译期常量。把该循环改成 `cbd.range()`，或者接受这个维度参与专门化。

## 缓存与复用模式

### 模块级缓存一份编译产物

`flash_kda` 的做法：用模块级变量 + 锁，保证全进程只编译一次：

```python
_COMPILED_KERNEL = None
_COMPILED_KERNEL_LOCK = threading.Lock()


def _get_compiled_kernel():
    global _COMPILED_KERNEL
    with _COMPILED_KERNEL_LOCK:
        if _COMPILED_KERNEL is not None:
            return _COMPILED_KERNEL

        physical_batches = cannbotdsl.Dim("P")
        storage_length = cannbotdsl.Dim("L")
        batch = cannbotdsl.Dim("B")
        core_num = cannbotdsl.Dim("CORE_NUM", min=1, max=MAX_AIC_CORES)
        ...
        _COMPILED_KERNEL = cannbotdsl.compile(FlashKDA().run, qk_spec, v_spec, ...)
        return _COMPILED_KERNEL
```

多配置场景（不同 dtype、不同 layout）可以把缓存做成字典，key 用配置元组。`indexer_prologue_k` 就是这种「one compiled `ProviderCallable` per key」的写法。

### 缓存是两级的，而且能落盘

一个常见的误解是「JIT 缓存只在进程内，所以重启就得重编译一遍」。实际上缓存分两级，磁盘级可以跨进程复用（实现见 `cannbotdsl/core/compiler/cache.py`）。

| 级别 | 怎么开 | 行为 |
| --- | --- | --- |
| **L1 内存 LRU** | 默认开 | 进程内复用。容量由 `CANNBOTDSL_CACHE_MEM_MAX` 控制，默认 **128** 条；`0` 禁写，`none` 不限 |
| **L2 磁盘** | **设置 `CANNBOTDSL_CACHE_DIR`** | **跨进程复用**。布局 `$DIR/v2/<key 前两位>/<key>.so` 加一个 sidecar |

两个临时开关：`CANNBOTDSL_CACHE_MEM_DISABLE` / `CANNBOTDSL_CACHE_DISK_DISABLE`——已有条目保留，重新开启后仍可复用。`CANNBOTDSL_CACHE_FORMAT_TAG` 用来人工作废全部旧条目。

缓存键由这几节内容拼成：`verified-ir`、`call-contract`、`environment`、`compile-options`、`target`、`runtime`、`toolchain`、`format`、`user-format`。设计原则是**黑名单制**——新增的环境字段默认进 key，只有能证明「不影响产物」时才排除，因为 *stale miss 只是慢，stale hit 是正确性 bug*。

::: tip 冷启动有两条路，不只 AOT
**轻量方案**：部署镜像里设好 `CANNBOTDSL_CACHE_DIR`，构建阶段跑一遍预热脚本把 `.so` 烤进镜像。改造成本低，但要命中键才有效（键里含工具链与环境），工具链一升级就全失效。

**彻底方案**：[AOT 与 Native 算子包](./aot-packaging.md)。配置要事先枚举，但产物可审计、可复现。

要求严格可复现的线上服务仍然走 AOT；内部服务和 CI 用磁盘缓存通常就够了。
:::

### 清空缓存

```python
cannbotdsl.clear_compile_cache()                      # 只清进程内 L1
cannbotdsl.clear_compile_cache(clear_disk=True)       # 连磁盘 L2 一起清
```

完整签名是 `clear_compile_cache(*, clear_disk=False, reap_orphans=False, orphan_min_age_seconds=3600.0)`，全部为关键字参数。`reap_orphans=True` 回收孤儿产物，`orphan_min_age_seconds` 是回收年龄下限。

测试里常用无参形式，确保每个用例从干净状态开始。生产代码一般不需要。

### 在测试中预编译

`test/rms_norm/test_rms_norm.py` 的做法是在用例里显式 `compile` 一次，把编译开销和执行开销分开：

```python
cannbotdsl.compile(
    op.run,
    cannbotdsl.TensorSpec((2, num_col), dsl_dtype),
    cannbotdsl.TensorSpec((1, num_col), dsl_dtype),
    cannbotdsl.TensorSpec((2, num_col), dsl_dtype),
    cannbotdsl.TensorSpec((2, 1), cannbotdsl.dtypes.float32),
    ...
)
```

## 动态 shape 在 Kernel 里怎么用

Host 侧用 `Dim` 声明的动态维度，到了 Kernel 里就是运行期整数。它们可以：

- 作为 `cbd.range()` 的边界；
- 参与地址和循环次数计算；
- 作为 `Channel` / `Buffer` 的动态 shape（**必须同时提供静态 `capacity`**）。

它们不能：

- 作为 `range_constexpr()` 的参数；
- 作为 `const_expr()` 的条件；
- 决定片上资源的物理容量（容量由 `capacity` 决定）。

```python
# Kernel 内：动态行数 + 静态容量上界
ch = Channel(MemLoc.UB, shape=(rows, 128), dtype=dtypes.float32,
             depth=2, capacity=(MAX_ROWS, 128))
```

动态维的运行时值必须大于 0 且不超过 capacity，**设备侧不会执行越界检查**，这由调用方保证。

## 决策清单

写一个新算子时，按这个顺序想：

1. **哪些维度必须参与专门化？** 通常是最内层维度（决定矢量循环次数、UB 预算、是否对齐）。这些保持静态。
2. **哪些维度应该动态？** 通常是 batch、序列长度、token 数这类高频变化的外层维度。声明成 `Dim`，并给出合理的 `min` / `max` / `multiple_of`。
3. **哪些参数应该是 Kernel 标量形参？** 不影响代码结构的数值（`epsilon`、`scale`、`softmax` 的缩放系数）。
4. **哪些配置应该进 `@kernel` 类的 `__init__`？** 所有影响代码结构的东西：tile 大小、分支选择、流水深度。
5. **部署时需要 AOT 吗？** 需要的话在设计阶段就把配置集合枚举清楚。

## 下一步

[torch 接口与 stream 语义](./torch-interop.md)：把算子接到框架上。
