# AI CPU 与调度计划

变长序列、稀疏注意力、分页 KV Cache 这类算子有一个共同难题：**每个核该干多少活，编译期算不出来，运行期又不适合让 AI Core 串行去算。** `@aicpu_kernel` 就是为这件事准备的——先让 AI CPU 算出一份调度计划，再让 AI Core 按计划执行。

仓库里所有带 `_metadata` 后缀的样例都是这个模式：`flash_attn_metadata`、`flash_kda_metadata`、`qsa_indexer_metadata`、`stem_indexer_metadata`、`quant_block_sparse_attn_metadata`、`mixed_quant_sparse_flash_mla_metadata`、`quant_lightning_indexer_metadata`、`quant_sparse_lightning_indexer_metadata`。

::: info 阅读入口
本章从调度计划的角度介绍 AI CPU 与 AI Core 的分工。完整开发流程见 [AI CPU 编程概述](../api/aicpu/index.md)，参数和接口契约见 [AI CPU API](../api/aicpu/index.md)。
:::

## 三个执行位置，不是两个

[代码生成](./code-generation.md)讲过构图期与设备执行期。加上 AI CPU，一个完整算子最多跨三个执行位置：

| 位置 | 装饰器 | 什么时候算 | 典型职责 |
| --- | --- | --- | --- |
| Host CPU | `@host`、`@jit` | 构图期 | tiling 推导、实现选择、Kernel 启动 |
| **AI CPU** | `@aicpu_kernel` | **运行期，AI Core 之前** | 调度计划、元数据、轻量控制计算 |
| AI Core | `@kernel`、`@jit` | 运行期 | 搬运、Cube / Vector 计算 |

这里表格中的 Host 指 `@host` DSL 构图入口。它与普通 Python 运行时编排要区分：普通 Python 可以准备缓冲、调用 `compile_aicpu_kernel(...)`、通过 `compiled.launch(...)` 下发 AI CPU，再调用 AI Core 的 Host 入口；这不是在 `@host` 函数体内启动 AI CPU。

关键区别是 **Host 构图与 AI CPU 的执行时机不同**：

- Host 侧的计算在**构图期**完成，结果固化成编译期常量。它看不到运行时的张量内容；这不代表普通 Python 运行时不能处理数据。
- AI CPU 的计算在**运行期**完成，而且**能读 GM 上的真实数据**。因此适合根据每次调用的 `cu_seqlens` 生成调度计划。

## 什么时候该用 AI CPU

| 情况 | 放哪 |
| --- | --- |
| tile 大小、循环次数、分支选择 | Host（编译期） |
| 依赖张量**形状**但不依赖**内容** | Host，或声明成 `Dim` |
| 依赖张量**内容**的分核计划 | **AI CPU** |
| 变长序列的前缀和、有效长度 | **AI CPU** |
| 稀疏候选块的筛选与物理页映射 | **AI CPU** |
| 能用几条标量指令在 AI Core 上算完 | AI Core（别绕一趟） |

判据是：**这段计算是否依赖运行时数据，且结果要被所有核共享。** 两个条件同时成立才值得走 AI CPU，否则每个核各自算一遍更省。

## 接口一览

```python
from cannbotdsl.aicpu import (
    GmIn, GmOut,              # 输入 / 输出 GM 指针声明
    I32, I64, U32, U64,       # 标量类型
    aicpu_kernel,             # 装饰器
    current_raw_stream,       # 取当前 NPU stream
)
from cannbotdsl.aicpu.toolchain import compile_aicpu_kernel
```

| 名字 | 作用 |
| --- | --- |
| `@aicpu_kernel` | 声明一个在 AI CPU 上执行的函数 |
| `GmIn(T)` / `GmOut(T)` | 参数注解：这是一个指向 GM 的只读 / 可写指针，元素类型为 `T` |
| `I32` / `I64` / `U32` / `U64` | 标量与 GM 元素类型 |
| `current_raw_stream(device_id)` | 取当前设备的原生 stream，下发时需要 |
| [`compile_aicpu_kernel(...)`](../api/aicpu/build/compile.md) | 编译 AI CPU kernel，得到带 `launch(...)` 方法的对象 |
| `pack_struct_bytes` | 把结构化参数打包成字节 |
| `X86Buffer` | 本机 C++ 编译验证所用缓冲，不是 `run_host` 的必需参数 |
| `AicpuTraceError` | AI CPU 侧的异常类型 |

::: warning `compile_aicpu_kernel` 不在 `cannbotdsl.aicpu` 的 `__all__` 里
`cannbotdsl.aicpu` 导出的是：`aicpu_kernel`、`AicpuTraceError`、`GmIn`、`GmOut`、`I32`、`I64`、`U32`、`U64`、`X86Buffer`、`current_raw_stream`、`pack_struct_bytes`。

`compile_aicpu_kernel` 要从子模块导入：

```python
from cannbotdsl.aicpu.toolchain import compile_aicpu_kernel

compile_aicpu_kernel(kernel, *, workdir, launch_mode,
                     npu_arch="dav-3510", build_x86=False)
```
:::

## 参数怎么声明

AI CPU kernel 的参数用一个**参数类**描述，字段注解标明每个参数是输入还是输出、元素类型是什么：

```python
class MetadataParams:
    cu_seqlens_q: GmIn(I32)      # 输入：查询的累积序列长度
    cu_seqlens_kv: GmIn(I32)     # 输入：KV 的累积序列长度
    seqused_q: GmIn(I32)         # 输入：实际使用长度
    seqused_kv: GmIn(I32)
    metadata: GmOut(U32)         # 输出：调度计划
```

要点：

- **`GmIn` / `GmOut` 声明的是指针，不是张量。** AI CPU 侧按裸指针加偏移访问，没有 Layout 抽象，也没有 `tile_slice`。
- 标量参数直接用 `I32` / `I64` 这类类型注解。
- 输出同样由调用方预先分配，AI CPU 只负责填。

## 编译与下发

AI CPU kernel 不像 `@kernel` 那样由 `@host` 用方括号启动，而是**显式编译一次、按 stream 下发**：

```python
from functools import lru_cache

@lru_cache(maxsize=1)
def _get_compiled():
    return compile_aicpu_kernel(...)        # 全进程只编译一次


def run_metadata(sequences: dict, metadata: torch.Tensor, device_id: int):
    compiled = _get_compiled()
    compiled.launch(
        current_raw_stream(device_id),
        **{name: (0 if t is None else t.data_ptr()) for name, t in sequences.items()},
        metadata=metadata.data_ptr(),
    )
```

三件事值得注意：

1. **GM 参数传的是 `data_ptr()`**，即设备张量的裸地址。`0` 仅用于内核约定为可缺失的输入（示例中为 `None`），不表示任意空张量；内核必须通过分支保护缺失输入，不能解引用零地址。
2. **第一个参数是 stream。** AI CPU 与后续 AI Core kernel 下发到同一条 stream 时按顺序执行；跨 stream 消费或 Host 读取结果需要相应同步。下发后仍需保持输入、输出缓冲有效，直到设备工作完成。
3. **用 `lru_cache` 包住编译函数**是推荐做法——这里缓存的是普通 Python 函数的结果，安全；不要把 `lru_cache` 套在 DSL 装饰器上（见[类型系统与宿主语言边界](./type-system.md)）。

## AI Core 怎么消费调度计划

AI CPU 的输出是一块普通 GM 张量，通常是 `int32` / `uint32` 一维数组。AI Core 侧按**约定的字段偏移**用标量读取：

```python
# 约定好的字段下标，两侧必须一致
FD_WORKSPACE_IDX_INDEX = 2
FD_WORKSPACE_NUM_INDEX = 3

@kernel
class MyKernel:
    def __call__(self, metadata, ...):
        base = self._metadata_base_of(get_block_idx())
        slot = metadata[base + FD_WORKSPACE_IDX_INDEX]     # 运行期标量
        num = metadata[base + FD_WORKSPACE_NUM_INDEX]
        for i in cannbotdsl.range(num):                    # 运行期循环边界
            ...
```

要点：

- 读出来的是**运行期值**，只能用于 `cbd.range()` 的边界、地址计算和设备分支，不能进 `range_constexpr` 或 `const_expr`。
- 下标读取走标量流水，**只适合读少量元数据**。调度计划本身应该设计得紧凑（几十到几百个 int32），不要让 AI Core 去遍历大数组。
- 两侧的字段布局是**手工约定的契约**。样例的做法是把下标常量定义在一处、两边共享导入，这个习惯值得照搬。

## Host 侧参考运行

`@aicpu_kernel` 装饰出来的对象带一个 [`run_host`](../api/aicpu/validation/run-host.md) 方法，在 Host 上解释执行同一份函数体，用于检查调度逻辑：

```python
status = params_kernel.run_host(**kwargs)   # 按参数名传，结果写入 kwargs 中的输出数组
```

`run_host(self, **kwargs)` 只接受关键字参数。它返回函数的状态值（常见约定为 `0`），计算结果写入调用方准备的输出数组；无需为此构造 `X86Buffer`。可以将这些输出与独立的 Python 参考算法比较，再与设备结果比较。

Host 解释执行通过，只能说明该路径的参考逻辑符合预期，不能证明 C++ 生成、设备编译和 AI CPU 实际执行正确。设备验证还需下发、同步并比较输出，具体步骤见 [AI CPU 编程概述](../api/aicpu/index.md)。

## 一个完整的分工示例

以 `flash_attn` 为例，三层分工是这样的：

```text
┌─ Host（构图期）───────────────────────────────────────┐
│ 按 dtype / D / layout 选 tile 配置，决定启动多少核       │
└──────────────────────┬────────────────────────────────┘
                       ▼
┌─ AI CPU（运行期，AI Core 之前）──────────────────────┐
│ 读 cu_seqlens / seqused，算每个核负责哪些 batch、       │
│ 任务前缀和、核间任务区间、是否需要 FlashDecode 分片      │
│ → 写出 metadata 数组                                   │
└──────────────────────┬────────────────────────────────┘
                       ▼
┌─ AI Core（运行期）───────────────────────────────────┐
│ 按 get_block_idx() 从 metadata 取出自己的任务区间，      │
│ 搬运、QK、softmax、PV、写回                            │
└───────────────────────────────────────────────────────┘
```

好处是：**调度配置不变时 metadata 可以跨层复用**。`flash_kda` 把这一点做成了显式接口——`metadata` 是必选参数，由调用方决定复用还是重算。

## 注意事项

- **AI CPU 不是性能热点，但它在关键路径上。** 它的执行时间会直接加到算子总耗时里，所以调度算法本身要足够轻。复杂的筛选（例如稀疏 TopK）应该放在 AI Core 上做，AI CPU 只做分核规划。
- **AOT 打包时 AI CPU 是独立的一条编译路径。** 产物装到 `ops/_aicpu/`，运行时优先加载预编译文件；直接跑源码时走原有的即时编译路径。见 [AOT 与 Native 算子包](./aot-packaging.md)。
- **登记 AOT 时别漏文件。** metadata 算子通常是独立的 `.py`，必须在 `operator_groups.toml` 里和主算子一起列出，漏了会导致 wheel 内缺模块。

## 速查

| 你想做的事 | 用什么 |
| --- | --- |
| 声明 AI CPU 函数 | `@aicpu_kernel` |
| 声明输入 / 输出 GM 指针 | `GmIn(I32)` / `GmOut(U32)` |
| 编译 | `compile_aicpu_kernel(...)`，用 `lru_cache` 包住 |
| 下发 | `compiled.launch(current_raw_stream(device_id), **kwargs)`；GM 参数传设备地址，标量参数传整数 |
| AI Core 侧读计划 | `metadata[base + FIELD_INDEX]`，结果是运行期值 |
| 按计划循环 | `cbd.range(num)`，不能用 `range_constexpr` |
| 校验计划 | 用 `run_host(...)` 检查输出数组，再下发设备、同步并比较结果 |

## 下一步

[融合算子的设计方法论](./fusion-design.md)：调度计划算出来了，计算阶段怎么划。
