# AI CPU 与调度计划

变长序列、稀疏注意力、分页 KV Cache 这类算子有一个共同难题：**每个核该干多少活，编译期算不出来，运行期又不适合让 AI Core 串行去算。** `@aicpu_kernel` 就是为这件事准备的——先让 AI CPU 算出一份调度计划，再让 AI Core 按计划执行。

仓库里所有带 `_metadata` 后缀的样例都是这个模式：`flash_attn_metadata`、`flash_kda_metadata`、`qsa_indexer_metadata`、`stem_indexer_metadata`、`quant_block_sparse_attn_metadata`、`mixed_quant_sparse_flash_mla_metadata`、`quant_lightning_indexer_metadata`、`quant_sparse_lightning_indexer_metadata`。

::: info 本章的核实口径
AI CPU 在官方文档站的 [/zh/api/aicpu/](/zh/api/aicpu/) 目前只是一个接口名索引，没有接口详情页。本章内容以 `samples/flash_attn/flash_attn_metadata.py` 的实际实现为依据整理，用于说明**这条路径长什么样**；参数的确切语义请以样例和后续发布的接口文档为准。
:::

## 三个执行位置，不是两个

[代码生成](/zh/guide/programming-model/code-generation)讲过构图期与设备执行期。加上 AI CPU，一个完整算子最多跨三个执行位置：

| 位置 | 装饰器 | 什么时候算 | 典型职责 |
| --- | --- | --- | --- |
| Host CPU | `@host`、`@jit` | 构图期 | tiling 推导、实现选择、Kernel 启动 |
| **AI CPU** | `@aicpu_kernel` | **运行期，AI Core 之前** | 调度计划、元数据、轻量控制计算 |
| AI Core | `@kernel`、`@jit` | 运行期 | 搬运、Cube / Vector 计算 |

关键区别是 **Host 与 AI CPU 都是 CPU，但执行时机完全不同**：

- Host 侧的计算在**构图期**完成，结果固化成编译期常量。它看不到运行时的张量内容。
- AI CPU 的计算在**运行期**完成，而且**能读 GM 上的真实数据**。`cu_seqlens` 这种每次调用都变的东西，只有它能算。

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
| `compile_aicpu_kernel(...)` | 编译 AI CPU kernel，得到可调用对象 |
| `pack_struct_bytes` | 把结构化参数打包成字节 |
| `X86Buffer` | x86 侧缓冲（用于 Host 参考运行） |
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
    compiled(
        current_raw_stream(device_id),
        **{name: (0 if t is None else t.data_ptr()) for name, t in sequences.items()},
        metadata=metadata.data_ptr(),
    )
```

三件事值得注意：

1. **参数传的是 `data_ptr()`**，即 torch 张量的裸地址。空张量传 `0`。
2. **第一个参数是 stream。** AI CPU 的执行被排在同一条 stream 上，所以它与后续的 AI Core kernel 之间天然有序，不需要额外同步。
3. **用 `lru_cache` 包住编译函数**是推荐做法——这里缓存的是普通 Python 函数的结果，安全；不要把 `lru_cache` 套在 DSL 装饰器上（见[类型系统与宿主语言边界](/zh/guide/programming-model/type-system)）。

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

`@aicpu_kernel` 装饰出来的对象带一个 `run_host` 方法，在 Host 上解释执行同一份函数体，用于校验调度计划本身是否正确：

```python
golden = params_kernel.run_host(**kwargs)    # 按参数名传，不是位置参数
```

::: info 包内可用 · 文档站未收录
签名是 `run_host(self, **kwargs)`（`cannbotdsl/aicpu/kernel.py`），内部走 AI CPU 解释器的 host 入口。**只接受关键字参数。**

官方文档站没有这个接口的详情页，返回形式也未成文。更稳妥的做法仍然是样例的结构：`flash_attn_metadata.py` 把调度逻辑写成可独立调用的纯 Python 函数，再分别喂给 AI CPU kernel 和 CPU 参考路径，两边比对。**调度算法与执行载体解耦**比依赖某个具体接口更抗版本变化。
:::

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
- **AOT 打包时 AI CPU 是独立的一条编译路径。** 产物装到 `ops/_aicpu/`，运行时优先加载预编译文件；直接跑源码时走原有的即时编译路径。见 [AOT 与 Native 算子包](/zh/guide/programming-model/aot-packaging)。
- **登记 AOT 时别漏文件。** metadata 算子通常是独立的 `.py`，必须在 `operator_groups.toml` 里和主算子一起列出，漏了会导致 wheel 内缺模块。

## 速查

| 你想做的事 | 用什么 |
| --- | --- |
| 声明 AI CPU 函数 | `@aicpu_kernel` |
| 声明输入 / 输出 GM 指针 | `GmIn(I32)` / `GmOut(U32)` |
| 编译 | `compile_aicpu_kernel(...)`，用 `lru_cache` 包住 |
| 下发 | 第一个参数传 `current_raw_stream(device_id)`，其余传 `data_ptr()` |
| AI Core 侧读计划 | `metadata[base + FIELD_INDEX]`，结果是运行期值 |
| 按计划循环 | `cbd.range(num)`，不能用 `range_constexpr` |
| 校验计划 | 把调度算法写成纯 Python 函数，两边比对 |

## 下一步

[融合算子的设计方法论](/zh/guide/programming-model/fusion-design)：调度计划算出来了，计算阶段怎么划。
