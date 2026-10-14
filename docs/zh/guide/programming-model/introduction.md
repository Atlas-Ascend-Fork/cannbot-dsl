# 简介

## CANNBot-DSL 是什么

CANNBot-DSL 是一个用 Python 编写昇腾 NPU 算子的领域专用语言。你用普通 Python 语法描述数据怎么搬、怎么算，CANNBot-DSL 在调用时把这段描述编译成 AI Core 上的设备代码，并通过 PyTorch 的 NPU 后端执行。

它的位置在 CANNBot 仓群中：CANNBot 是 CANN 社区的 Infra 智能体层，用 Agent 完成 AscendC/PyPTO/TileLang/Triton 等各类语言的算子开发、模型迁移与推理优化。`cannbot-dsl` 是其中的 DSL 仓，提供 **Agent 亲和的编程范式**——写法显式、局部、可组合，既适合人读，也适合程序生成与校验。

一个最小的例子：

```python
import torch
import torch_npu  # noqa: F401  # 向 PyTorch 注册 Ascend NPU 后端

from cannbotdsl import Channel, host, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import update_mask, vadd, vload, vstore
from cannbotdsl.tensor import MemLoc


@kernel
def _reg_add_kernel(src0, src1, dst):
    buf0 = Channel(MemLoc.UB, (64,), src0.dtype, depth=1)
    buf1 = Channel(MemLoc.UB, (64,), src1.dtype, depth=1)
    out = Channel(MemLoc.UB, (64,), dst.dtype, depth=1)

    # GM 数据搬运至 UB
    mem_copy(buf0.produce(), src0)
    mem_copy(buf1.produce(), src1)

    in0 = buf0.consume()
    in1 = buf1.consume()
    res = out.produce()
    # Vector Function：更新 mask、搬入寄存器、计算、搬出
    with vf(mode="simd"):
        mask, _ = update_mask(64, 32)
        acc = vadd(vload(in0, 0), vload(in1, 0), mask=mask)
        vstore(res, 0, acc, mask)

    # UB 数据搬运至 GM
    mem_copy(dst, out.consume())


@host
def run(src0, src1, dst):
    _reg_add_kernel[1](src0, src1, dst)


def main():
    src0 = torch.arange(64, dtype=torch.float32, device="npu:0")
    src1 = torch.arange(64, dtype=torch.float32, device="npu:0") + 1.0
    dst = torch.empty_like(src0)

    run(src0, src1, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), (src0 + src1).cpu())
    print("reg compute example passed")


if __name__ == "__main__":
    main()
```

即使还没读过后面的章节，这段代码里也已经出现了编程模型的全部骨架：

- `@kernel` 标出跑在 AI Core 上的代码，`@host` 标出从 Python 调用的入口。
- `Channel` 声明片上存储，`mem_copy` 在存储层级之间搬数据。
- `with vf(...)` 圈出一段矢量计算。`mode="simd"` 是寄存器级 SIMD，也是主力模式；另有 `mode="simt"` 的线程模型，见[三类计算单元](/zh/guide/programming-model/compute#simt-模式)。
- `_reg_add_kernel[1]` 中方括号里的 `1` 是启动的 block 数量。
- 输出 Tensor `dst` 由调用方准备好传进来，kernel 不返回值。

## 三个设计取舍

### 1. 存储层级是显式的

很多编程模型把片上存储和寄存器的分配交给编译器决定。CANNBot-DSL 不这样做：UB、L1、L0A、L0B、L0C 等每一级片上存储都要你用 `Buffer` 或 `Channel` 显式声明，`mem_copy` 的每一次搬运都要你显式写出来。

原因是昇腾 AI Core 的性能几乎完全由数据通路决定：同一个算子，把数据停在 L1 复用还是每次从 GM 读，性能可以差一个数量级。把这件事交给编译器猜，调优时就没有抓手了。

### 2. 同步由框架生成，排布由你决定

显式搬运最容易出错的地方是同步：什么时候数据已经搬完可以算了、什么时候上一轮算完了可以覆盖缓冲区了。这部分样板代码由 `Channel` 承担——你声明槽位数量 `depth`，用 `produce()` 取写入槽位、`consume()` 取读取槽位。这两个调用本身只推进游标、不会等待；搬运与计算之间的等待由 lowering 按配对插入。VF 里的 UB 重叠仍要自己写 `vmem_bar`。

但 `Channel` 只消除同步样板，**不替你做软件流水调度**。预取几拍、主循环怎么排、尾部怎么排空，仍然是你写出来的程序顺序决定的。详见[片上存储与流水](/zh/guide/programming-model/onchip-memory)。

### 3. Python 既是宿主语言，也是元编程语言

你的 DSL 函数会在构图期被 Python 解释器真正执行一遍。这意味着 `math.ceil`、类的 `__init__`、编译期 `if`/`else` 选择不同实现，这些 Python 能力都可以用来**生成**设备代码，而生成出来的设备代码里不会留下它们的痕迹。

注意边界：列表推导、`lambda`、`try` 等在**设备循环体**里并不支持。它们只适合写在 `__init__`、普通 Python 或编译期展开的循环中。详见[控制流](/zh/guide/programming-model/control-flow)。

仓库里的 `rms_norm` 样例就是典型：Host 侧先算 UB 够不够放下一整行，够就实例化 `RowFullLoadKernel`，不够就实例化 `ColSplitKernel`，两个类生成完全不同的设备代码。详见[代码生成](/zh/guide/programming-model/code-generation)。

## 六个装饰器

CANNBot-DSL 的全部程序结构由六个装饰器声明。

| 装饰器 | 主要用途 | 从哪导入 | 典型使用位置 |
| --- | --- | --- | --- |
| `@host` | 定义可从普通 Python 调用的 Host 编排入口 | `cbd.host` | 算子入口、Kernel 启动 |
| `@jit` | 定义 DSL 内部的内联辅助函数 | `cbd.jit` | 可复用的 Host 或设备逻辑 |
| `@kernel` | 定义在 NPU AI Core 上执行的 Kernel | `cbd.kernel` | 算子的设备计算代码 |
| `@datastruct` | 定义可传入 DSL 程序的结构化数据 | `cbd.datastruct` | Kernel 配置、复合参数 |
| `@aicpu_kernel` | 定义在 AI CPU 上执行的 Kernel | **`from cannbotdsl.aicpu import aicpu_kernel`** | 元数据、调度或轻量控制计算 |
| `@export` | 声明需要发布的算子配置 | **`from cannbotdsl.aot import export`** | Native 算子包的构建入口 |

前四个在包顶层，后两个**不在**——它们分别属于 `cannbotdsl.aicpu` 和 `cannbotdsl.aot` 子模块，写 `cbd.aicpu_kernel` 会 `AttributeError`。

完整说明见 [API 文档 · 装饰器](/zh/api/decorators.html)。下面只讲它们之间的关系。

### `@host`：唯一的外部入口

按[装饰器](/zh/api/decorators.html)的约定，普通 Python 代码应当只直接调用 `@host` 函数。它负责：根据运行时参数做 Host 侧 tiling、挑选 Kernel 实现、启动 Kernel。

部分 API 示例页仍用 `@jit` 包一层 `run()` 再启动 Kernel。那是示例写法，不是推荐入口：`@jit` 不能从普通 Python 调用，也不能作为 `cbd.compile(...)` 的编译入口；Kernel 也不能由 `@jit` 间接启动。

```python
import cannbotdsl as cbd


@cbd.kernel
def add_kernel(x, y, out):
    ...


class Add:
    @cbd.host
    def run(self, x, y, out):
        add_kernel[8](x, y, out)
```

`@host` 函数应返回 `None`；输出 Tensor 由调用方准备，通过参数传入。它既可以直接调用：

```python
Add().run(x, y, out)
```

也可以用 `TensorSpec` 等参数规格提前编译：

```python
spec = cbd.TensorSpec((1024,), cbd.dtypes.float16)
program = cbd.compile(Add().run, spec, spec, spec)
program(x, y, out)
```

### `@kernel`：设备代码

`@kernel` 定义在 AI Core 上执行的函数，返回值应为 `None`。SIMD 与混合 SIMD + SIMT Kernel 在方括号内指定整数 block 数量：

```python
add_kernel[8](x, y, out)   # 启动 8 个 block
add_kernel(x, y, out)      # 省略时默认为 1
```

纯 SIMT Kernel 则由 Host 传 `dim3(block)` 和 `dim3(thread)`；两种启动形式见[SIMT · Host 启动规则](/zh/guide/programming-model/compute#host-启动规则)。SIMT 需要 CANN 9.2.0 及以上版本。

`@kernel` 也可以装饰类，用于组织多个相关的 Kernel 方法和共享配置：

```python
@cbd.kernel
class AddKernel:
    def __init__(self, tile_size):
        self.tile_size = tile_size      # 编译期配置

    def __call__(self, x, y, out):      # 运行时参数
        ...


@cbd.host
def add(x, y, out):
    AddKernel(128)[8](x, y, out)
```

类装饰器记录构造参数，并在设备函数构建时执行 `__init__`。**构造参数用于传递编译期配置，运行时参数应通过 Kernel 入口方法传入。** 这个区分很重要：`__init__` 里算出来的 tile 大小、分支选择、循环次数都会被固化进生成的设备代码。

`@kernel` 也可以直接装饰普通类的方法，`samples/matmul/matmul/matmul.py` 就是这种写法：

```python
class MatmulKernel:
    def __init__(self, tiling: MatmulTiling):
        self.t = tiling
        ...

    @kernel
    def matmul_kernel(self, gm_a, gm_b, gm_c):
        ...

    @host
    def run(self, gm_a, gm_b, gm_c):
        self.matmul_kernel[self.t.used_core_num](gm_a, gm_b, gm_c)
```

### `@jit`：拆分与复用

`@jit` 定义 DSL 内部的内联辅助函数，可由 `@host`、`@kernel` 或其他 `@jit` 函数调用。它沿用调用方的执行环境（Host 或设备），可以有返回值。

```python
@cbd.jit
def tile_offset(block_idx, tile_size):
    return block_idx * tile_size


@cbd.kernel
def add_kernel(x, y, out):
    offset = tile_offset(cbd.get_block_idx(), 128)
    ...
```

`@jit` 函数会被内联，不产生调用开销。把长 kernel 按阶段拆成若干 `@jit` 方法是推荐做法——`rms_norm` 样例就拆成了 `_compute_x_squared_sum`、`_compute_rstd`、`_compute_y` 三段。

### 调用约定

| 调用方 | 被调用方 | 是否允许 | 说明 |
| --- | --- | :---: | --- |
| 普通 Python | `@host` | ✓ | 触发编译、缓存复用和执行 |
| 普通 Python | `@jit` | ✗ | `@jit` 只能在 DSL 内部调用 |
| 普通 Python | `@kernel` | ✗ | Kernel 必须由 `@host` 启动 |
| `@host` | `@kernel` | ✓ | 启动语句必须直接位于 `@host` 函数体中 |
| `@host` | `@jit` | ✓ | 编译期内联 |
| `@host` | 普通 Python 函数 | ✓ | 构图期执行，结果作为编译期常量 |
| `@jit` | `@kernel` | ✗ | 不能由 `@jit` 辅助函数间接启动 Kernel |
| `@kernel` | `@jit` | ✓ | 编译期内联 |
| `@kernel` | 普通 Python 函数 | ✓ | 构图期执行 |
| `@kernel` | `@kernel` | ✗ | Kernel 之间不能互相启动 |

一句话记忆：**Kernel 启动只能写在 `@host` 函数体里**。

### `@datastruct` 与 `@aicpu_kernel`

`@datastruct` 声明一组有名字和类型的字段，用于成组传递配置：

```python
@cbd.datastruct
class TileConfig:
    rows: cbd.dtypes.int32
    columns: cbd.dtypes.int32


config = TileConfig(rows=64, columns=128)
```

类体仅用于字段声明，不应定义默认值、继承关系或业务方法。

`@aicpu_kernel` 定义在 AI CPU 上执行的函数，通常用于生成元数据、计算调度信息或完成轻量控制任务。仓库里凡是带 `_metadata` 后缀的样例（`flash_attn_metadata`、`qsa_indexer_metadata`、`mixed_quant_sparse_flash_mla_metadata` 等）都走这条路径：先由 AI CPU 算出每个核该处理哪些 tile，再由 AI Core 按计划执行。

它的接口自成一套（`GmIn` / `GmOut` 声明指针、按 stream 下发），和 `@kernel` 完全不同，详见 [AI CPU 与调度计划](/zh/guide/programming-model/aicpu)。

## 一个算子通常包含哪几层

把仓库里任意一个样例拆开，都是同一个三层结构：

```text
┌──────────────────────────────────────────────────────┐
│ 第 3 层：torch 接口函数（普通 Python）                │
│   校验 shape/dtype、分配输出、reshape、调用 @host      │
│   例：samples/rms_norm/rms_norm.py 的 rms_norm()      │
├──────────────────────────────────────────────────────┤
│ 第 2 层：Host 侧 tiling + @host 入口                  │
│   算 tile 大小、选实现、决定 block 数、启动 Kernel     │
│   例：MatmulTiling 类 + MatmulKernel.run()            │
├──────────────────────────────────────────────────────┤
│ 第 1 层：@kernel 设备代码                             │
│   声明 Channel、切 tile、搬运、Cube/Vector 计算、写回  │
│   例：MatmulKernel.matmul_kernel()                    │
└──────────────────────────────────────────────────────┘
```

读样例时按这个顺序从下往上看（先看 kernel 干了什么，再看 tiling 怎么决定参数，最后看接口怎么包装），通常比从上往下顺畅。

## 下一步

- 想先建立硬件直觉：[硬件与执行模型](/zh/guide/programming-model/hardware-model)
- 想先搞清楚「我的 Python 到底什么时候执行」：[代码生成](/zh/guide/programming-model/code-generation)
- 想直接动手写向量算子：[写出第一个算子](/zh/guide/programming-model/first-operator)
- 想直接动手写矩阵乘：[第一个 Cube 算子与第一个 Mix 算子](/zh/guide/programming-model/cube-and-mix)
- 有 Ascend C 背景，想先对齐概念：[附录](/zh/guide/programming-model/appendix)
