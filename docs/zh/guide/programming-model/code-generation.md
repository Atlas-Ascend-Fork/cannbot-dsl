# 代码生成

理解 CANNBot-DSL 的关键只有一句话：**同一份 DSL 代码会经历两个阶段——构图和设备执行。**

构图阶段跑的是 Python；设备阶段跑的是编译出来的二进制，不再是 Python。搞清楚哪一行属于哪个阶段，大部分「为什么这里报错」「为什么打印出来是这个」的困惑就消失了。

## 两个世界

```text
┌─ 构图期（编译期）──────────────────────────────────────┐
│  在 Host CPU 上，由 Python 解释器执行                   │
│                                                        │
│  • Python 的 int / float / str / list / dict           │
│  • math、numpy 等普通库调用                             │
│  • 类的 __init__、循环展开、分支选择                     │
│  • 内建 print()                                        │
│                                                        │
│  作用：决定生成什么样的设备代码                          │
└────────────────────────┬───────────────────────────────┘
                         │ 生成设备 IR → 毕昇编译器 → 二进制
                         ▼
┌─ 设备执行期（运行期）──────────────────────────────────┐
│  在 NPU AI Core 上执行                                 │
│                                                        │
│  • Tensor 中的真实数据                                  │
│  • Kernel 的标量形参                                    │
│  • mem_copy / matmul / v* 计算                          │
│  • cbd.print()                                          │
│                                                        │
│  作用：真正算出结果                                      │
└────────────────────────────────────────────────────────┘
```

## 编译流程

```text
  你的 Python 源码
   （@host / @jit / @kernel）
          │
          ▼
  ① 前端识别
     读取函数的 Python 语法结构，识别受支持的分支、循环、
     逻辑表达式和 DSL 调用
          │
          ├── 条件 / 迭代过程能在编译时确定
          │      └─► 在生成设备代码时直接选择、计算或展开
          │
          └── 依赖 Kernel 运行时数据
                 └─► 保留为设备执行时的分支或循环
          │
          ▼
  ② 构图：Python 解释器执行 DSL 函数体
     普通 Python 值直接参与计算；Tensor 操作、搬运、
     计算接口被记录为设备 IR
          │
          ▼
  ③ 后端编译：IR 降级、优化、生成目标架构（dav-3510）二进制
          │
          ▼
  ④ 加载并在 NPU 上执行
```

第 ① ② 步是构图：Python 解释器会执行 `@host` / `@jit` / `@kernel` 的函数体，以便收集控制流并记录搬运、计算接口。第 ④ 步才是真正算数，跑的是编译产物，不是再解释一遍 Python。

## 什么是编译期值，什么是运行期值

| 来源 | 属于 | 举例 |
| --- | --- | --- |
| Python 字面量、普通变量 | 编译期 | `VL = 64`、`tile = 128` |
| `@kernel` 类 `__init__` 里算出的值 | 编译期 | `self._num_seg = math.ceil(num_col / VL)` |
| `get_mem_size()` / `get_platform_info()` 的返回值 | 编译期 | `L1_SIZE = get_mem_size("l1")` |
| Host 侧 tiling 推导的参数 | 编译期 | `base_m`、`base_n`、`k_l1` |
| Kernel 的标量形参 | **运行期** | `def __call__(self, ..., epsilon: dtypes.float32)` |
| Tensor 的动态维度 | **运行期** | `nr = gm_x.shape[0]`（当 shape 是动态维时） |
| Tensor 元素读取值 | **运行期** | `x[i]` |
| `get_block_idx()` / `get_subblock_id()` | **运行期** | 每个核取值不同 |

::: warning 常见误区
`gm_x.shape[0]` 不一定是编译期值。用 `TensorSpec` 加 `Dim` 提前编译时它是运行期值；直接传真实 Tensor 即时编译时，框架会按实际 shape 专门化，它就是编译期值。写 kernel 时不要假设它是哪一种——需要编译期确定的地方（例如 `range_constexpr` 的上界）应该用 `__init__` 里算好的常量。
:::

## `print()` 与 `cbd.print()`

这是区分两个世界最直观的工具。

```python
import cannbotdsl as cbd


@cbd.kernel
def demo_kernel(x: cbd.Tensor, out: cbd.Tensor):
    tile = 128                     # 编译期值
    print("[构图期] tile =", tile)  # Python 内建 print，构图期打印

    buf = cbd.Channel(cbd.MemLoc.UB, (tile,), x.dtype, depth=1)
    cbd.mem_copy(buf.produce(), x)
    slot = buf.consume()

    cbd.print(slot, label="ub_data")   # 设备侧打印，执行期输出真实数据
```

- **`print()`** 在构图期执行，看到的是 Python 对象：tile 大小、shape、分支选择、你算出来的循环次数。用它调试「我生成了什么样的 kernel」。
- **[`cbd.print()`](/zh/api/kernel/debug/print.html)** 会被编译进 Kernel，在设备上执行，输出真实数据。用它调试「算出来对不对」。

::: danger `cannbotdsl` 把设备打印也命名为 `print`
包顶层导出的 `print` 就是设备侧打印（`cannbotdsl.core.diag.debug.print`）。

- `import cannbotdsl as cbd` → 没问题，`print` 还是内建，`cbd.print` 是设备打印。**本章一律这么写。**
- `from cannbotdsl import *` → **内建 `print` 被覆盖**，构图期打印会变成设备打印并在非 kernel 上下文报错。别这么写。
- `from cannbotdsl import print` → 同上，而且更隐蔽。
:::

`cbd.print` 的三条硬约束（完整清单见[调试与精度验证](/zh/guide/programming-model/debugging#设备侧调试-cbd-print)）：

- 只支持 **UB、L1、L0C** 上的 Tensor，以及 Kernel 局部标量和点索引标量。**看不到矢量寄存器**——那要用 [`dump_reg`](/zh/guide/programming-model/debugging#看寄存器-dump_reg)。
- Channel 不能直接打印，要先 `produce()` / `consume()` 取出 Tensor。
- 会增加设备内存占用并触发 Host 回读与同步，**不能用于性能测量**。

Host 侧还可以拿到结构化记录：

```python
from cannbotdsl.core.diag.debug import clear_debug_prints, get_debug_prints

clear_debug_prints()
run(...)
torch.npu.synchronize()
records = get_debug_prints()   # 含标签、dtype、shape、存储位置、block/subblock
```

::: tip 还有一条更直接的路
从 0.7.0 起可以直接把生成的 AscendC 源码 dump 出来：`CANNBOTDSL_DUMP_ASCENDC=1`。构图期 `print` 回答「我以为生成了什么」，`.asc` 文件回答「实际生成了什么」。见[编译选项与产物观察](/zh/guide/programming-model/compiler-options#转储-mlir-与-ascendc-源码)。
:::

## 元编程：用 Python 生成不同的 Kernel

因为构图期是真正的 Python 执行，所以你可以用任何 Python 手段去**选择**和**生成**设备代码。这是 CANNBot-DSL 性能调优的主要杠杆之一。

### 编译期分支选择实现

`samples/rms_norm/rms_norm.py` 的 Host 入口会先判断 UB 能否装下一整行，再决定用哪个 Kernel 类：

```python
@host
def run(self, gm_x, gm_gamma, gm_y, gm_rstd, eps: dtypes.float32):
    num_col = gm_x.shape[1]
    num_row = gm_x.shape[0]
    block_factor = math.ceil(num_row / DEFAULT_BLOCK_NUM)
    block_dim = math.ceil(num_row / block_factor)
    if const_expr(self._is_row_full_load(num_col, self.dtype)):
        op = RowFullLoadKernel(num_col, self.dtype)
    else:
        op = ColSplitKernel(num_col, self.dtype)
    op[block_dim](gm_x, gm_gamma, gm_y, gm_rstd, eps)
```

两个类生成的设备代码完全不同。运行期不存在这个判断，也不存在没被选中的那份代码。

### 编译期算好所有常量

`RowFullLoadKernel.__init__` 里做的全是普通 Python 算术：

```python
def __init__(self, num_col, dtype=dtypes.bfloat16):
    self._num_col = int(num_col)
    self._x_is_16bit = dtype in (dtypes.bfloat16, dtypes.float16)
    self._num_seg = math.ceil(self._num_col / VL)
    self._w = self._num_seg * VL
    ...
    ub_factor = (UB_SIZE - RETAINED_SIZE_1K - nca * gamma_bytes) // (...)
    self._row_factor = max(int(ub_factor), 1)

    if self._num_col <= VL:
        self._reduce_branch = 1
    elif self._num_col <= VL * 2:
        self._reduce_branch = 2
    ...
```

这些值随后在设备代码里以常量形式出现：

```python
if const_expr(self._reduce_branch <= 2):
    ...   # 只有这一支会进入生成的设备代码
else:
    ...
```

`const_expr()` 的作用就是告诉前端「这个条件在编译期已经确定，只保留命中的分支」。详见[控制流](/zh/guide/programming-model/control-flow)。

### 编译期查询硬件能力

```python
import cannbotdsl as cbd

TARGET_ARCH = cbd.get_platform_info().npu_arch
IS_DAV_3510 = TARGET_ARCH == "dav-3510"


@cbd.kernel
def compile_time_kernel(out: cbd.Tensor):
    if cbd.target_version(IS_DAV_3510):
        out[1] = 950
    else:
        out[1] = 0
```

`target_version()` 本身不查询硬件版本，版本信息和比较逻辑必须由调用方提供；它只是把一个已经算好的 Python 布尔值标记为编译期条件。

## 每种代码能写在哪

这张表来自各 API 页的「数据类型使用范围」，是排查「这行为什么报错」的首选。

| 用法 | 普通 Python | `@host` | `@host` 内的 `@jit` | `@kernel` | `@kernel` 内的 `@jit` |
| --- | :---: | :---: | :---: | :---: | :---: |
| 把 `Tensor` 用作类型注解 | ✓ | ✓ | ✓ | ✓ | ✓ |
| 接收或传递 Tensor 值 | ✗ | ✓ | ✓ | ✓ | ✓ |
| 读写或计算 Tensor 数据 | ✗ | ✗ | ✗ | ✓ | ✓ |
| `Buffer(...)` 分配片上存储 | ✗ | ✗ | ✗ | ✓ | ✓ |
| 创建 `Channel` | ✗ | ✗ | ✗ | ✓ | ✓ |
| `produce()` / `consume()` | ✗ | ✗ | ✗ | ✓ | ✓ |
| 创建 `DelayLineGroup` 并调用其方法 | ✗ | ✗ | ✗ | ✓ | ✓ |
| `const_expr()` / `target_version()` 作为控制流条件 | ✗ | 见下 | ✓ | ✓ | ✓ |
| 启动 Kernel（`k[n](...)`） | ✗ | ✓ | ✗ | ✗ | ✗ |

官方接口页写明：`const_expr()` / `target_version()` 作为 `if`/`while` 的完整条件时，只能用在 `@jit` 或 `@kernel` 里。`@host` 上若条件已经是 Python 布尔值，直接写 `if` 即可。`rms_norm` 的 `@host` 里也会写 `if const_expr(...)` 来挑选 Kernel 类——这是构图期选择实现，不是设备分支。

简化成一句话：**片上资源只能在设备上下文里创建，Kernel 只能在 `@host` 函数体里启动。**

## 三段执行位置

完整的一个算子最多会跨三种执行位置：

| 位置 | 装饰器 | 典型职责 |
| --- | --- | --- |
| Host CPU | `@host`、`@jit`（host 上下文） | tiling 推导、实现选择、Kernel 启动 |
| AI Core | `@kernel`、`@jit`（设备上下文） | 搬运、Cube / Vector 计算 |
| AI CPU | `@aicpu_kernel` | 元数据生成、分核调度计划、轻量控制计算 |

AI CPU 这一段在变长序列、稀疏注意力类算子里很常见：序列长度、每个核处理多少行这类信息无法在编译期确定，又不适合让 AI Core 串行算，就交给 AI CPU 先生成一份调度计划再启动 AI Core。仓库里带 `_metadata` 后缀的样例都是这个模式，接口见 [AI CPU API](/zh/api/aicpu/)。

## 常见问题

**构图期 `print` 打出来是一个对象而不是数值。**
说明这个值是运行期值（Tensor 读取、Kernel 标量形参、`get_block_idx()` 的结果等）。想看它的真实值要用 `cbd.print`。

**「运行时数据不能传给只在编译期处理的接口」。**
典型是把运行期整数传给了 `range_constexpr()` 或 `const_expr()`。改用 `cannbotdsl.range()` 或普通 `if`。

**改了 tile 大小但性能没变化。**
确认这个值确实参与了设备代码生成——如果它只是一个 Host 侧变量，没有传进 `@kernel` 类的 `__init__` 或 Kernel 形参，生成的代码不会变。

## 下一步

[类型系统与宿主语言边界](/zh/guide/programming-model/type-system)：这些值分别是什么类型，Python 的哪些能力能跨过构图期这条线。
