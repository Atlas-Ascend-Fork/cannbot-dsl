# 控制流

官方控制流语义作用在 `@jit` 和 `@kernel` 上。前端读取函数的语法结构，**根据条件或迭代过程能否在编译时确定**，选择在生成设备代码时直接处理，或者保留为设备执行时的控制流。

`@host` 里的 `if`/`for` 首先是普通 Python：用来算 tiling、选实现、启动 Kernel。`cannbotdsl.range()` 虽然允许出现在 `@host` 的 `for` 里，但它是设备循环标记，不会返回可遍历的 Python `range` 对象。Host 侧遍历请用 `builtins.range()` 或直接迭代序列。

本节讲怎么控制这个选择，以及设备控制流有哪些限制。完整参考见 [API 文档 · 控制流](/zh/api/kernel/control-flow/control-flow.html)。

## 支持的写法

| 来源 | 支持的写法 |
| --- | --- |
| Python 语法 | `if` / `elif` / `else`、条件表达式、`and` / `or` / `not`、链式比较、`for`、`while`、`for ... else`、`while ... else` |
| Python 内建函数 | `min()`、`max()`、`any()`、`all()`、`builtins.range()` |
| CANNBotDSL API | `cannbotdsl.range()`、`range_constexpr()`、`const_expr()`、`target_version()` |

## 核心对照表

这张表是本节的全部要点。

| 控制流写法 | 设备执行时处理 | 生成设备代码时处理 |
| --- | :---: | :---: |
| `if const_expr(...)` | ✗ | ✓ |
| `if target_version(...)` | ✗ | ✓ |
| `if` 使用运行时条件 | ✓ | ✗ |
| `while const_expr(...)` | ✗ | ✓ |
| `while target_version(...)` | ✗ | ✓ |
| `while` 使用运行时条件 | ✓ | ✗ |
| `for i in range_constexpr(...)` | ✗ | ✓ |
| `for i in 固定 tuple/list` | ✗ | ✓ |
| `for i in builtins.range(...)` | ✓ | ✗ |
| `for i in cannbotdsl.range(...)` | ✓ | ✗ |
| `for i in cannbotdsl.range(..., unroll=N)` | ✓ | ✗ |
| `for i in cannbotdsl.range(..., unroll_full=True)` | ✓ | ✗ |

「生成设备代码时处理」= 条件选择或循环展开在编译过程中完成，设备执行时不再保留。
「设备执行时处理」= 前端保留相应控制流，由设备按运行时条件或循环边界执行。

::: warning 注意 `unroll`
`unroll` 和 `unroll_full` 是编译时确定的展开**设置**，但它们作用于**设备循环**，不会把 `cannbotdsl.range()` 变成编译期循环。
:::

## 分支

### 运行时分支

不加任何标记的 `if`，如果条件依赖运行期数据，就保留为设备分支：

```python
@cbd.kernel
def k(x: cbd.Tensor, out: cbd.Tensor):
    bi = cbd.get_block_idx()
    if bi < 8:              # 运行时条件 → 设备分支
        ...
    else:
        ...
```

### 编译期分支：`const_expr()`

把编译期已确定的 Python 值包进 `const_expr()`，编译器只保留命中的代码路径：

```python
if cbd.const_expr(USE_OFFSET):
    out[0] = 1
else:
    out[0] = 0
```

这是实现**功能开关**和**按形状特化**的标准做法。`rms_norm` 样例里到处是这种写法：

```python
if const_expr(self._reduce_branch <= 2):
    ...       # 小列数路径
else:
    ...       # 折半归约路径

if const_expr(is_16bit):
    xu = vload_unpack(x_ch, x_base, unpack_mode=UnpackMode.B16_TO_B32)
    x0 = vcast(xu, dtypes.float32, mask=preg)
else:
    x0 = vload(x_ch, x_base)
```

没被选中的那一支完全不会出现在生成的代码里——既不占指令空间，也不影响寄存器分配。

::: danger 不要把运行期值传给 `const_expr()`
```python
# ✗ 错误：dynamic_var 是运行期值
if cbd.const_expr(dynamic_var == 10):
    ...

# ✓ 正确
if dynamic_var == 10:
    ...
```
:::

### 编译期分支：`target_version()`

`target_version()` 专门用于标记**调用方已经算好的目标版本判断结果**。它本身不读取或比较硬件版本：

```python
TARGET_ARCH = cbd.get_platform_info().npu_arch
IS_DAV_3510 = TARGET_ARCH == "dav-3510"


@cbd.kernel
def k(out: cbd.Tensor):
    if cbd.target_version(IS_DAV_3510):
        ...
    else:
        ...
```

两个接口的 `value` 都必须在编译期确定，并且只能在 `@jit` / `@kernel` 修饰的上下文中作为控制流条件使用。

### 条件表达式与 `min` / `max`

```python
out[0] = x[0] if flag != 0 else x[1]
out[1] = min(x[0], x[1])
out[2] = max(x[0], x[1])
```

条件表达式的**两个候选表达式都会参与编译**，必须都是合法可追踪的；不能在候选表达式中创建新 Tensor 并带出表达式。`min()` / `max()` 的运行时形式只支持整数、index 或浮点标量，不接受运行时布尔值。详见[值选择](/zh/api/kernel/control-flow/value-selection.html)。

## 循环

### 常见的 `for` 写法

```python
import builtins
import cannbotdsl as cbd


@cbd.kernel
def k(out: cbd.Tensor, bound):
    # ① 设备循环，计数由编译期常量给出
    for i in builtins.range(4):
        ...

    # ② 设备循环，等价于 ①，但支持展开参数
    for i in cbd.range(4):
        ...

    # ③ 设备循环 + 展开提示
    for i in cbd.range(8, unroll=4):
        ...
    for i in cbd.range(4, unroll_full=True):
        ...

    # ④ 编译期展开，循环体被复制 4 份，设备上没有循环
    for i in cbd.range_constexpr(4):
        ...

    # ⑤ 设备循环，边界是运行期值
    for i in cbd.range(bound):
        ...

    # ⑥ 编译期展开固定序列
    for left, right in ((1, 2), (3, 4)):
        ...
```

选择原则：

| 场景 | 推荐 |
| --- | --- |
| 迭代次数大，循环体不小 | `cbd.range(n)` 或 `builtins.range(n)` |
| 迭代次数小且固定，想消除循环开销 | `cbd.range_constexpr(n)` |
| 迭代次数大，但想按组展开提升流水重叠 | `cbd.range(n, unroll=N)` |
| 迭代次数由运行期数据决定 | `cbd.range(bound)` |
| 遍历一组编译期配置（如多种 dtype、多个固定偏移） | 直接迭代 tuple/list |

::: tip `range_constexpr` 的代价
展开次数直接增加生成代码体积，适合较小且固定的迭代范围。循环体很大时全展开会拖慢编译、撑大 ICache，反而更慢。
:::

`cbd.range()` 是供前端识别的循环范围标记，**只能直接用于 `for ... in cbd.range(...)`**，它不会像 `builtins.range()` 那样返回可以被普通 Python 代码遍历的 `range` 对象。

### `while`

```python
# 编译期循环
while cbd.const_expr(n < 10):
    n += 1

# 设备循环
while dynamic_var == 10:
    ...
```

### `for ... else` / `while ... else`

语法支持，但由于设备循环不支持 `break`，`else` 块在设备循环结束后总会执行——不能表达 Python 里「通过 `break` 跳过 `else`」的语义。

## 设备控制流的六条约束

这些约束只作用于**不在编译期展开**的控制流。

| 约束类别 | 核心要求 |
| --- | --- |
| 路径完整性 | 控制流结束后使用的变量，必须在所有可能路径上都有确定值 |
| 状态结构一致性 | 跨分支或跨迭代传递的变量，数据结构必须一致 |
| 状态类型兼容性 | 同一变量在不同分支或迭代中的数值类型必须能够统一 |
| 作用域与资源生命周期 | 控制流内部创建的 Tensor 等资源不能直接带到外部；循环变量也不能直接带出循环 |
| 提前退出限制 | 不能通过 `break`、`continue`、`return`、`raise` 跳出 |
| 编译期与运行时边界 | 运行时数据不能传给只在编译期处理的接口 |

另外，设备循环体内不支持推导式、嵌套函数或类、`lambda`、`try`、`yield`、异步语法、`del` 和 `match`。

### 典型违规与改法

**想提前退出循环**

```python
# ✗ 设备循环不支持 break
for i in cbd.range(n):
    if found:
        break
```

改法一：把退出条件变成 `while` 的条件。

```python
i = 0
while i < n and not found:
    ...
    i = i + 1
```

改法二（更常用）：不退出，用掩码或条件写让后续迭代空转。矢量计算里直接用 `mask` 控制有效元素数量，是比分支更自然的写法：

```python
remaining = mask_counter(total)
for j in range(col_loops):
    preg, remaining = update_mask(remaining, elem_bits=32)
    ...                      # 超出范围的 lane 由 preg 屏蔽
```

**想把分支里创建的 Tensor 带出来**

```python
# ✗ buf 在分支内创建，分支外不可见
if cond:
    buf = cbd.Buffer(cbd.MemLoc.UB, (128,), dtypes.float32)
cbd.mem_copy(out, buf)
```

改法：把资源声明提到控制流外面（通常放在 `@kernel` 类的 `__init__` 里），分支内只决定怎么用。

```python
class K:
    def __init__(self, n):
        self._buf = Buffer(MemLoc.UB, (n,), dtypes.float32)

    def __call__(self, ...):
        if cond:
            mem_copy(self._buf, src_a)
        else:
            mem_copy(self._buf, src_b)
```

**变量在分支里改了类型**

```python
# ✗ n 在分支里从 int 变成 float
n = 10
if pred:
    n = 10.0
```

**循环变量带出循环**

```python
# ✗ 循环结束后 i 不可用
for i in cbd.range(n):
    ...
out[0] = i
```

改法：在循环外维护一个累加变量，或者直接用 `n` 推导。

## 写在控制流上的性能提示

- **把 `if` 尽量提到编译期**。设备分支在 AI Core 上有代价，尤其在紧内层循环里。能用 `const_expr` 消掉的分支就消掉。
- **内层循环优先用 `unroll`**，让搬运与计算有机会重叠。`unroll` 与 `unroll_full` 不能同时指定。
- **把运行期分支挪到循环外**。与其在循环里每次判断 `is_last_tile`，不如把主循环和尾块拆成两段写，这也是 `rms_norm`、`matmul` 样例的做法。
- **`cbd.range(start, stop, step)` 的 `step` 必须为正值**，想倒序遍历要自己做下标变换。

## 下一步

[数据、布局与切块](/zh/guide/programming-model/data-and-layout)：Tensor 是怎么被描述和切分的。
