# 类型系统与宿主语言边界

[代码生成](/zh/guide/programming-model/code-generation)讲了「哪一行在构图期执行」。这一章回答紧接着的问题：**这些值是什么类型**，以及**Python 的哪些能力可以跨过构图期这条线**。踩坑最多的地方不是语法，而是把一个运行期值放在了只接受编译期值的位置上，或者反过来。

## 三类值

一份 DSL 代码里同时存在三种东西，它们能做的事完全不同：

| 类别 | 是什么 | 举例 | 能参与编译期算术 | 能进设备计算 |
| --- | --- | --- | :---: | :---: |
| **编译期值** | 真正的 Python 对象 | `VL = 64`、`self._tile`、`get_mem_size("ub")` | ✓ | 以常量形式固化 |
| **运行期标量** | 设备上的标量寄存器 | Kernel 标量形参、`get_block_idx()`、`x[i]` | ✗ | ✓ |
| **Tensor** | 存储位置 + 布局 | `gm_x`、`buf.consume()` | ✗ | ✓ |

判断一个值属于哪类，最快的办法是在构图期 `print` 它：**打出数值的是编译期值，打出对象的是运行期值。**

```python
@cbd.kernel
class K:
    def __init__(self, n):
        self._tile = n // 4
        print("tile =", self._tile)         # tile = 256 → 编译期值

    def __call__(self, gm_x, scale: dtypes.float32):
        bi = cbd.get_block_idx()
        print("bi =", bi)                   # bi = <对象> → 运行期值
        print("scale =", scale)             # scale = <对象> → 运行期值
```

## 编译期值的来源

| 来源 | 说明 |
| --- | --- |
| Python 字面量与普通变量 | 模块级常量、局部变量 |
| `@kernel` 类 `__init__` 里的任何算术 | `math.ceil`、整除、比较、分支选择 |
| `get_platform_info()` / `get_mem_size()` 的返回值 | 构图期就能拿到硬件参数 |
| Host 侧 tiling 推导出的参数 | `base_m`、`k_l1`、`used_core_num` |
| `@jit` 函数对编译期入参的返回值 | 会被内联，结果仍是编译期值 |

## 运行期值的来源

| 来源 | 说明 |
| --- | --- |
| Kernel 的标量形参 | `def __call__(self, ..., eps: dtypes.float32)` |
| `get_block_idx()` / `get_block_num()` / `get_subblock_id()` | 每个核取值不同 |
| Tensor 的元素读取 | `x[i]`、`metadata[base + 3]` |
| 用 `Dim` 声明的动态维度 | 到 Kernel 里就是运行期整数 |
| 运行期值参与的任何表达式 | 运行期性会传染 |

::: warning `shape[0]` 属于哪一类取决于你怎么编译
用 `TensorSpec` 加 `Dim` 提前编译时，`gm_x.shape[0]` 是**运行期值**；直接传真实 Tensor 即时编译时，框架按实际 shape 专门化，它是**编译期值**。写 Kernel 时不要依赖其中任何一种——需要编译期常量的地方（`range_constexpr` 的上界、`const_expr` 的条件）一律用 `__init__` 里算好的值。
:::

## 边界规则

### 规则一：运行期值不能进只接受编译期值的接口

```python
# ✗ dynamic_n 是运行期值
for i in cbd.range_constexpr(dynamic_n):
    ...
if cbd.const_expr(dynamic_n > 10):
    ...

# ✓
for i in cbd.range(dynamic_n):
    ...
if dynamic_n > 10:
    ...
```

报错信息形如「运行时数据不能传给只在编译期处理的接口」。

### 规则二：运行期值不能决定片上资源的物理容量

```python
# 动态 shape 必须同时给静态 capacity
ch = Channel(MemLoc.UB, shape=(rows, 128), dtype=dtypes.float32,
             depth=2, capacity=(MAX_ROWS, 128))
```

`capacity` 决定实际预留多少存储，必须是编译期值；`shape` 可以含运行期维度，但运行时值必须为正且不超过 capacity，**设备侧不做越界检查**。

### 规则三：类型在控制流里必须保持一致

设备控制流不支持依赖类型。同一个变量在不同分支或不同迭代里必须能统一到一种类型：

```python
# ✗ n 在分支里从整数变成浮点
n = 10
if pred:
    n = 10.0
```

### 规则四：控制流内创建的资源不能带出

```python
# ✗ buf 在分支内创建
if cond:
    buf = cbd.Buffer(cbd.MemLoc.UB, (128,), dtypes.float32)
cbd.mem_copy(out, buf)
```

改法是把资源声明提到控制流外面，通常放进 `@kernel` 类的 `__init__`，分支内只决定怎么用。详见[控制流](/zh/guide/programming-model/control-flow)。

## `self` 上能放什么

这是 `@kernel` 类写法最容易出问题的地方。文档推荐的组织方式是「`__init__` 放编译期配置，运行期参数走入口方法形参」——这句话的完整含义是：

| 放在哪 | 可以是 | 不建议 |
| --- | --- | --- |
| `__init__` 里赋给 `self` | tile 大小、循环次数、分支标志、`Buffer` / `Channel` 声明 | — |
| `__call__` 的形参 | Tensor、运行期标量 | — |
| `__call__` 里赋给 `self` | — | **运行期值**。它会让同一个对象在不同调用点携带状态，跨方法传递时作用域和类型都难以保证 |

安全范式：**运行期值通过参数在 `@jit` 方法之间传递，不要挂在 `self` 上。**

```python
@cbd.kernel
class K:
    def __init__(self, n, dtype):
        self._tile = compute_tile(n, dtype)          # ✓ 编译期配置
        self._x_ch = Channel(MemLoc.UB, (self._tile,), dtype, depth=2)   # ✓ 资源声明

    def __call__(self, gm_x, gm_y, eps: dtypes.float32):
        bi = cbd.get_block_idx()                     # 运行期值
        self._compute(gm_x, gm_y, bi, eps)           # ✓ 显式传参

    @cbd.jit
    def _compute(self, gm_x, gm_y, bi, eps):         # ✓ 形参接收
        ...
```

::: danger VF 内不要读成员变量
除了上面的作用域问题，在 `with vf(...)` 区域内访问 `self._xxx` 还有一个具体的性能后果：成员变量可能从栈溢出到矢量寄存器，既挤占[寄存器预算](/zh/guide/programming-model/vector-registers)，也会**阻断编译器的 VF 融合**。进入 VF 之前先取成局部变量。
:::

## Python 能力的可用范围

| 能力 | `__init__` / 普通 Python / 编译期展开的循环 | 设备循环体内 |
| --- | :---: | :---: |
| 列表 / 字典 / 元组作为容器 | ✓ | 作为编译期容器 ✓，运行期增删 ✗ |
| 列表推导、生成器表达式 | ✓ | ✗ |
| `lambda` | ✓ | ✗ |
| 嵌套函数、嵌套类 | ✓ | ✗ |
| `try` / `except` / `raise` | ✓ | ✗ |
| `yield`、`async` / `await` | ✓ | ✗ |
| `del`、`match` | ✓ | ✗ |
| `break` / `continue` / `return` | ✓ | ✗ |
| `math`、`numpy` 等普通库 | ✓ | ✗ |
| 类的继承与方法 | ✓（用于元编程） | ✗ |

一句话：**设备循环体里只能写「能被降成指令的东西」**；一切 Python 式的动态结构都属于构图期。

### `global` 与 `nonlocal`

不要在 DSL 函数里用 `global` 或 `nonlocal` 改写外层作用域的变量。构图期的求值时机与缓存复用都依赖函数体的纯度，靠全局变量传递配置会让「什么时候重新编译」变得不可预测。需要共享配置就通过 `@kernel` 类的构造参数或 `@datastruct` 传进去。

### `functools.lru_cache`

仓库样例确实用 `lru_cache` 缓存过 Host 侧的编译结果（例如 `flash_attn` 用它缓存 tile 配置、`flash_attn_metadata` 用它缓存 AI CPU 编译产物）。这是**对普通 Python 函数**的缓存，是安全的。

::: warning 不要用 lru_cache 直接包 DSL 装饰器函数
把 `lru_cache` 套在 `@host` / `@jit` / `@kernel` 外层是另一回事：DSL 对象通常与编译上下文绑定，按 `__hash__` 缓存可能把一个上下文里的对象拿到另一个上下文里用。要缓存编译产物，用 `cbd.compile(...)` 的返回值自己放进字典，见 [JIT 参数与编译缓存](/zh/guide/programming-model/jit-arguments)的「缓存与复用模式」。
:::

## 数值转换

| 想做的事 | 用什么 | 注意 |
| --- | --- | --- |
| 矢量侧转换数值类型 | `vcast(x, dtype, mask=..., rounding=...)` | 可指定舍入模式 |
| 标量侧转换数值类型 | `cast(x, dtype)` | 可配 `RoundingMode` |
| 按另一种 dtype 重新解释同一段字节 | `t.view(dtype=...)` / `t.reinterpret(dtype=...)` | **不做数值转换** |
| 搬入时展开 16 位到 32 位 lane | `vload_unpack(..., unpack_mode=UnpackMode.B16_TO_B32)` | 见[矢量寄存器](/zh/guide/programming-model/vector-registers) |
| 搬出时压回 16 位 | `vstore_pack(..., pack_mode=PackMode.B32_TO_B16)` | — |

::: info 待核实
Python 原生 `int` / `float` 作为 Kernel 标量形参传入时的默认设备类型宽度，以及混合类型表达式的提升规则，官方文档目前没有集中说明。保险做法是**显式标注形参类型**（`eps: dtypes.float32`），并在混合运算前显式 `cast`，不依赖隐式提升。
:::

## `vf()` 的参数不是循环参数

一个容易串台的地方：`unroll` 属于 `cbd.range()`，不属于 `vf()`。

```python
vf(*, mode="simd", thread=None, unroll=1, outputs=None)
```

`unroll` 和 `outputs` 虽然在签名里，但**传任何非默认值都会直接 `ValueError`**：

```python
with vf(mode="simd", unroll=4):      # ✗ ValueError
    ...

with vf(mode="simd"):                # ✓ 展开写在循环上
    for i in cbd.range(n, unroll=4):
        ...
```

`thread` 只对 `mode="simt"` 有意义：混合 SIMD + SIMT 编程由 VF 的 `thread=N` 指定线程数；纯 SIMT 编程省略该参数，由 Host 的 `dim3(thread)` 指定。见[两种 SIMT 编程模型](/zh/guide/programming-model/compute#两种-simt-编程模型)。

## 排查速查

| 报错或现象 | 先看 |
| --- | --- |
| 「运行时数据不能传给只在编译期处理的接口」 | 是不是把运行期值给了 `const_expr` / `range_constexpr` |
| 变量在控制流外不可用 | 是不是在 `if` / `for` 内部创建了资源或变量 |
| 类型不兼容 | 同一变量在不同分支 / 迭代里类型是否一致 |
| 构图期 `print` 打出对象而非数值 | 这个值是运行期值，要用 `cbd.print` 看 |
| 改了配置但生成代码没变 | 这个值是否真的进了 `__init__` 或 Kernel 形参 |
| VF 写长了反而变慢 | 活跃寄存器是否超过 32 个；VF 内是否读了 `self` |
| 每换一组参数就重新编译 | 哪些编译期值在变，能否改用 `Dim` |

## 下一步

[控制流](/zh/guide/programming-model/control-flow)：把编译期与运行期这条线落到每一个 `if` 和 `for` 上。
