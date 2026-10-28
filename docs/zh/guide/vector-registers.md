# 矢量寄存器与 lane 模型

Reg 编程模型的全部性能来源是「让中间结果留在寄存器里」。要用好它，先得知道一个寄存器有多宽、里面的元素怎么排、你一共有多少个。这一章只讲这三件事，它们是后面读懂掩码、`unpack` / `pack` 和 `unroll` 的前提。

## 一个数字就够：VL = 256 字节

Ascend 950PR / 950DT 的**矢量数据寄存器位宽是 256 字节**。可以在设备侧查询，不必硬编码：

```python
from cannbotdsl.ops.arch import get_vf_len

vl_bytes = get_vf_len()      # 在 Ascend 950PR / 950DT 上返回 256
```

::: warning 它的 docstring 写的是「in bits」，实际返回的是字节数
`cannbotdsl/ops/arch.py` 里 `get_vf_len` 的注释是 "vector register width VL in bits"，但真机返回 **256**，而 256 bit = 32 字节显然不对。按**字节**理解：`VL = 256 B`，元素数 = `256 * 8 / elem_bits`。包内的真机测试断言的也是 `vf_len == 256`。
:::

矢量单元每周期处理 256 字节。由此推出一个寄存器能放多少元素：

| 元素类型 | 元素位宽 | 一个寄存器的元素数 |
| --- | :---: | :---: |
| int8 / uint8 / fp8 | 8 | 256 |
| float16 / bfloat16 | 16 | 128 |
| float32 / int32 | 32 | **64** |
| int64 | 64 | 32 |

**这就是样例里那个 `VL = 64` 的来历**：fp32 路径下 256 ÷ 4 = 64。它不是一个任意选定的常数，而是「fp32 的 lane 数」。同一份代码改成 fp16 计算时，一个寄存器能装 128 个元素，循环次数会变。

::: tip 为什么样例仍把 64 写成常量
`get_vf_len()` 是设备侧接口，而 tile 大小和循环次数需要在**构图期**算好。所以样例的做法是在 Host 侧按 dtype 推出 lane 数当编译期常量用，设备侧接口留给需要运行期判断的场合。真正要避免的是把 64 当成「和 dtype 无关的魔法数」。
:::

## lane 模型

把一个矢量寄存器想成一排等宽的槽，每个槽叫一个 **lane**：

```text
RegTensor<float32>，VL = 256 B
┌────┬────┬────┬────┬ ... ┬────┐
│ l0 │ l1 │ l2 │ l3 │     │l63 │   64 条 lane，每条 32 bit
└────┴────┴────┴────┴ ... ┴────┘

RegTensor<float16>，同一个物理寄存器
┌──┬──┬──┬──┬──┬──┬ ... ┬────┐
│l0│l1│l2│l3│l4│l5│     │l127│   128 条 lane，每条 16 bit
└──┴──┴──┴──┴──┴──┴ ... ┴────┘
```

所有 `v*` 接口都是**逐 lane**运算：`vadd(a, b)` 是 64（或 128）个加法同时做，不是一个向量加法。

### 掩码就是 lane 的开关

掩码寄存器宽度是 **VL / 8 = 32 字节 = 256 位**，正好每个 lane 一位——这是按最小元素位宽（8 位、256 lane）设计的。所以 `update_mask` 必须同时知道**元素数量**和**元素位宽**才能算出该点亮哪些位：

```python
mask, _ = update_mask(64, 32)        # 64 个元素，每个 32 位 → 刚好一整个寄存器
preg, remaining = update_mask(remaining, elem_bits=32)   # 从计数器迭代推进
```

`elem_bits` 不是可选的装饰，它决定一个「元素」占几条位。把 32 写成 16 会让掩码覆盖范围差一倍。

::: danger 掩码必须先赋值
支持 `mask` 参数的接口，掩码必须先经 `full_mask` / `create_mask` / `update_mask` 赋值。未赋值的掩码寄存器内容不确定，表现是「有效元素位置错乱」而不是报错。
:::

### 16 位数据为什么要 unpack

Cube 和 UB 上的 bf16 / fp16 是紧凑存放的，但**矢量运算的中间精度通常要 fp32**。如果先搬进来再 `vcast`，就要占两个寄存器周期；更麻烦的是 128 lane 的 16 位数据转成 fp32 装不进一个寄存器。

所以 Reg 模型提供了**搬入即展开、搬出即压回**：

```python
xu = vload_unpack(x_ch, offset, unpack_mode=UnpackMode.B16_TO_B32)   # 16 位 → 32 位 lane
x = vcast(xu, dtypes.float32, mask=full)
...
vstore_pack(y_buf, offset,
            vcast(yval, out_dtype, mask=full, rounding=RoundingMode.RN),
            full, pack_mode=PackMode.B32_TO_B16)                     # 32 位 lane → 16 位
```

理解了 lane 宽度，这段代码就不再是咒语：`B16_TO_B32` 的意思是「把 128 个 16 位元素摊成 32 位 lane 来算」，一次处理其中 64 个。

## 寄存器预算

| 资源 | 数量 | 单个宽度 |
| --- | :---: | --- |
| 矢量数据寄存器 | **32** | 256 B（VL） |
| 掩码寄存器 | 8 | 32 B（VL / 8） |
| 地址寄存器 | 4 B × 8 | — |
| 非对齐搬入 / 搬出寄存器 | 各 4 | 32 B |

两条要记住的规则：

1. **同时存活的矢量寄存器超过 32 个会溢出到 UB**，占用 UB 顶部预留的 **6 KB** VF 溢出区，并付出访存代价。症状是「VF 写得越长反而越慢」。
2. **寄存器的生命周期限制在单个 VF 内。** 跨 `with vf(...)` 作用域传递寄存器值是不行的；要跨段保留就得落 UB。

::: tip 6 KB 还是 8 KB
两个数都对，但指的不是同一块：UB 顶部一共预留 **8 KB** = **6 KB** VF 寄存器溢出区 + **2 KB** native dump 缓冲（`dump_tensor` / `dump_reg` 用）。寄存器溢出能用的是 6 KB。

这 8 KB **已经从 `get_mem_size("ub")` 的返回值里扣掉了**（返回 253952 = 248 KB），算 UB 预算时不要再减一次。
:::

这两条合起来给出了 VF 的长度判据：**一段 VF 的活跃中间量应当控制在 32 个寄存器以内**。超了就该切分 VF，或者主动把中间结果搬到 UB 换取寄存器。

::: warning 不要在 VF 内访问对象成员变量
在 VF 区域里读 `self._xxx` 可能导致编译器把成员变量从栈溢出到矢量寄存器，既吃掉寄存器预算，也会**阻断 VF 融合**。推荐写法是在进入 `with vf(...)` 之前把需要的值取成局部变量：

```python
@jit
def _compute(self, x, out):
    w = self._w                      # 先取成局部变量
    seg = self._num_seg
    with vf(mode="simd"):
        for j in range(seg):         # VF 内只用局部变量
            ...
```
:::

## 对齐

| 项 | 要求 |
| --- | --- |
| `vload` / `vstore` 的 UB 起始地址 | **32 字节**对齐 |
| 不满足对齐时 | 用 `vload_unalign` / `vstore_unalign_post`，有额外代价 |
| tile 长度 | 建议对齐到 lane 数（fp32 下为 64） |

对不齐的时候优先调整 tile 大小去迁就对齐，而不是用非对齐接口硬扛。

## RegLayout

部分搬入 / 搬出接口接受 `reg_layout` 参数，用于指定数据在寄存器内的排布变体，样例里出现在成对的解交织加载上：

```python
even_exp, odd_exp = rr.vload_deinterleave(qk_ch, base, width="b32")
rr.vstore_strided(..., reg_layout=rr.RegLayout.ZERO)
rr.vstore_strided(..., reg_layout=rr.RegLayout.ONE)
```

::: info 包内可用 · 文档站未收录
`RegLayout` 在官方 API 文档站只出现在[关键特性](../api/compute/vector/key-features.md)的说明文字里，没有独立接口页。但包里（`cannbotdsl/ops/reg/cast.py`）的定义是明确的：

| 成员 | 值 | 什么时候用得到 |
| --- | :---: | --- |
| `UNKNOWN` | −1 | 不指定，由接口推导 |
| `ZERO` | 0 | 宽度比 2 和 4 都用 |
| `ONE` | 1 | 宽度比 2 和 4 都用 |
| `TWO` | 2 | **仅宽度比 4** |
| `THREE` | 3 | **仅宽度比 4** |

语义是「选取第几路分量」。源与目标元素位宽之比为 2 时（例如 b32 ↔ b16）只有 `ZERO` / `ONE` 两路，这就是 `flash_attn` 样例里只见到这两个的原因；位宽比为 4 时（fp4 / fp8 相关的部件选择）才会用到 `TWO` / `THREE`。只照样例写容易误以为一共只有两个取值。
:::

## 和 Hardware Loop 的关系

Hardware Loop 的约束（迭代变量从 0 起、步长 1、循环体内无运行时跳转、边界不可变、最多四层嵌套）作用在**生成代码**上，不是让你在 Python 里声明类型。但它和寄存器预算是一对相互制约的条件：

- 展开（`unroll`）能提升指令双发机会，但会**成倍增加活跃寄存器数**，撞上 32 个上限后反而溢出。
- 指令过多会触发 ICache Miss，此时应切分循环而不是继续展开。

所以调 VF 的顺序是：先让循环满足 Hardware Loop 规范，再在寄存器预算内加 `unroll`，最后用是否出现溢出或 ICache Miss 来决定停在哪里。详见[三类计算单元](./compute.md)的性能规则一节。

## 速查

| 你想知道的 | 答案 |
| --- | --- |
| 一个矢量寄存器多宽 | 256 字节；设备侧用 `get_vf_len()` 查，返回 `256` |
| fp32 一次算多少元素 | 64 |
| fp16 / bf16 一次算多少元素 | 128 |
| 掩码寄存器多宽 | 32 字节（VL / 8），每 lane 一位 |
| 有多少个数据寄存器 | 32，超了溢出到 UB 顶部的 6 KB 溢出区 |
| 寄存器能跨 VF 用吗 | 不能，生命周期限于单个 VF |
| 16 位数据怎么按 fp32 算 | `vload_unpack(B16_TO_B32)` + `vstore_pack(B32_TO_B16)` |
| UB 地址要对齐到多少 | 32 字节 |

## 下一步

[三类计算单元](./compute.md)：把 lane 模型落到具体的 Cube、Vector、Scalar 接口上。
