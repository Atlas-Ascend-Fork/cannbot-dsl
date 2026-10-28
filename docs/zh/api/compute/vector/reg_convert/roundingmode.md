# RoundingMode
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 简介

`RoundingMode`是寄存器级类型转换接口的舍入模式枚举，用于`vcast`的`rounding`参数，决定精度变化时如何选择结果值。它与标量转换使用的`scalar.RoundingMode`相互独立；饱和模式由单独的`saturate`参数控制。

## 定义

```python
from cannbotdsl.ops.reg import RoundingMode

RoundingMode.NA   # 不显式指定舍入模式，由转换接口按类型组合选择
RoundingMode.RN   # 舍入到最近的可表示值，中点取偶数
RoundingMode.RNA  # 舍入到最近的可表示值，中点远离零
RoundingMode.RD   # 向负无穷方向舍入
RoundingMode.RU   # 向正无穷方向舍入
RoundingMode.RZ   # 向零方向舍入
RoundingMode.RO   # 奇数舍入，用于 float32 转 float16
```

## 模式概述

类型转换接口通过以下两种独立设置处理舍入和溢出：

- **舍入模式**：主要用于浮点数之间的转换或浮点数转整数时，确定如何将高精度数值舍入到低精度表示。
- **饱和模式/非饱和模式**：处理数值超出目标数据类型表示范围时的溢出行为。

## 舍入模式说明

`RoundingMode`枚举支持以下舍入模式：

**表1** 舍入模式说明

| 舍入模式 | 枚举成员 | 说明 | 示例 |
| :------ | :------ | :--- | :--- |
| RINT | `RoundingMode.RN` | 四舍六入五成双舍入（Banker's Rounding）。当小数部分为0.5时，舍入到最近的偶数。 | 1.5 → 2, 2.5 → 2, 3.5 → 4 |
| ROUND | `RoundingMode.RNA` | 舍入到最近的可表示值；恰好位于中点时选择远离零的一侧。 | 1.4 → 1, 1.5 → 2, -1.5 → -2 |
| FLOOR | `RoundingMode.RD` | 向负无穷方向舍入。总是向下取整。 | 1.7 → 1, -1.3 → -2 |
| CEIL | `RoundingMode.RU` | 向正无穷方向舍入。总是向上取整。 | 1.3 → 2, -1.7 → -1 |
| TRUNC | `RoundingMode.RZ` | 向零方向舍入。直接截断小数部分。 | 1.7 → 1, -1.7 → -1 |
| ODD | `RoundingMode.RO` | 奇数舍入，仅用于`float32`转`float16`，控制浮点表示的舍入，不表示取整到最近的奇数。 | 不适用整数取整示例 |

`RoundingMode.NA`为默认值，表示未显式指定舍入模式。

`vcast`的`rounding`参数也接受对应的字符串`"na"`、`"rn"`、`"rna"`、`"rd"`、`"ru"`、`"rz"`、`"ro"`，不区分大小写。可用模式与源、目标类型有关，具体子集见 [`vcast`](./vcast.md#约束说明)。

### 舍入模式详细说明

#### RINT（四舍六入五成双）
也称为"银行家舍入法"，是一种统计学上最无偏的舍入方法。
- 当小数部分小于0.5时，向下舍入
- 当小数部分大于0.5时，向上舍入
- 当小数部分等于0.5时，舍入到最近的偶数

**示例：**
```text
1.4 → 1  (小数部分 < 0.5，向下)
1.5 → 2  (小数部分 = 0.5，舍入到偶数)
2.5 → 2  (小数部分 = 0.5，舍入到偶数)
3.5 → 4  (小数部分 = 0.5，舍入到偶数)
-1.5 → -2 (小数部分 = 0.5，舍入到偶数)
```

#### ROUND（四舍五入）
舍入到最近的可表示值；恰好位于两个候选值的中点时，选择绝对值更大的一侧。下例为整数精度下的舍入说明：

**示例：**
```text
1.4 → 1  (小数部分 < 0.5)
1.5 → 2  (小数部分 ≥ 0.5)
1.6 → 2  (小数部分 ≥ 0.5)
-1.5 → -2 (向绝对值更大的方向舍入)
```

#### FLOOR（向下取整）
始终向负无穷方向舍入，即使结果变小。

**示例：**
```text
1.7 → 1   (向下)
-1.3 → -2 (向负无穷方向，变得更小)
```

#### CEIL（向上取整）
始终向正无穷方向舍入，即使结果变大。

**示例：**
```text
1.3 → 2   (向上)
-1.7 → -1 (向正无穷方向，变得更大)
```

#### TRUNC（向零截断）
直接截断小数部分，向零方向舍入。

**示例：**
```text
1.7 → 1   (向零方向)
-1.7 → -1 (向零方向)
```

#### ODD（奇数舍入）
用于`float32`转`float16`的浮点精度转换，不等价于舍入到最近的奇数整数。不能使用整数取整示例解释本模式。

## 饱和模式与非饱和模式说明

当数据类型转换时，源数据的值可能超出目标数据类型的表示范围，此时需要通过饱和模式或非饱和模式来处理溢出。

### 配置方法

饱和模式与非饱和模式通过转换接口的`saturate`参数配置，详细说明请参见[`vcast`](./vcast.md)。

### 非饱和模式（Non-Saturation Mode）

在非饱和模式下，当数据超出目标类型范围时：

#### 整型转换
- **结果被截断**：保留最低有效位（LSB），高位直接丢弃
- **二进制截断示例**：将int32转换为int8时，只保留低8位

**二进制截断示例（int32 → int8）：**
```text
输入值（int32）:  0x000001FF (511)
二进制表示:       0000 0000 0000 0000 0000 0001 1111 1111
截断后（int8）:   1111 1111 (0xFF, -1)
说明：只保留低8位，高位丢弃

输入值（int32）:  0x00000100 (256)
二进制表示:       0000 0000 0000 0000 0000 0001 0000 0000
截断后（int8）:   0000 0000 (0x00, 0)
说明：只保留低8位，高位丢弃

输入值（int32）:  0xFFFFFF00 (-256)
二进制表示:       1111 1111 1111 1111 1111 1111 0000 0000
截断后（int8）:   0000 0000 (0x00, 0)
说明：只保留低8位，高位丢弃
```

#### 浮点数转换
- **超出范围**：返回输出类型的对应符号inf值（正/负无穷）
- **±inf输入**：返回±inf
- **NaN输入**：返回NaN

**示例（float → half）：**
```text
输入值: 70000.0 (超出half范围)
输出值: +inf

输入值: -70000.0
输出值: -inf

输入值: NaN
输出值: NaN
```

#### 浮点转整型
- **超出范围**：结果被截断至目标格式的数据宽度
- **±inf输入**：返回输出类型的最值
- **NaN输入**：返回0

**示例（float → int16）：**
```text
输入值: 100000.5 (超出int16范围)
非饱和模式: 结果被截断（具体值取决于二进制表示）

输入值: +inf
输出值: 32767 (int16最大值)

输入值: NaN
输出值: 0
```

### 饱和模式（Saturation Mode）

在饱和模式下，当数据超出目标类型范围时，结果被限制在目标类型的可表示范围内：

#### 整型转换
- **超出范围**：返回目标类型的最大值或最小值
- **保持数值安全**：不会出现因截断导致的意外结果

**示例（int32 → int8）：**
```text
输入值: 511 (超出int8范围)
饱和输出: 127 (int8最大值)

输入值: -300 (超出int8范围)
饱和输出: -128 (int8最小值)

输入值: 100 (在int8范围内)
饱和输出: 100 (保持不变)
```

**示例（int32 → uint8）：**
```text
输入值: 500 (超出uint8范围)
饱和输出: 255 (uint8最大值)

输入值: -10 (超出uint8范围)
饱和输出: 0 (uint8最小值)

输入值: 100 (在uint8范围内)
饱和输出: 100 (保持不变)
```

#### 浮点数转换
- **超出范围**：返回输出类型的对应最值
- **±inf输入**：返回输出类型的对应最值
- **NaN输入**：返回0

**示例（float → half）：**
```text
输入值: 70000.0 (超出half范围)
饱和输出: 65504.0 (half最大值)

输入值: -70000.0
饱和输出: -65504.0 (half最小值)

输入值: +inf
饱和输出: 65504.0 (half最大值)

输入值: NaN
饱和输出: 0
```

#### 浮点转整型
- **超出范围**：返回输出类型的对应最值
- **±inf输入**：返回输出类型的对应最值
- **NaN输入**：返回0

**示例（float → int16）：**
```text
输入值: 100000.5 (超出int16范围)
饱和输出: 32767 (int16最大值)

输入值: -50000.0
饱和输出: -32768 (int16最小值)

输入值: +inf
饱和输出: 32767

输入值: NaN
饱和输出: 0
```

### 模式对比总结

**表2** 饱和模式与非饱和模式说明

| 场景 | 非饱和模式 | 饱和模式 |
| :--- | :--- | :--- |
| **整型转整型（溢出）** | 截断低位，高位丢弃 | 返回目标类型最值 |
| **浮点转浮点（超出范围）** | 返回±inf | 返回目标类型最值 |
| **浮点转整型（超出范围）** | 截断至目标宽度 | 返回目标类型最值 |
| **浮点转整型（±inf）** | 返回目标类型最值 | 返回目标类型最值 |
| **浮点转浮点（±inf）** | 返回±inf | 返回目标类型最值 |
| **NaN输入** | 返回NaN或0 | 返回0 |

## 约束说明

- 本枚举无独立执行语义；约束见其使用接口。

## 调用示例

以下示例使用`RoundingMode.RN`将寄存器中的浮点数转换为`float16`，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, dtypes, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import PackMode, RoundingMode, create_mask, vcast, vload, vstore_pack
from cannbotdsl.tensor import MemLoc

@kernel
def _rounding_mode_kernel(src, dst):
    buf = Buffer(MemLoc.UB, src.dtype, make_layout((64,)))
    out = Buffer(MemLoc.UB, dst.dtype, make_layout((64,)))

    mem_copy(buf, src)

    in0 = buf
    res = out
    with vf(mode="simd"):
        mask = create_mask(pattern="all", elem_bits=32)
        h = vcast(vload(in0, 0), dtypes.float16, mask=mask,
                         rounding=RoundingMode.RN)
        vstore_pack(res, 0, h, mask, pack_mode=PackMode.B32_TO_B16)

    mem_copy(dst, out)

@host
def run(src, dst):
    _rounding_mode_kernel[1](src, dst)

def main():
    src = (torch.arange(64, dtype=torch.float32, device="npu:0") - 32.0) / 8.0
    dst = torch.empty((64,), dtype=torch.float16, device="npu:0")

    run(src, dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), src.cpu().half())
    print("RoundingMode example passed")
    print(f"first={float(dst.cpu()[0]):.4f}, last={float(dst.cpu()[-1]):.4f}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
RoundingMode example passed
first=-4.0000, last=3.8750
```
