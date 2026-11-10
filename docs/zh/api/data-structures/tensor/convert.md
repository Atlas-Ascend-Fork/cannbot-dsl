# Tensor.convert

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

描述 L0C 结果搬出时的类型转换、缩放和可选激活，交给 `mem_copy(dst, src.convert(...))` 执行。`convert` 本身不搬运、不分配结果存储；同一 L0C 可分别构造不同表达式写到多个输出，但不是一次调用同时写两个目标。

支持 `offset`、`negative_scale`、`relu_limit`。`negative_scale` 按原累加值是否为负选择独立系数，并非先乘 scale 再乘一个负半轴斜率；`relu_limit` 则先对原累加值做 ReLU，转换后再上限截断。与 `src.relu()` 不同，它还包含输出阈值。

令 x 为 `src` 的累加元素、s 为 scale、a 为 negative_scale，负分支模式在输出转换前计算：

$$z=\begin{cases}ax,&x<0,\\sx,&x\ge0.\end{cases}$$

标量 INT8/UINT8 量化中，offset 加在缩放结果舍入并限制到有符号 9 位之后，再按输出类型饱和：

$$y=\operatorname{sat}_{\mathrm{dtype}}\!\left(\operatorname{clip}_{[-256,255]}(\operatorname{round}_{\mathrm{even}}(sx))+\mathrm{offset}\right).$$

`relu_limit` 模式先取 `max(x,0)`，再按缩放、量化规则计算输出，最后取与阈值的较小值。scale 为负时输出仍可能为负，不应把它理解成对最终输出做 `clamp(0, limit)`。

## 函数原型

```python
Tensor.convert(
    dtype: DType, *,
    scale: float | Tensor | None = None,
    offset: int = 0,
    negative_scale: float | Tensor | None = None,
    relu_limit: float | StorageBits | None = None,
    round_mode: str | None = None,
) -> Tensor
```

标量 scale/negative_scale 接受 Python 数值或 DSL float32 标量；offset 接受 Python 整数或 DSL int64 标量。数值 relu_limit 接受 Python 数值或 DSL float32 标量。

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| self | 输入 | L0C 累加结果 Tensor，常用 float32 或 int32。表达式在后续 mem_copy 执行时读取源值及参数表，而非构造表达式时保存数据快照。 |
| dtype | 输入 | 目的元素类型，必须与 mem_copy 目标一致。offset 限定为 int8/uint8；relu_limit 支持 float16、bfloat16、int8、uint8、float8_e4m3fn、hifloat8，不支持 float32。具体类型组合还须满足累加类型和搬运目标的转换能力。 |
| scale | 输入 | 标量缩放系数，或逐输出通道的编码量化表；表格式见[量化参数表格式](#量化参数表格式)。不启用量化控制时省略表示普通类型转换；使用激活或偏移控制而省略 scale 时默认系数为 1.0。 |
| offset | 输入 | 标量 INT8/UINT8 输出偏移，范围 -256～255，默认 0。与量化表一起使用时不得另传非零或动态偏移，表内已含偏移。 |
| negative_scale | 输入 | 对原始值小于 0 的 L0C 结果使用的独立缩放系数；可传入标量，对所有负值使用同一系数，也可传入逐输出通道参数表，分别缩放各通道的负值。该能力不限定卷积场景。<br>默认 `None`，不单独处理负值。支持数值或位于 L1/FBUF_ACT 的 `Tensor`；参数表须为连续 `float32[N]`，每项对应一个输出通道，N 须覆盖输出列数且为 1～512。L1 表自动暂存时长度须为 64B 的整数倍、源起点按 32B 对齐；FBUF_ACT 表起点须按 64B 对齐。表可从 GM 经 L1 准备，或用 `mem_copy` 显式搬入 FBUF_ACT 后复用；其分配及手动访问方式见 [FBUF_ACT.view](../storage/fbuf-act-view.md)。 |
| relu_limit | 输入 | 输出上限，必须在目的类型中为正且有限；整数输出阈值必须为整数且不超过类型最大值。FP8/HIF8 必须用 [storage_bits](storage-bits.md) 提供有效单元素编码；FP16/BF16/INT8/UINT8 也可使用编码。 |

| round_mode | 输入 | 默认 None，保持默认舍入；`"hybrid"` 用于带 scale 的 float32→hifloat8 GM 输出。 |

### 量化参数表格式

`scale` 传入 Tensor 时，使用逐输出通道的编码量化表，每个通道对应一个 64 位编码项，通常以 `int32[2*N]` 存放，N 为表项数。它不是直接存放缩放系数的 `float32[N]` 数组。

| 编码字段 | 含义 |
| --- | --- |
| 低 32 位 | FP32 缩放系数的位编码。 |
| 第 37～45 位 | INT8 输出偏移，采用有符号 9 位编码。 |
| 第 46 位 | 选择有符号输出。 |

浮点输出示例可将每项的高 32 位清零。表从 GM 搬到 L1 后可作为 `scale` 传入，也可先通过 `mem_copy` 显式搬入 FBUF 后复用；FBUF 目标起点须按 128B 对齐。使用前须完成参数表初始化，并保持有效直至搬运完成。

此编码表与 `negative_scale` 使用的连续 FP32 系数表不同，两者及其存储位置 FBUF、FBUF_ACT 不能互换。

## 返回值说明

返回转换结果的 Tensor 表达式；由 `mem_copy` 消费后才产生目标数据。未被搬运消费的表达式不执行转换。

## 约束说明

- `negative_scale` 与 `relu_limit` 互斥，不能在同一表达式中同时启用。
- 保持 L0C、量化表、负分支表有效，直到消费表达式的搬运完成；重复输出可以共享 L0C，但各输出必须绑定各自参数。
- 不把本接口用于任意 GM/UB 上的逐元素转换；本页描述的是 L0C 结果搬出路径。

## 调用示例

以下完整示例通过矩阵乘法生成正负累加值，分别验证 INT8 偏移、独立负分支系数、FBUF_ACT 表和两种阈值表达方式。

<!-- example: convert_example.py -->

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
import torch
import torch_npu
import cannbotdsl as cbd
from cannbotdsl import MemLoc, dtypes, make_layout

class Output:
    def __init__(self, mode):
        self.mode = mode

    @cbd.kernel
    def compute(self, x: cbd.Tensor, w: cbd.Tensor, alpha: cbd.Tensor, y: cbd.Tensor):
        xl = cbd.Buffer(MemLoc.L1, dtypes.float16, make_layout((16, 32)))
        wl = cbd.Buffer(MemLoc.L1, dtypes.float16, make_layout((32, 32)))
        a = cbd.Buffer(MemLoc.L0A, dtypes.float16, make_layout((16, 32)))
        b = cbd.Buffer(MemLoc.L0B, dtypes.float16, make_layout((32, 32)))
        c = cbd.Buffer(MemLoc.L0C, dtypes.float32, make_layout((16, 32)))
        cbd.mem_copy(xl, x)
        cbd.mem_copy(wl, w)
        cbd.mem_copy(a, xl)
        cbd.mem_copy(b, wl)
        cbd.matmul(c, a, b)
        if self.mode == "offset":
            cbd.mem_copy(y, c.convert( dtypes.int8, scale=0.5, offset=-3))
        elif self.mode == "negative":
            cbd.mem_copy(y, c.convert( dtypes.float16, scale=0.5, negative_scale=0.25))
        elif self.mode == "table" or self.mode == "space":
            al = cbd.Buffer(MemLoc.L1, dtypes.float32, make_layout((32,)), data_format="nd")
            af = (cbd.Buffer(MemLoc.FBUF_ACT, dtypes.float32, make_layout((32,)), data_format="nd")
                  if self.mode == "table" else
                  cbd.make_buffer(cbd.FBUF_ACT.view(2048)[64:192,].view(dtype=dtypes.float32).view(32)))
            cbd.mem_copy(al, alpha)
            cbd.mem_copy(af, al)
            cbd.mem_copy(y, c.convert( dtypes.float16, scale=0.5, negative_scale=af))
        elif self.mode == "clip":
            cbd.mem_copy(y, c.convert( dtypes.float16, scale=0.5, relu_limit=6.0))
        else:
            cbd.mem_copy(y, c.convert( dtypes.float16, scale=0.5, relu_limit=cbd.storage_bits(0x4600)))

    @cbd.host
    def run(self, x: cbd.Tensor, w: cbd.Tensor, alpha: cbd.Tensor, y: cbd.Tensor):
        self.compute[1](x, w, alpha, y)

def main():
    assert torch.npu.is_available(), "NPU is required"
    x = (torch.arange(512).reshape(16, 32) % 19 - 9).half()
    w = (torch.arange(1024).reshape(32, 32) % 17 - 8).half()
    alpha = 2.0 ** -(torch.arange(32) % 4 + 1).float()
    product = x.float() @ w.float().T
    for mode in ("offset", "negative", "table", "space", "clip", "clip_bits"):
        dtype = torch.int8 if mode == "offset" else torch.float16
        y = torch.empty((16, 32), dtype=dtype, device="npu")
        Output(mode).run(x.npu(), w.npu(), alpha.npu(), y)
        torch.npu.synchronize()
        if mode == "offset":
            ref = ((product * 0.5).round().clamp(-256, 255) - 3).clamp(-128, 127)
        elif mode in ("negative", "table", "space"):
            ref = torch.where(product < 0, product * (0.25 if mode == "negative" else alpha), product * 0.5)
        else:
            ref = (product.clamp_min(0) * 0.5).clamp_max(6)
        torch.testing.assert_close(y.cpu(), ref.to(dtype), rtol=0, atol=0)
    print("convert example passed (6 scenarios)")

if __name__ == "__main__":
    main()
```
<!-- end-example -->

预期结果：输出 `convert example passed (6 scenarios)`。
