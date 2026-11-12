# storage_bits

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

包装单元素存储编码，不执行数值转换。可用于 [convert](convert.md) 的 relu_limit，特别是必须显式给出位模式的 FP8/HIF8 截断阈值；它不是 [Load3DPadding.bits](../convolution/load3d-padding.md)，后者只用于 load3d 的边界补值。

## 函数原型

```python
storage_bits(value: int) -> StorageBits
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| value | 输入 | 单元素位模式，Python 整数或 DSL int64 标量。用作 relu_limit 时，8 位输出要求 0～255，16 位输出要求 0～65535，且编码必须表示目标类型中的有限正数。fp16 的 6.0 编码为 0x4600。 |

## 返回值说明

返回 StorageBits 包装对象；其 value 成员保存所传编码，不执行存储读写。

## 约束说明

编码须与接收该参数的接口所要求的数据类型一致。例如，为`float16`输出指定6.0的截断阈值时，使用`storage_bits(0x4600)`。返回对象不能作为普通数值参与算术运算。

## 调用示例

以下示例的 clip_bits 模式与数值阈值 6.0 做对照，完成实际矩阵乘法、截断和结果校验。

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
