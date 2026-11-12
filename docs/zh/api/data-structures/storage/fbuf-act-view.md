# FBUF_ACT.view

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

返回负分支参数存储区的完整字节视图，用于手动放置参数表。先选取字节区间并绑定 FP32 类型，再用 `make_buffer` 绑定后供搬运和 [convert](../tensor/convert.md) 使用；只想让框架自动安排地址时，直接使用 `Buffer(MemLoc.FBUF_ACT, ...)`。

本入口与 FBUF 不同，保存的是 FP32 负分支系数，而不是编码量化表。参数格式及长度要求见 [Tensor.convert](../tensor/convert.md#参数说明)。

## 函数原型

```python
FBUF_ACT.view(*shape: int, dtype: DType | None = None) -> Tensor
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| shape | 输入 | 覆盖完整空间的静态形状，元素数乘积必须为 2048；常用 `view(2048)`。也可传入一个形状元组或列表。单位为字节，不是 FP32 元素数。 |
| dtype | 输入 | 仅允许 None 或 dtypes.uint8；绑定实际 FP32 表应在切出字节区间后调用 Tensor.view(dtype=...)。 |

## 返回值说明

返回 uint8 Tensor 空间视图，不执行读写，也不清零数据。同一 kernel 中重复访问此空间共享底层存储，不是重复分配。

## 约束说明

- 在同一 kernel 的 FBUF_ACT 空间内，不能混用手动空间视图与自动 Buffer 分配；其他存储空间可以独立选择方式。
- 手动放置的参数表须位于FBUF_ACT空间范围内，起点按64字节对齐。先通过`make_buffer`绑定，再搬入参数数据；初始化完成后，才能将其传给`convert`的`negative_scale`参数。
- 此参数空间不支持 RingBuffer，也不能通过不同别名规避表的读写依赖。

## 调用示例

以下完整示例的 space 模式把 FP32[32] 放在 FBUF_ACT 的字节区间 [64,192)，其他模式给出自动分配和标量对照。

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
