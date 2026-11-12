# Load3DPadding

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

为 [load3d](../../data-movement/convolution/load3d.md) 指定特征图边界外的单元素补值。`value` 按源元素类型解释数值；`bits` 直接提供该元素的存储位模式，不执行数值转换。它与通用 `storage_bits` 包装类型不同，不能相互替代。

## 函数原型

```python
Load3DPadding(payload: int | float, storage_bits: bool) -> Load3DPadding
Load3DPadding.value(value: int | float) -> Load3DPadding
Load3DPadding.bits(bits: int) -> Load3DPadding
```

使用`Load3DPadding.value(0.75)`按数值指定填充值，或使用`Load3DPadding.bits(0x3a00)`按位编码指定同一个`float16`填充值。参数也可使用核函数中计算得到的对应数值或整数编码。

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| payload | 输入 | 构造器保存的数值或单元素位模式，具体解释由 storage_bits 决定。 |
| storage_bits | 输入 | True 按存储位模式解释，False 按数值解释；与同名包装函数不是同一对象。 |
| value | 输入 | 数值补值。普通类型按源元素类型转换；FP8/HIF8 的非零补值使用 bits。数值零可直接使用默认配置。 |
| bits | 输入 | 单个源元素的无符号位模式。源元素位宽为 b 时，要求 $0\le\mathrm{bits}<2^b$。不是把一个字节重复填满所有类型，例如 fp16 的 0.75 为 0x3a00。 |

## 返回值说明

返回不可变的 `Load3DPadding` 配置对象。

## 约束说明

只影响展开中命中边界补值的位置，不修改 L1 原数据，也不清零目标分形块中所有未覆盖元素。

## 调用示例

以下示例展开 L1 特征图，通过单位矩阵乘法读回结果并校验。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
import cannbotdsl as cb
import torch
import torch_npu

@cb.kernel
def kernel(src: cb.Tensor, eye: cb.Tensor, out: cb.Tensor):
    fmap = cb.Buffer(cb.MemLoc.L1, cb.dtypes.float16, cb.make_layout((256,)), data_format="nd")
    identity = cb.Buffer(cb.MemLoc.L1, cb.dtypes.float16, cb.make_layout((16, 16)))
    a = cb.Buffer(cb.MemLoc.L0A, cb.dtypes.float16, cb.make_layout((16, 16)))
    b = cb.Buffer(cb.MemLoc.L0B, cb.dtypes.float16, cb.make_layout((16, 16)))
    c = cb.Buffer(cb.MemLoc.L0C, cb.dtypes.float32, cb.make_layout((16, 16)))
    cb.mem_copy(fmap, src)
    cb.mem_copy(identity, eye)
    cb.load3d(a, fmap,
        feature=cb.Load3DFeatureMap(height=2, width=8, channels=16),
        window=cb.Load3DWindow(m_extent=16, k_extent=16),
        destination=cb.Load3DDestination(k_stride_fractals=1),
        repeat=cb.Load3DRepeat(count=1), padding=cb.Load3DPadding.value(0))
    cb.mem_copy(b, identity)
    cb.matmul(c, a, b, init=True)
    cb.mem_copy(out, c)

@cb.host
def launch(src: cb.Tensor, eye: cb.Tensor, out: cb.Tensor):
    kernel[1](src, eye, out)

def main():
    matrix = torch.arange(256, dtype=torch.float16).reshape(16, 16)
    # One C1 plane: (C1, H, W, C0) = (1, 2, 8, 16).
    src = matrix.flatten().to("npu:0")
    eye = torch.eye(16, dtype=torch.float16, device="npu:0")
    out = torch.empty((16, 16), dtype=torch.float32, device="npu:0")
    launch(src, eye, out)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), matrix.float(), rtol=0, atol=0)
    print("load3d example passed")

if __name__ == "__main__":
    main()
```

预期结果：校验通过，输出 `load3d example passed`。
