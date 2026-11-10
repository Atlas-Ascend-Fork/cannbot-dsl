# Load3DFeatureMap

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

描述 [load3d](../../data-movement/convolution/load3d.md) 的单批源特征图几何及四条边界的补值宽度。它只描述几何，不分配或填充 L1；补值的数值由 [Load3DPadding](load3d-padding.md) 决定。

## 函数原型

```python
Load3DFeatureMap(
    *, height: int, width: int, channels: int,
    pad_top: int = 0, pad_bottom: int = 0,
    pad_left: int = 0, pad_right: int = 0,
) -> Load3DFeatureMap
```

整数参数也接受 DSL 整数标量表达式。

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| height | 输入 | 输入特征图高度，1～32767，单位为行，不含 padding。 |
| width | 输入 | 输入特征图宽度，1～32767，单位为列，不含 padding。 |
| channels | 输入 | 逻辑输入通道数，0～65535；0 表示空搬运。令 C0=32/元素字节数：8 位类型余数允许 0、4、8、16，16 位允许 0、4、8，32 位允许 0、4；余数指 channels%C0。 |
| pad_top | 输入 | 上边界补值行数，0～255。 |
| pad_bottom | 输入 | 下边界补值行数，0～255。 |
| pad_left | 输入 | 左边界补值列数，0～255。 |
| pad_right | 输入 | 右边界补值列数，0～255。 |

## 返回值说明

返回不可变的 `Load3DFeatureMap` 配置对象，作为 `load3d(feature=...)` 的参数。

## 约束说明

声明的几何必须与源数据的实际排布一致；构造配置对象不会搬运数据。与 filter、stride、dilation 的组合要求见 [load3d](../../data-movement/convolution/load3d.md#约束说明)。

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
        feature=cb.Load3DFeatureMap(height=4, width=4, channels=16),
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
    # One C1 plane: (C1, H, W, C0) = (1, 4, 4, 16).
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
