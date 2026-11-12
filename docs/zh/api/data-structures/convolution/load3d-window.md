# Load3DWindow

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

选择 [load3d](../../data-movement/convolution/load3d.md) 展开矩阵的一个 M×K 窗口，起点和大小均针对转置之前的矩阵。与 [Load3DDestination](load3d-destination.md) 不同，本配置描述读哪些展开元素，不描述写到目标存储的什么位置。

## 函数原型

```python
Load3DWindow(
    *, m_extent: int, k_extent: int,
    m_start: int = 0, k_start: int = 0,
) -> Load3DWindow
```

整数参数也可传入核函数中计算得到的整数值，取值范围见参数说明。

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| m_extent | 输入 | 窗口的输出空间位置数，0～65535；0 表示空搬运。8/16 位类型在非末尾 M 块中须为 16 的倍数。 |
| k_extent | 输入 | 窗口的卷积展开元素数，0～65535；0 表示空搬运。未触及完整 K 的最后一个分形块时须为 C0 的倍数。 |
| m_start | 输入 | M 轴起点，0～32767，单位为展开矩阵元素位置，不是输入特征图行号。 |
| k_start | 输入 | K 轴起点，0～65535，须为 C0 的倍数。C0 为一个 32B 通道块包含的元素数。 |

## 返回值说明

返回不可变的 `Load3DWindow` 配置对象。

## 约束说明

窗口的起点和大小按[load3d](../../data-movement/convolution/load3d.md)展开后的矩阵计算。启用转置时，目标Tensor的M、K两维须分别容纳窗口的K、M两维。

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
