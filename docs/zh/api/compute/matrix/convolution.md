# 使用矩阵乘实现卷积

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

使用已有原语完成单批 3×3 卷积：将输入和权重搬到 L1，使用 [load3d](../../data-movement/convolution/load3d.md) 展开输入，使用 `matmul` 计算输出，再由 [mem_copy](../../data-movement/mem-copy.md) 写回。卷积的 bias、量化或激活可在相应计算及输出阶段扩展；本例只计算卷积，避免过多辅助依赖。

权重按 `(输出通道, 核高, 核宽, 输入通道)` 排列。示例的输入通道数为 16，恰好只有一个 C1 块，因此输入 NHWC 数据可直接作为单批 NC1HWC0 的载荷；多通道块时须先正确组织特征图，不能直接照搬该简化。

## 函数原型

无。本页展示原语组合，不向框架新增 `conv2d` 公共接口；示例中的同名函数是用户定义 kernel。

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| x | 输入 | 示例 Host 输入，连续 fp16[256]，由 NHWC[1,4,4,16] 展平。 |
| weight | 输入 | 示例 Host 权重，连续 fp16[16,144]，由 OHWI[16,3,3,16] 展平核位置和输入通道。 |
| y | 输出 | 示例 Host 输出，float32[16,16]；两轴分别是展开的输出空间位置和输出通道，完成后可按 NHWC[1,4,4,16] 读取。 |

## 返回值说明

示例 Host 不返回 Tensor，卷积结果写入 y。

## 约束说明

该最小示例仅演示固定尺寸、单批和单通道块的直接卷积。不把它当作动态尺寸或任意卷积调度；分组、多批和大形状需要分块及显式调度，可参考配套提交的 `examples/conv` 和本目录的接口说明。

## 调用示例

以下完整代码独立执行卷积，并与 PyTorch CPU conv2d 结果逐元素比较。

<!-- example: conv2d_example.py -->

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
import torch
import torch.nn.functional as F
import torch_npu
import cannbotdsl as cbd
from cannbotdsl import MemLoc, dtypes, make_layout

@cbd.kernel
def conv2d(x: cbd.Tensor, weight: cbd.Tensor, y: cbd.Tensor):
    # One NHWC batch, H=W=4, C=16; for C1=1 its payload is NC1HWC0.
    feature = cbd.Buffer(MemLoc.L1, dtypes.float16, make_layout((256,)), data_format="nd")
    weights = cbd.Buffer(MemLoc.L1, dtypes.float16, make_layout((16, 144)))
    a = cbd.Buffer(MemLoc.L0A, dtypes.float16, make_layout((16, 144)))
    b = cbd.Buffer(MemLoc.L0B, dtypes.float16, make_layout((16, 144)))
    c = cbd.Buffer(MemLoc.L0C, dtypes.float32, make_layout((16, 16)))
    cbd.mem_copy(feature, x)
    cbd.mem_copy(weights, weight)
    cbd.mem_copy(b, weights)
    cbd.load3d(a, feature,
        feature=cbd.Load3DFeatureMap(height=4, width=4, channels=16,
            pad_top=1, pad_bottom=1, pad_left=1, pad_right=1),
        window=cbd.Load3DWindow(m_extent=16, k_extent=144), filter=(3, 3))
    cbd.matmul(c, a, b)
    cbd.mem_copy(y, c)

@cbd.host
def run(x: cbd.Tensor, weight: cbd.Tensor, y: cbd.Tensor):
    conv2d[1](x, weight, y)

def main():
    assert torch.npu.is_available(), "NPU is required"
    x = ((torch.arange(256).reshape(1, 4, 4, 16) % 13 - 6).float() / 8).half()
    w = ((torch.arange(16 * 3 * 3 * 16).reshape(16, 3, 3, 16) % 11 - 5).float() / 8).half()
    expected = F.conv2d(x.permute(0, 3, 1, 2).float(), w.permute(0, 3, 1, 2).float(), padding=1)
    y = torch.empty((16, 16), dtype=torch.float32, device="npu")
    run(x.flatten().npu(), w.reshape(16, 144).npu(), y)
    torch.npu.synchronize()
    torch.testing.assert_close(y.cpu().reshape(1, 4, 4, 16), expected.permute(0, 2, 3, 1), rtol=0, atol=0)
    print("conv2d example passed")

if __name__ == "__main__":
    main()
```
<!-- end-example -->

预期结果：输出 `conv2d example passed`。
