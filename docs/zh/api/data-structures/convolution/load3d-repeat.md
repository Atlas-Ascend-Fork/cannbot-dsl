# Load3DRepeat

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

控制 [load3d](../../data-movement/convolution/load3d.md) 的重复展开和源窗口推进。`stride` 控制源窗口间隔；目标结果按所选方向连续拼接，不等同于在目标中留下同样大小的空洞。

## 函数原型

```python
Load3DRepeat(
    *, count: int = 1, stride: int = 0, direction: str | int = "m",
) -> Load3DRepeat
```

整数参数也可传入核函数中计算得到的整数值，取值范围见参数说明。

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| count | 输入 | 重复次数，0～255；0 不访问数据，默认 1。 |
| stride | 输入 | 相邻重复的源窗口起点差，0～65535。M 方向以 16 个 M 元素为单位；K 方向以 C0 个 K 元素为单位。默认 0，重复读取相同源窗口。 |
| direction | 输入 | `"m"`/`0` 沿展开矩阵 M 方向推进；`"k"`/`1` 沿 K 方向推进。 |

## 返回值说明

返回不可变的 `Load3DRepeat` 配置对象。

## 约束说明

多次展开应同时传入 [Load3DDestination](load3d-destination.md)，并为所有重复结果预留容量；自动目标推导只适用于单次展开。

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
