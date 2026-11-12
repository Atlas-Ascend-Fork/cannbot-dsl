# conv2d_store_output

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

将 `matmul` 在 L0C 中生成的当前卷积结果块搬到 GM，按 NCHW 顺序写入完整输出 Tensor 中的对应位置。搬运时完成 NZ2DN 排布转换，并将 `float32` 结果转换为目标的 `float16` 或 `bfloat16` 类型。

L0C 结果矩阵的行对应输出空间位置（将输出高、宽按行展开），列对应输出通道。写回 GM 后，同一输出通道的空间数据按高、宽顺序存放，再排列下一个输出通道，如下图展示排布转换关系。。

![通过 NZ2DN 将卷积结果转换为 NCHW 输出](../../figures/conv2d_output_nc1hwc0_nchw.png)

图中 C 表示输出通道，H、W 表示输出特征图的高、宽；左侧为 L0C 中按通道分块的结果，右侧为 NCHW 输出。

## 函数原型

```python
conv2d_store_output(dst: Tensor, src: Tensor, tile: Conv2dTile, *, l2_cache_ctl: int = 0) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| dst | 输出 | 目标 GM 输出 `Tensor`。<br>必填，无默认值。类型为位于 GM 的 `Tensor`，按 NCHW 排布，形状为 `tile.spec.output_shape`；元素类型支持 `float16`、`bfloat16`。 |
| src | 输入 | 待写回的 L0C 结果块。<br>必填，无默认值。类型为位于 L0C 的 `Tensor`，元素类型为 `float32`；使用 `tile.l0c_view(...)` 获取当前块的有效视图。 |
| tile | 输入 | 根据 `batch`、`group`、输出位置块编号 `coord[0]` 和输出通道块编号 `coord[1]`，确定结果在完整输出中的位置及有效范围。<br>必填，无默认值。类型为 [Conv2dTile](../../data-structures/convolution/conv2d-tile.md)，由 [spec.tile](../../data-structures/convolution/make-conv2d-spec/tile.md) 创建，须与源结果块和目标输出对应。 |
| l2_cache_ctl | 输入 | L2 搬出缓存策略。<br>默认 `0`。支持编译期整数 `0`、`1`、`2`、`4`。 |

## 返回值说明

无返回值，结果写入 dst。

## 约束说明

特征图加载、权重加载和结果写回须使用相同的卷积配置，并对应同一次分块计算。输入和权重须在加载完成后参与计算；沿输入通道分块计算时，只在第一块初始化累加结果，全部分块累加完成后再写回。

## 调用示例

下例完成一个 1×1 卷积计算块。输入高、宽为 2、8，输入、输出通道数分别为 16、32。矩阵乘完成后，`conv2d_store_output` 将 L0C 结果写回 NCHW 输出。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
import cannbotdsl as cb
import torch
import torch_npu


@cb.kernel
def conv2d(x: cb.Tensor, weight: cb.Tensor, output: cb.Tensor):
    dtype = cb.dtypes.float16
    spec = cb.make_conv2d_spec(
        dtype=dtype, input_shape=(1, 16, 2, 8), filter_shape=(32, 16, 1, 1)
    )
    tile = spec.tile(tile=(16, 32, 16), coord=(0, 0, 0))
    fmap = cb.make_conv2d_fmap_l1(dtype, spec, tile=(16, 16))
    weights = cb.Buffer(cb.MemLoc.L1, dtype, cb.make_layout((32, 16))).permute(1, 0)
    a = cb.Buffer(cb.MemLoc.L0A, dtype, cb.make_layout((16, 16)))
    b = cb.Buffer(cb.MemLoc.L0B, dtype, cb.make_layout((32, 16))).permute(1, 0)
    c = cb.Buffer(cb.MemLoc.L0C, cb.dtypes.float32, cb.make_layout((16, 32)))

    cb.conv2d_load_fmap(fmap, x, tile)
    cb.conv2d_load_filter(weights, weight, tile)
    a_view = tile.l0a_view(a)
    b_view = tile.l0b_view(b)
    c_view = tile.l0c_view(c)
    cb.conv2d_load_im2col(a_view, fmap, tile)
    cb.mem_copy(b_view, tile.l1b_view(weights))
    cb.matmul(c_view, a_view, b_view, init=True)
    cb.conv2d_store_output(output, c_view, tile)


@cb.host
def run(x: cb.Tensor, weight: cb.Tensor, output: cb.Tensor):
    conv2d[1](x, weight, output)


if __name__ == "__main__":
    x = ((torch.arange(256).reshape(1, 16, 2, 8) % 13 - 6) / 8).half()
    w = ((torch.arange(512).reshape(32, 16, 1, 1) % 7 - 3) / 8).half()
    y = torch.empty((1, 32, 2, 8), dtype=torch.float16, device="npu")
    run(x.npu(), w.npu(), y)
    torch.npu.synchronize()
    expected = torch.nn.functional.conv2d(x.float(), w.float()).half()
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)
    print("Conv2dTile example passed")
```

预期结果为形状 `(1, 32, 2, 8)` 的输出，与 CPU 参考结果逐元素一致，并打印 `Conv2dTile example passed`。
