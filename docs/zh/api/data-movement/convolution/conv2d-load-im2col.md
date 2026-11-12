# conv2d_load_im2col

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

按卷积规则从 L1 特征图中提取每个输出位置所需的输入窗口，将窗口内的数据展开为矩阵的一行，并把当前计算块搬入 L0A，供 `matmul` 作为左矩阵使用。本接口只提取、重排数据，不进行乘加计算。

窗口按 `spec` 中的步长移动，按膨胀率取点，超出输入边界的位置补零。相邻窗口重叠时，同一输入值会出现在多行中。窗口展开原理及配图参见 [load3d 功能说明](load3d.md#功能说明)。展开矩阵描述的是逻辑关系，L0A 中实际采用矩阵计算所需的分形排布。

## 函数原型

```python
conv2d_load_im2col(dst: Tensor, src: Tensor, tile: Conv2dTile) -> None
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| dst | 输出 | 目标 L0A `Tensor`。<br>必填，无默认值。底层缓冲区按 `(BM, Kh×Kw×BC)` 预留容量，元素类型须与 `tile.spec.dtype` 相同，支持 `float16`、`bfloat16`。 |
| src | 输入 | 源 L1 `Tensor`，表示输入特征图区域。<br>必填，无默认值。类型为位于 L1 的 `Tensor`，数据按 NC1HWC0 排布，须与当前 `tile` 对应；元素类型须与 `tile.spec.dtype` 相同。 |
| tile | 输入 | 确定当前输出位置块、输入通道块及其所需输入窗口，并提供步长、填充和膨胀率等卷积配置。<br>必填，无默认值。类型为 [Conv2dTile](../../data-structures/convolution/conv2d-tile.md)，由 [spec.tile](../../data-structures/convolution/make-conv2d-spec/tile.md) 创建，须与源特征图加载时的切块信息及目标视图匹配。 |

## 返回值说明

无返回值，结果写入 dst。

## 约束说明

特征图加载、权重加载和结果写回须使用相同的卷积配置，并对应同一次分块计算。输入和权重须在加载完成后参与计算；沿输入通道分块计算时，只在第一块初始化累加结果，全部分块累加完成后再写回。

## 调用示例

将 `(1, 16, 4, 4)` 特征图按 3×3 窗口、步长 1、四周补零 1 展开到 L0A，得到 16 个输出位置、每个位置 144 个元素。为校验结果，将展开矩阵乘单位矩阵后写回 GM，不执行完整卷积。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
import torch
import torch_npu
import cannbotdsl as cb


@cb.kernel
def expand(x: cb.Tensor, identity: cb.Tensor, y: cb.Tensor):
    spec = cb.make_conv2d_spec(
        dtype=cb.dtypes.float16,
        input_shape=(1, 16, 4, 4),
        filter_shape=(16, 16, 3, 3),
        padding=(1, 1, 1, 1),
    )
    tile = spec.tile(tile=(16, 16, 16), coord=(0, 0, 0))
    fmap = cb.make_conv2d_fmap_l1(cb.dtypes.float16, spec, tile=(16, 16))
    a = cb.Buffer(cb.MemLoc.L0A, cb.dtypes.float16, cb.make_layout((16, 144)))
    cb.conv2d_load_fmap(fmap, x, tile)
    cb.conv2d_load_im2col(tile.l0a_view(a), fmap, tile)

    # L0A 无直接写回 GM 的通路，乘单位矩阵以读回展开结果。
    b_l1 = cb.Buffer(cb.MemLoc.L1, cb.dtypes.float16, cb.make_layout((144, 144)))
    b = cb.Buffer(cb.MemLoc.L0B, cb.dtypes.float16, cb.make_layout((144, 144)))
    c = cb.Buffer(cb.MemLoc.L0C, cb.dtypes.float32, cb.make_layout((16, 144)))
    cb.mem_copy(b_l1, identity)
    cb.mem_copy(b, b_l1)
    cb.matmul(c, a, b, init=True)
    cb.mem_copy(y, c)


@cb.host
def run(x: cb.Tensor, identity: cb.Tensor, y: cb.Tensor):
    expand[1](x, identity, y)


def main():
    x = torch.arange(256).reshape(1, 16, 4, 4).remainder(100).half()
    y = torch.empty((16, 144), dtype=torch.float32, device="npu")
    run(x.npu(), torch.eye(144, dtype=torch.float16, device="npu"), y)
    torch.npu.synchronize()
    # unfold 为通道优先；调整为卷积核高、宽、输入通道的顺序。
    expected = torch.nn.functional.unfold(x.float(), kernel_size=3, padding=1)
    expected = expected.reshape(16, 3, 3, 16).permute(3, 1, 2, 0).reshape(16, 144)
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)
    print("conv2d_load_im2col example passed")


if __name__ == "__main__":
    main()
```

预期结果：展开结果与 Host 参考结果一致，输出 `conv2d_load_im2col example passed`。
