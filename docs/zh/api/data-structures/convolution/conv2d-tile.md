# Conv2dTile

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

描述当前卷积切块的有效范围和操作数视图，由 Conv2dSpec.tile 创建。视图方法不搬运数据。

## 函数原型

```python
tile.l0a_view(slot: Tensor) -> Tensor
tile.l1b_view(slot: Tensor) -> Tensor
tile.l0b_view(slot: Tensor) -> Tensor
tile.l0c_view(slot: Tensor) -> Tensor
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| slot | 输入 | 对应存储位置的已分配 Tensor。L0A 按 `(BM, Kh*Kw*BC)`；权重 L1 和 L0B 按示例的轴交换布局；L0C 按 `(BM, BN)` 预留容量。 |

## 返回值说明

返回共享 slot 存储的有效切块视图。bm、bn、bc 为块容量；m_begin、n_begin、ci_begin 为逻辑起点；m_actual、n_actual、ci_actual 为尾块裁剪后的有效长度；ci_program、k_program 为搬运采用的补齐范围。input_window 返回不可变的 Conv2dInputWindow，描述裁剪后的 L1 输入窗口。其字段含义如下：

| 字段 | 含义 |
| --- | --- |
| h_raw_begin、h_raw_end | 补边前推导的输入行区间，左闭右开，可越出实际输入边界。 |
| h_gm_begin、h_gm_end | 裁剪到输入范围内的 GM 行区间，左闭右开。 |
| l1_height、l1_width | 当前 L1 特征窗口的高、宽。 |
| pad_top、pad_bottom、pad_left、pad_right | 当前窗口四边的补零宽度。 |
| m_start、m_extension | 当前窗口内展开矩阵的 M 起点和范围。 |

## 约束说明

同一轮特征图、权重、矩阵乘与输出搬运须使用匹配的 tile；视图容量须覆盖补齐后的物理访问范围。

## 调用示例

以下示例执行带边界填充的 3×3 Conv2D，并与 Host 参考结果比较。

```python
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
import torch
import torch_npu
import cannbotdsl as cb
from cannbotdsl import (Tensor, Buffer, RingBuffer, MemLoc, dtypes, make_layout,
    make_conv2d_spec, conv2d_load_fmap, conv2d_load_im2col,
    conv2d_load_filter, conv2d_store_output, mem_copy, matmul, kernel, host)

class Conv2dNchwKernel:
    def __init__(
        self,
        *,
        dtype=dtypes.float16,
        input_shape=(1, 16, 4, 4),
        filter_shape=(16, 16, 3, 3),
        tile_shape=(16, 16, 16),
        stride=(1, 1),
        padding=(1, 1, 1, 1),
        dilation=(1, 1),
        groups=1,
    ):
        self.dtype = dtype
        self.tile_shape = tile_shape
        self.spec = make_conv2d_spec(
            dtype=dtype,
            input_shape=input_shape,
            filter_shape=filter_shape,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    @kernel
    def conv2d_kernel(self, gm_x: Tensor, gm_filter: Tensor, gm_y: Tensor):
        bm, bn, bc = self.tile_shape
        khkw = self.spec.kernel_height * self.spec.kernel_width
        m_tiles, n_tiles, ci_tiles = self.spec.tile_count(self.tile_shape)

        fmap_l1 = RingBuffer(
            MemLoc.L1,
            self.dtype,
            make_layout(self.spec.fmap_l1_shape(tile=(bm, bc))),
            depth=2,
            data_format="nd",
        )
        filter_l1 = RingBuffer(
            MemLoc.L1,
            self.dtype,
            make_layout((bn, khkw * bc)),
            depth=2,
        )
        l0a = RingBuffer(
            MemLoc.L0A,
            self.dtype,
            make_layout((bm, khkw * bc)),
            depth=2,
        )
        l0b = RingBuffer(
            MemLoc.L0B,
            self.dtype,
            make_layout((bn, khkw * bc)),
            depth=2,
        )
        l0c = RingBuffer(
            MemLoc.L0C,
            dtypes.float32,
            make_layout((bm, bn)),
            depth=2,
        )

        for batch in range(self.spec.batch_size):
            for group in range(self.spec.groups):
                for m_idx in range(m_tiles):
                    for n_idx in range(n_tiles):
                        l0c_slot = l0c.next_write()
                        for ci_idx in range(ci_tiles):
                            tile = self.spec.tile(
                                tile=self.tile_shape,
                                coord=(m_idx, n_idx, ci_idx),
                                batch=batch,
                                group=group,
                            )

                            fmap_slot = fmap_l1.next_write()
                            conv2d_load_fmap(fmap_slot, gm_x, tile)

                            filter_slot = filter_l1.next_write().permute(1, 0)
                            conv2d_load_filter(filter_slot, gm_filter, tile)

                            fmap_ready = fmap_l1.next_read()
                            l0a_slot = l0a.next_write()
                            conv2d_load_im2col(tile.l0a_view(l0a_slot), fmap_ready, tile)

                            filter_ready = filter_l1.next_read().permute(1, 0)
                            l0b_slot = l0b.next_write().permute(1, 0)
                            mem_copy(
                                tile.l0b_view(l0b_slot),
                                tile.l1b_view(filter_ready),
                            )

                            l0a_ready = l0a.next_read()
                            l0b_ready = l0b.next_read().permute(1, 0)
                            matmul(
                                tile.l0c_view(l0c_slot),
                                tile.l0a_view(l0a_ready),
                                tile.l0b_view(l0b_ready),
                                init=(ci_idx == 0),
                            )

                        output_tile = self.spec.tile(
                            tile=self.tile_shape,
                            coord=(m_idx, n_idx, 0),
                            batch=batch,
                            group=group,
                        )
                        l0c_ready = l0c.next_read()
                        conv2d_store_output(
                            gm_y,
                            output_tile.l0c_view(l0c_ready),
                            output_tile,
                        )

    @host
    def run(self, gm_x: Tensor, gm_filter: Tensor, gm_y: Tensor):
        self.conv2d_kernel[1](gm_x, gm_filter, gm_y)

def main():
    op = Conv2dNchwKernel()
    x = (torch.arange(256).reshape(1, 16, 4, 4) % 7 - 3).half() / 8
    w = (torch.arange(2304).reshape(16, 16, 3, 3) % 5 - 2).half() / 8
    out = torch.empty(op.spec.output_shape, dtype=torch.float16, device="npu:0")
    op.run(x.to("npu:0"), w.to("npu:0"), out)
    torch.npu.synchronize()
    reference = torch.nn.functional.conv2d(x.float(), w.float(), padding=1)
    torch.testing.assert_close(out.cpu(), reference.half(), rtol=0, atol=0)
    print("conv2d example passed")

if __name__ == "__main__":
    main()
```

预期结果：校验通过，输出 `conv2d example passed`。
