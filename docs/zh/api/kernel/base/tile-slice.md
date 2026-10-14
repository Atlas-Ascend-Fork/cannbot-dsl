---
title: tile_slice
api_name: tile_slice
category: tensor-view
api_group: kernel
layer: tensor
call_context: device
execution_unit: none
status: experimental
since: 待追溯
---

# `tile_slice`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

在 Tensor 上按 tile 坐标创建 tile 视图。该接口只创建别名视图，不复制数据。

## 函数原型

```python
def tile_slice(
    input: Tensor,
    tiler: Tiler,
    coord: Coord,
) -> Tensor: ...
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `input` | `Tensor` | 无 | 输入 Tensor。使用 Channel 时，应先通过 `produce()` 或 `consume()` 取得 Tensor。 |
| `tiler` | 整数、整数序列或 `Tiler` | 无 | 每个 tile 的大小。需要动态大小或对齐要求时，传入 `make_tiler()` 创建的 `Tiler`。 |
| `coord` | 整数、整数序列或 `Coord` | 无 | tile 坐标，结构需与 `tiler` 对应；`coord` 表示 tile 序号，而不是元素下标。例如，`tiler=(64, 128)`、`coord=(2, 1)` 表示从元素坐标 `(128, 128)` 开始取一个 `64 × 128` 的 tile。如果最后一个 tile 不完整，返回视图的实际 shape 会缩小到剩余范围。 |

当输入维度为 `N`、`tiler` 维度为 `K` 时，`tile_slice` 从输入最后 `K`
个逻辑维度取 tile，并返回包含这 `K` 个维度的视图。

## 返回值说明

返回与输入共享同一存储的 Tensor 视图。

返回视图保留输入对应维度的 stride，不会将非连续数据自动压紧。

## 使用约束

- `tiler` 和 `coord` 不能为空，且展开后的维度数必须相同；tile 维度数不能超过输入维度数。
- 静态 tile 大小必须为正整数，动态 tile 当前仅支持 Identity/ND 的 GM Tensor。
- 动态 tile 的对齐要求应在创建 `Tiler` 时通过 `make_tiler(..., alignment=...)` 声明。
- 对 NZ/ZN 等映射布局切分时，内部 tile 边界必须满足对应布局的分块对齐要求。
  NZ（代码中的 ND2NZ）最后两个逻辑维度分别按 `16` 和 `32 / 元素字节数`
  对齐；ZN（代码中的 DN2NZ）交换上述两个维度的对齐粒度。以 `coord=0`
  覆盖完整逻辑轴时，不要求该轴的 `tiler` 本身是分块大小的整数倍。
- 调用者应保证 `coord` 位于有效 tile 范围内。静态非整除尾块会自动裁剪，但动态坐标
  和超大 tile 不提供通用的运行时越界保护。
- 视图与原 Tensor 共享数据。对同一存储的并发读写仍需由调用者保证依赖关系。

## 调用示例

以下示例把长度为 32 的 GM Tensor 切成两个长度为 16 的 tile，并将第 2 个 tile
写入输出 Tensor。运行环境需安装 cannbotdsl、CANN、PyTorch 和 torch_npu，并具有
受支持的 NPU。

```python
import cannbotdsl as cb
import torch
import torch_npu

@cb.kernel
def copy_second_tile_kernel(src: cb.Tensor, dst: cb.Tensor):
    src_tile = cb.tile_slice(src, (16,), (1,))
    tmp = cb.Channel(
        cb.MemLoc.UB,
        shape=(16,),
        dtype=src.dtype,
        depth=1,
    )

    slot = tmp.produce()
    cb.mem_copy(slot, src_tile)
    slot = tmp.consume()
    cb.mem_copy(dst, slot)

@cb.host
def copy_second_tile(src: cb.Tensor, dst: cb.Tensor):
    copy_second_tile_kernel[1](src, dst)

def main():
    src = torch.arange(32, dtype=torch.float32, device="npu:0")
    dst = torch.empty(16, dtype=torch.float32, device="npu:0")

    copy_second_tile(src, dst)
    torch.npu.synchronize()

    actual = dst.cpu()
    expected = torch.arange(16, 32, dtype=torch.float32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    print("output:", actual.tolist())

if __name__ == "__main__":
    main()
```

### 预期输出

```text
output: [16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0]
```
