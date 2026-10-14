---
title: make_channel
api_name: make_channel
category: types-and-views
api_group: kernel
layer: frontend
call_context: device
execution_unit: varies
status: experimental
since: 待追溯
---

# `make_channel(...)`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

使用 `make_channel()` 显式空间构造时不会重新申请或重排内存，而是直接将
`dsl.UB.view()` 派生的静态 UB Tensor 视图依次绑定为槽位。这些视图可以具有
不同的起始地址，因此整组槽位不要求连续。该接口用于需要手工切分和管理 UB 地址
的场景。

## 函数原型

```python
make_channel(
    views: list[Tensor] | tuple[Tensor, ...],
    *,
    kind: ChannelKind | None = None,
) -> Channel
```

## 参数说明

| 参数 | 说明 |
| --- | --- |
| `views` | 待绑定为 Channel 槽位的 UB Tensor 视图序列。 |
| `kind` | 指定 SameCore 或 CrossCore 同步范围。 |

`views` 接受 `list[Tensor]` 或 `tuple[Tensor, ...]`。两者在 `make_channel()` 中
没有语义差异；省略号表示 tuple 中可以包含任意数量的 Tensor。

## 返回值说明

返回固定为 `MemLoc.UB`、ND 格式的 Channel。

## 约束说明

- `views` 必须是非空 list 或 tuple；元素顺序就是槽位顺序，元素数量就是
  `depth`。
- 每个视图都必须由 `dsl.UB.view()` 派生，并且尚未绑定为其他 Buffer 或 Channel
  资源。
- 传入的派生视图必须具有静态 shape 和静态 stride。
- 所有视图必须具有完全相同的 shape、stride 和 layout。
- 各槽位通常应使用互不重叠的 UB 区域。框架不会为用户手工造成的地址重叠自动
  建立同步保护。
- 同一个 Kernel 调用 `dsl.UB.view()` 后，不能再调用普通
  `Channel(MemLoc.UB, ...)`；后续 UB Channel 应通过 `make_channel()` 创建。

## 调用示例

下面的脚本从同一个完整 UB 根视图中划分两个互不重叠的 Channel。Channel A 包含
3 个 4 字节槽位，Channel B 包含 2 个 8 字节槽位；各槽位的起始地址按 32 字节
对齐，其余 UB 空间仍可继续派生其他视图。

```python
import torch
from torch import as_tensor as from_torch_npu

import cannbotdsl as cbd


@cbd.kernel
def multi_channel_kernel(
    x_a: cbd.Tensor,
    y_a: cbd.Tensor,
    x_b: cbd.Tensor,
    y_b: cbd.Tensor,
):
    root = cbd.UB.view(256 * 1024)

    # Channel A：3 个槽位，每个槽位包含 4 个元素。
    a0 = root.reinterpret(shape=(4,), offset=0)
    a1 = root.reinterpret(shape=(4,), offset=32)
    a2 = root.reinterpret(shape=(4,), offset=64)
    channel_a = cbd.make_channel([a0, a1, a2])

    # Channel B：继续使用后续 UB 空间；2 个槽位各包含 8 个元素。
    b0 = root.reinterpret(shape=(8,), offset=96)
    b1 = root.reinterpret(shape=(8,), offset=128)
    channel_b = cbd.make_channel([b0, b1])

    cbd.mem_copy(channel_a.produce(), x_a)
    cbd.mem_copy(y_a, channel_a.consume())

    cbd.mem_copy(channel_b.produce(), x_b)
    cbd.mem_copy(y_b, channel_b.consume())


@cbd.host
def run(x_a, y_a, x_b, y_b):
    multi_channel_kernel[1](x_a, y_a, x_b, y_b)


if __name__ == "__main__":
    x_a = torch.tensor([1, 2, 3, 4], device="npu").byte()
    y_a = torch.empty_like(x_a)

    x_b = torch.arange(10, 18, device="npu").byte()
    y_b = torch.empty_like(x_b)

    run(
        from_torch_npu(x_a),
        from_torch_npu(y_a),
        from_torch_npu(x_b),
        from_torch_npu(y_b),
    )
    torch.npu.synchronize()

    torch.testing.assert_close(y_a, x_a)
    torch.testing.assert_close(y_b, x_b)
    print("channel_a input :", x_a.cpu().tolist())
    print("channel_a output:", y_a.cpu().tolist())
    print("channel_b input :", x_b.cpu().tolist())
    print("channel_b output:", y_b.cpu().tolist())
```

### 示例输入

```text
完整 UB 根视图：offset [0, 262144)

Channel A：
  a0：offset [0, 4)，shape=(4,)
  a1：offset [32, 36)，shape=(4,)
  a2：offset [64, 68)，shape=(4,)
  depth=3

Channel B：
  b0：offset [96, 104)，shape=(8,)
  b1：offset [128, 136)，shape=(8,)
  depth=2

x_a = [1, 2, 3, 4]
x_b = [10, 11, 12, 13, 14, 15, 16, 17]
```

### 预期输出

```text
channel_a input : [1, 2, 3, 4]
channel_a output: [1, 2, 3, 4]
channel_b input : [10, 11, 12, 13, 14, 15, 16, 17]
channel_b output: [10, 11, 12, 13, 14, 15, 16, 17]
```

