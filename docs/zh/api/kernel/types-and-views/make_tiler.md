---
title: make_tiler
api_name: make_tiler
category: types-and-views
api_group: kernel
layer: frontend
call_context: device
execution_unit: varies
status: experimental
since: 待追溯
---

# `make_tiler(...)`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

`make_tiler()` 将各维切块长度和可选 alignment 组合成可复用的 `Tiler`。返回值可以
作为一个整体传给 `tile_slice()`、`reinterpret()` 等接受 Tiler 的接口，避免重复
声明相同的切块规格和对齐条件。

## 函数原型

```python
make_tiler(
    tiler: int | tuple,
    *,
    alignment: int | tuple[int, ...] | list[int] | None = None,
) -> Tiler
```

## 参数说明

| 参数 | 类型 | 必选 | 默认值 | 详细说明 |
| --- | --- | --- | --- | --- |
| `tiler` | Python `int`、Kernel 运行时整数值或由这些值组成的嵌套 tuple/list | 是 | 无 | 指定各维切块长度及其分组结构。 |
| `alignment` | 正 Python `int`、正整数 tuple/list 或 `None` | 否 | `None` | 声明各维切块长度的对齐粒度；不会改变 `tiler` 中的切块长度。 |

### `tiler` 参数可选值

| 可选形式 | 含义 | 约束 |
| --- | --- | --- |
| 正 Python `int` | 编译期确定的一维切块长度。 | 必须大于 0。 |
| Kernel 标量形参、循环索引或整数计算结果 | Kernel 运行时确定的一维切块长度。 | 必须能转换为 Kernel 运行时使用的 64 位整数。 |
| 由上述值组成的 tuple/list | 按从左到右的顺序描述多维切块；嵌套结构会保留在返回的 Tiler 中。 | 每个值分别满足对应的编译期或运行期约束。 |

### `alignment` 参数可选值

| `tiler` | `alignment` | 对齐关系与约束 |
| --- | --- | --- |
| `(m, n)` | `None` | 不额外声明对齐粒度。 |
| `(m, n)` | `16` | 单个正 Python `int` 应用到每个最终维度值，因此 `m` 和 `n` 的对齐粒度均为 16。 |
| `(m, n)` | `(16, 8)` | tuple/list 只能包含正 Python `int`，元素数量必须等于最终维度值的数量；此处依次对应 `m` 和 `n`。 |
| `((m, n), k)` | `(16, 8, 4)` | tuple/list 按从左到右的顺序逐项对应最终维度值，此处依次对应 `m`、`n` 和 `k`。 |

`alignment` 不接受 bool、0、负数或运行时整数。

## 返回值说明

返回一个 Tiler。

## 调用示例

```python
import torch
from torch import as_tensor as from_torch_npu

import cannbotdsl as cbd


@cbd.kernel
def make_tiler_kernel(x: cbd.Tensor, y: cbd.Tensor):
    tiler = cbd.make_tiler((16, 16), alignment=(16, 16))
    tile = cbd.tile_slice(x, tiler=tiler, coord=(0, 0))
    y[0] = tile[0, 0]
    y[1] = tile[15, 15]


@cbd.host
def run(x, y):
    make_tiler_kernel[1](x, y)


if __name__ == "__main__":
    x = torch.arange(400, dtype=torch.int64, device="npu").reshape(20, 20)
    y = torch.empty(2, dtype=torch.int64, device="npu")
    run(from_torch_npu(x), from_torch_npu(y))
    torch.npu.synchronize()
    print(y.cpu().tolist())
```

### 示例输入

```text
x = torch.arange(400).reshape(20, 20)
tiler=(16, 16)，alignment=(16, 16)，coord=(0, 0)
```

### 预期输出

```text
[0, 315]
```
