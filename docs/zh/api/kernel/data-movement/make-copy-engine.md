---
title: make_copy_engine
api_name: make_copy_engine
category: data-movement
api_group: kernel
layer: tensor
call_context: device
execution_unit: varies
status: experimental
since: 待追溯
---

# `make_copy_engine`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

创建一个数据搬运配置，供 `mem_copy(..., engine=...)` 使用。它用于声明格式转换、
多维搬运、目标排布、分区搬运、尾块对齐和 ReLU 等静态选项，本身不会搬运数据。

通常不需要显式创建 engine；当默认搬运无法表达所需行为时再使用本接口。

## 函数原型

```python
def make_copy_engine(
    *,
    kind: str = "auto",
    format_transform: str | None = None,
    transpose: bool | None = None,
    layout_transform: str | None = None,
    tail_alignment: str | None = None,
    padding_mode: str | None = None,
    split_axis: int | None = None,
    split_alignment: int = 1,
    block_mapping: CopyBlockMapping | None = None,
    relu: bool = False,
    smallc0: bool = False,
    dst_nd_arrangement: str | None = None,
) -> CopyEngine: ...
```

所有参数均须使用关键字传入。

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `kind` | `str` | `"auto"` | 搬运方式。`"auto"` 由系统选择；`"nddma"` 用于 GM→UB 的 1～5 维搬运。 |
| `format_transform` | `str` 或 `None` | `None` | 格式转换。`None` 表示根据源、目标布局自动选择。支持值见下表。 |
| `transpose` | `bool` 或 `None` | `None` | 是否对 L1→L0A/L0B 搬运执行转置。`True` 等价于 `format_transform="transpose"`。 |
| `layout_transform` | `str` 或 `None` | `None` | 布局变换。支持 `"none"` 和 `"nz_fold_n_to_m"`。 |
| `tail_alignment` | `str` 或 `None` | `None` | GM→UB 尾块的 UB 行对齐方式。支持 `"tiler"` 和 `"ub"`。 |
| `padding_mode` | `str` 或 `None` | `None` | NDDMA padding 模式。`"constant"` 使用 `mem_copy` 的 `pad_value`；`"nearest"` 使用最近的边界值。 |
| `split_axis` | `int` 或 `None` | `None` | 将二维搬运分成两部分：`0` 按外轴切分，`1` 按内轴切分。 |
| `split_alignment` | `int` | `1` | 分区大小的对齐值，必须为正整数；非 1 值仅支持 `split_axis=0`。 |
| `block_mapping` | `CopyBlockMapping` 或 `None` | `None` | 描述二维块的重复搬运方式，适用于规则的广播、对角或跨步块搬运。 |
| `relu` | `bool` | `False` | 对 L0C→GM/UB/L1 搬运结果应用 ReLU。 |
| `smallc0` | `bool` | `False` | 为 DN2NZ GM→L1 搬运启用紧凑 C0 模式；输入的 D 维不能大于 4。 |
| `dst_nd_arrangement` | `str` 或 `None` | `None` | ND2NZ GM→L1 的目标排布。`None` 或 `"batched"` 保持各批次独立；`"stack_m"` 将批次沿 M 轴拼成一个 NZ 矩阵。 |

### `format_transform` 支持值

| 值 | 典型方向 | 说明 |
| --- | --- | --- |
| `"identity"` | 多种同格式搬运 | 不做格式转换。 |
| `"nd2nz"` | GM/UB→L1 | ND 转 NZ。 |
| `"dn2nz"` | GM→L1 | DN 转 ZN。 |
| `"nz2nd"` | L0C→GM/UB | Fractal 格式转 ND。 |
| `"transpose"` | L1→L0A/L0B | 转置加载。 |
| `"mx_scale_a"`、`"mx_scale_b"` | GM→L1 | MX ScaleA/ScaleB 的兼容格式转换。 |
| `"mx_scale_and"`、`"mx_scale_adn"` | GM→L1 | 成对 E8M0 ScaleA 的布局转换。 |
| `"mx_scale_bnd"`、`"mx_scale_bdn"` | GM→L1 | 成对 E8M0 ScaleB 的布局转换。 |

`transpose` 与显式 `format_transform` 必须一致。例如，不能同时指定
`format_transform="nd2nz"` 和 `transpose=True`。

### `CopyBlockMapping`

```python
class CopyBlockMapping:
    def __init__(
        self,
        block_shape,
        *,
        repeats,
        src_step,
        dst_step,
    ) -> None: ...
```

- `block_shape`、`src_step`、`dst_step` 均为长度为 2 的整数序列。
- `block_shape` 和 `repeats` 必须为正数，步长必须为非负数。
- 第 `i` 次搬运从 `i * src_step` 开始，写入 `i * dst_step`；坐标单位为元素。
- 仅支持二维 ND2NZ GM→L1 和 L1→L0A/L0B 搬运。L1→L0A/L0B 时仅支持
  16 位元素和 `block_shape=(16, 16)`。

## 返回值说明

返回 `CopyEngine`，传给 `mem_copy` 的 `engine` 参数。

## 使用约束

- 配置必须与 `mem_copy` 的源、目标、数据类型和布局匹配；不适用的组合会在编译时报错。
- `kind="nddma"` 仅用于 Identity GM→UB；源、目标维度必须相同且为 1～5，
  dtype 必须一致。padding 的数量和值通过 `mem_copy` 传入。
- `tail_alignment="ub"` 仅用于 Identity GM→UB，不能与 NDDMA 或布局变换组合。
- `split_axis` 仅用于二维搬运。GM→UB、UB→GM、UB→L1 需要在 `mem_copy`
  中提供 `part_id`；L0C→UB 的双目标搬运不提供 `part_id`。
- `layout_transform="nz_fold_n_to_m"` 仅用于 UB→L1，要求
  `format_transform="identity"`，并需配合 `split_axis=1` 和 `part_id`。
- `relu=True` 仅用于 L0C→GM/UB/L1，不能与 L0C→UB 双目标分区组合。
- `block_mapping` 不能与分区、NDDMA、布局变换、ReLU、`smallc0`、scale 或 padding 组合。
- `dst_nd_arrangement="stack_m"` 仅用于 3维 `(b, m, k)` ND GM 源到 2维
  `(b*m, k)` NZ L1 目标，必须同时指定 `format_transform="nd2nz"`，并保持
  `kind="auto"`。不能与 `block_mapping`、`split_axis` 或多源搬运组合。
- 使用 `"stack_m"` 时，L1 目标应按可能出现的最大 `b*m` 声明容量。若 `b` 或 `m`
  为运行时值，调用者必须保证实际 `b*m` 不超过该容量；设备侧不会执行越界检查。

## 调用示例

以下示例使用 NDDMA 将 32 个 `float32` 元素从 GM 搬入 UB，并在两侧各填充
8 个 `-1.0`，随后写回 GM。运行环境需安装 cannbotdsl、CANN、PyTorch 和
torch_npu，并具有受支持的 NPU。

```python
import cannbotdsl as cb
import torch
import torch_npu

@cb.kernel
def padded_copy_kernel(src: cb.Tensor, dst: cb.Tensor):
    tmp = cb.Channel(
        cb.MemLoc.UB,
        shape=(48,),
        dtype=src.dtype,
        depth=1,
    )
    engine = cb.make_copy_engine(
        kind="nddma",
        padding_mode="constant",
    )

    slot = tmp.produce()
    cb.mem_copy(
        slot,
        src,
        engine=engine,
        left_padding=(8,),
        right_padding=(8,),
        pad_value=-1.0,
    )
    slot = tmp.consume()
    cb.mem_copy(dst, slot)

@cb.host
def padded_copy(src: cb.Tensor, dst: cb.Tensor):
    padded_copy_kernel[1](src, dst)

def main():
    src = torch.arange(32, dtype=torch.float32, device="npu:0")
    dst = torch.empty(48, dtype=torch.float32, device="npu:0")

    padded_copy(src, dst)
    torch.npu.synchronize()

    actual = dst.cpu()
    expected = torch.full((48,), -1.0, dtype=torch.float32)
    expected[8:40] = torch.arange(32, dtype=torch.float32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    print(f"shape={tuple(actual.shape)}, first={actual[0].item()}, last={actual[-1].item()}")

if __name__ == "__main__":
    main()
```

### 预期输出

```text
shape=(48,), first=-1.0, last=-1.0
```
