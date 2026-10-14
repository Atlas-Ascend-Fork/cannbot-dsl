---
title: mem_copy
api_name: mem_copy
category: data-movement
api_group: kernel
layer: tensor
call_context: device
execution_unit: varies
status: experimental
since: 待追溯
---

# `mem_copy`

## 产品支持情况

- Ascend 950PR/Ascend 950DT：支持
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
- Atlas 200I/500 A2 推理产品：不支持
- Atlas 推理系列产品 AI Core：不支持
- Atlas 推理系列产品 Vector Core：不支持
- Atlas 训练系列产品：不支持

## 功能说明

在 Kernel 内将数据从 `src` 搬运到 `dst`。接口根据源、目标的存储位置和布局选择
搬运方式，通过 `make_copy_engine` 指定格式转换、分区、NDDMA 等行为。

## 函数原型

```python
def mem_copy(
    dst: Tensor | tuple[Tensor, ...] | list[Tensor],
    src: Tensor | tuple[Tensor, ...] | list[Tensor],
    *,
    engine: CopyEngine | None = None,
    transpose: bool | None = None,
    deq_scale: int | float | Scalar | Tensor | None = None,
    dst_subblock: int | Scalar | None = None,
    unit_flag: int | Scalar = 0,
    pad_value: int | float | Scalar | None = None,
    left_padding: int | Scalar | tuple | list | None = None,
    right_padding: int | Scalar | tuple | list | None = None,
    mx_scale: Tensor | None = None,
    part_id: int | Scalar | None = None,
    actual: tuple | list | None = None,
    l2_cache_ctl: int | Scalar = 0,
    atomic_add: bool | Scalar = False,
    axis: int | None = None,
) -> None: ...
```

除 `dst` 和 `src` 外，其他参数均须使用关键字传入。

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `dst` | `Tensor` 或 Tensor 序列 | 无 | 目标 Tensor。使用 Channel 时，通过 `produce()` 取得目标 Tensor；传入序列时，将一个 GM 源广播写入多个 UB 或 L1 目标。 |
| `src` | `Tensor` 或 Tensor 序列 | 无 | 源 Tensor。使用 Channel 时，通过 `consume()` 取得源 Tensor；传入序列时，将多个 GM 或 UB 源按顺序写入同一个目标。 |
| `engine` | `CopyEngine` | `None` | `make_copy_engine` 创建的搬运配置。为 `None` 时，根据搬运方向和 Tensor 布局自动选择默认搬运方式。 |
| `transpose` | `bool`   | `None` | L1→L0A/L0B 转置加载的简写。与 `engine` 中的显式格式配置必须一致。 |
| `deq_scale` | `int`/`float`标量或 `Tensor` | `None` | L0C 搬出时的 Fixpipe 转换系数。标量形式供整次搬运共享；Tensor 形式用于逐通道反量化。 |
| `dst_subblock` | `int`标量 | `None` | L0C→UB 单目标搬运的目标子块，只能为 0 或 1。 |
| `unit_flag` | `int`标量 | `0` | L0C 搬运的 unit flag，取值为 0～3。 |
| `pad_value` | `int`/`float`标量 | `None` | padding 填充值。 |
| `left_padding` | `int`标量或`Tuple`或`list` | `None` | 各轴左侧 padding 数量。单个值等价于长度为 1 的序列。 |
| `right_padding` | `int`标量或`Tuple`或`list` | `None` | 各轴右侧 padding 数量。 |
| `mx_scale` | `Tensor` | `None` | L1→L0A/L0B 搬运使用的 E8M0 scale Tensor。 |
| `part_id` | `int`标量 | `None` | 选择 `split_axis` 配置产生的分区，通常传入当前 subblock id。 |
| `actual` | `tuple` 或 `list` | `None` | 分区搬运的实际二维 shape；能够从 Tensor 推导时可省略。 |
| `l2_cache_ctl` | `int`标量 | `0` | L2 Cache 策略，只能取 0、1、2、4。 |
| `atomic_add` | `bool` | `False` | 写入 GM 时是否执行原子累加。 |
| `axis` | `int`标量 | `None` | 多源搬运的拼接轴，当前只支持 `0`；单源和多目标搬运不能指定。 |

### `engine=None` 的默认行为

不指定 `engine` 时，接口会根据源、目标的存储位置和 Tensor 布局选择搬运方式：

- 一般情况下执行不改变逻辑布局的普通搬运。
- GM→L1 时，如果目标 Tensor 的布局要求 ND→NZ 或 DN→NZ，自动执行对应的格式转换。
- GM→UB 使用 padding，或源 Tensor 为适合 NDDMA 搬运的非连续视图时，自动使用 NDDMA；
  其他情况使用普通搬运。

`engine=None` 不会自动启用转置、分区、ReLU、布局折叠或块映射等可选行为。
需要这些行为时，应通过 `make_copy_engine` 显式配置；L1→L0A/L0B 的转置也可以直接指定
`transpose=True`。

## 支持的搬运方向

| 源 | 目标 | 支持的行为 |
| --- | --- | --- |
| GM | UB | 普通搬运、NDDMA、分区搬运 |
| GM | L1 | 普通搬运、ND2NZ、DN2NZ、MX scale 格式转换 |
| UB | GM | 普通搬运、分区搬运、原子累加 |
| UB | UB | 相同布局的普通搬运 |
| UB | L1 | 普通搬运、ND2NZ、分区搬运 |
| L1 | L0A/L0B | 普通加载、转置加载、MX scale 加载 |
| L1 | BIAS | Bias 加载 |
| L1 | UB | 相同 Dtype 的连续搬运，可选择目标 Vector SubBlock |
| L0C | GM/UB | 普通或 NZ2ND 搬运、量化/反量化；写 GM 时可原子累加 |
| L0C | L1 | 普通搬运 |

当 `dst` 为序列时，一个 ND GM 源可广播到多个 UB 目标，或多个 ND/NZ L1 目标。

源、目标的 dtype、shape、stride 和物理布局还需满足所选方向的要求，编译器会对
不匹配的组合报错。

## 返回值说明

返回 `None`。

## 使用约束

### 通用约束

- `src` 和 `dst` 必须是 Tensor 或对应模式支持的 Tensor 序列；不能同时传入源序列和
  目标序列。Channel 应先通过 `produce()` / `consume()` 选择 Tensor。
- 普通搬运要求源、目标的数据范围和布局兼容。UB→UB 只支持相同布局，不支持格式转换。
- L1→BIAS 仅支持连续的一维数据：float16/bfloat16/float32 可写入 float32，
  int32 可写入 int32；目标数据量不能超过 4096 字节。

### 多源搬运

- `src` 为序列时，源必须非空，且均为同一存储空间、同 dtype 的 ND Tensor。
  GM 源可写入 UB 或 L1；UB 源只能写入 ND L1。
- 仅支持 `axis=0`。各源必须能按输入顺序组成规则、连续的搬运区间。
- 多源模式只支持默认搬运、匹配目标布局的 ND2NZ engine 和 `l2_cache_ctl`；
  不能与分区、scale、padding 或原子累加组合。

### 多目标搬运

- `dst` 为多个 Tensor 的序列时，源必须是一个 ND GM Tensor，目标必须全部位于 UB
  或全部位于 L1，并具有相同的 dtype、shape 和布局。单元素目标序列等价于普通单目标搬运。
- 多个目标应按固定地址间隔排列；接口根据前两个目标确定间隔，调用者须保证其余目标
  延续相同的地址排列。
- 多目标模式只支持默认 Identity 搬运或匹配 NZ L1 目标的 ND2NZ engine，以及
  `l2_cache_ctl`；不能与分区、scale、padding、原子累加等选项组合。

### Padding 与 NDDMA

- `left_padding`、`right_padding` 每项取值为 0～255，序列长度为 1～5。
- NDDMA 仅支持 GM→UB，源和目标维度同且为 1～5，dtype 必须一致；支持
  8/16/32 位整数、float16、bfloat16 和 float32。
- NDDMA padding 序列长度必须与源维度一致；静态目标 shape 应等于源 shape 加左右 padding。
- 非 NDDMA 的 padding 仅支持 Identity GM→L1，每侧只能提供一个最内层 padding 数量，
  且不能超过 32 字节数据量。
- `padding_mode="nearest"` 仅支持 NDDMA；常量 padding 使用 `pad_value`。

### Scale、分区及其他选项

- `mx_scale` 仅用于 L1→L0A/L0B；scale 必须位于 L1、dtype 为 `fp8_e8m0`，
  且 shape 和布局与数据匹配。
- 标量 `deq_scale` 仅用于 L0C→GM/UB/L1 的受支持 Fixpipe 类型转换。
  Tensor `deq_scale` 仅用于 int32 L0C→GM/UB 的逐通道反量化，目标须为
  float16 或 bfloat16；scale 表须位于 L1 或 FBUF。
- `part_id` 和 `actual` 仅能与配置了 `split_axis` 的 engine 一起使用。
  GM→UB、UB→GM、UB→L1 分区搬运需要 `part_id`；L0C→UB 双目标搬运必须省略它。
- `atomic_add` 仅支持 UB/L0C→GM。目标 dtype 支持 int8、int16、int32、
  bfloat16、float16 和 float32。
- 非零 `l2_cache_ctl` 仅支持 GM→UB、UB→GM、GM→L1 和 L0C→GM。

## 调用示例

### 通过 UB 搬运

以下示例通过 UB 将 16 个 `float32` 元素从一个 GM Tensor 搬运到另一个 GM Tensor。

```python
import cannbotdsl as cb
import torch
import torch_npu

@cb.kernel
def copy_kernel(src: cb.Tensor, dst: cb.Tensor):
    tmp = cb.Channel(
        cb.MemLoc.UB,
        shape=(16,),
        dtype=src.dtype,
        depth=1,
    )

    slot = tmp.produce()
    cb.mem_copy(slot, src)
    slot = tmp.consume()
    cb.mem_copy(dst, slot)

@cb.host
def copy(src: cb.Tensor, dst: cb.Tensor):
    copy_kernel[1](src, dst)

def main():
    src = torch.arange(16, dtype=torch.float32, device="npu:0")
    dst = torch.empty_like(src)

    copy(src, dst)
    torch.npu.synchronize()

    actual = dst.cpu()
    expected = torch.arange(16, dtype=torch.float32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    print(f"elements={actual.numel()}, first={actual[0].item()}, last={actual[-1].item()}")

if __name__ == "__main__":
    main()
```

#### 预期输出

```text
elements=16, first=0.0, last=15.0
```

### 通过 Cube 搬运

以下示例将两个 GM 矩阵依次搬入 L1 和 L0A/L0B，通过 Cube 完成矩阵乘，
再将 L0C 中的结果搬回 GM。

```python
import cannbotdsl as cb
import torch
import torch_npu

@cb.kernel
def cube_copy_kernel(lhs: cb.Tensor, rhs: cb.Tensor, dst: cb.Tensor):
    l1_lhs = cb.Channel(
        cb.MemLoc.L1,
        shape=(16, 16),
        dtype=lhs.dtype,
        depth=1,
    )
    l1_rhs = cb.Channel(
        cb.MemLoc.L1,
        shape=(16, 16),
        dtype=rhs.dtype,
        depth=1,
    )
    l0a = cb.Channel(
        cb.MemLoc.L0A,
        shape=(16, 16),
        dtype=lhs.dtype,
        depth=1,
    )
    l0b = cb.Channel(
        cb.MemLoc.L0B,
        shape=(16, 16),
        dtype=rhs.dtype,
        depth=1,
    )
    l0c = cb.Channel(
        cb.MemLoc.L0C,
        shape=(16, 16),
        dtype=cb.dtypes.float32,
        depth=1,
    )

    nd2nz = cb.make_copy_engine(format_transform="nd2nz")
    cb.mem_copy(l1_lhs.produce(), lhs, engine=nd2nz)
    cb.mem_copy(l1_rhs.produce(), rhs, engine=nd2nz)
    cb.mem_copy(l0a.produce(), l1_lhs.consume())
    cb.mem_copy(l0b.produce(), l1_rhs.consume())
    cb.matmul(l0c.produce(), l0a.consume(), l0b.consume(), init=True)
    cb.mem_copy(dst, l0c.consume())

@cb.host
def cube_copy(lhs: cb.Tensor, rhs: cb.Tensor, dst: cb.Tensor):
    cube_copy_kernel[1](lhs, rhs, dst)

def main():
    lhs = (
        torch.arange(256, dtype=torch.float16, device="npu:0")
        .reshape(16, 16)
        .div_(16)
    )
    rhs = torch.eye(16, dtype=torch.float16, device="npu:0")
    dst = torch.empty((16, 16), dtype=torch.float32, device="npu:0")

    cube_copy(lhs, rhs, dst)
    torch.npu.synchronize()

    actual = dst.cpu()
    expected = lhs.cpu().float()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    print(
        f"shape={tuple(actual.shape)}, "
        f"first={actual[0, 0].item()}, last={actual[-1, -1].item()}"
    )

if __name__ == "__main__":
    main()
```

#### 预期输出

```text
shape=(16, 16), first=0.0, last=15.9375
```
