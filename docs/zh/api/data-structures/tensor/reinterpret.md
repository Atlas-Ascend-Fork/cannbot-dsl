# Tensor.reinterpret

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

返回与当前 Tensor 共享存储的新视图，可指定目标元素类型、逻辑布局、物理布局和字节偏移。新视图起点为当前 Tensor 起点加上 `offset`。

## 函数原型

```python
reinterpret(
    self,
    dtype=None,
    layout=None,
    *,
    physical_layout=None,
    offset=0,
    data_format=None,
) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| `dtype` | 输入 | 目标元素类型。默认 `None`，保留原类型。可选值：`dtypes.float16`、`dtypes.float32`、`dtypes.float64`、`dtypes.bfloat16`、`dtypes.float8_e4m3fn`、`dtypes.float8_e5m2`、`dtypes.float8_e8m0`、`dtypes.hifloat8`、`dtypes.int8`、`dtypes.int16`、`dtypes.int32`、`dtypes.int64`、`dtypes.uint8`、`dtypes.uint16`、`dtypes.uint32`、`dtypes.uint64`、`dtypes.fp4x2_e1m2`、`dtypes.fp4x2_e2m1`、`dtypes.int4x2`。具体说明见[预定义数据类型](../data-types.md)。 |
| `layout` | 输入 | 目标逻辑布局。默认 `None`，沿用原逻辑布局；可传入 `make_layout()` 返回的 Layout，指定静态或运行时的形状、跨度。 |
| `physical_layout` | 输入 | 目标物理布局。默认 `None`；提供新逻辑布局时按目标格式推导，逻辑/物理布局与格式全部省略时保留原布局。可显式传入 Layout。 |
| `offset` | 输入 | 相对当前 Tensor 起点的字节偏移。默认 `0`，可为 `0`～`2^63-1` 的 Python 整数，或运行时非负整数值。 |
| `data_format` | 输入 | 目标格式。默认 `None`，保留源格式；可指定 `"nd"`、`"nz"`、`"zn"`。 |

## 返回值说明

返回新的 Tensor。`dtype`、逻辑/物理布局和格式由参数及原 Tensor 确定；`shape` 和 `rank` 取自目标逻辑布局，存储层级保持不变。原 Tensor 的属性不变。

## 约束说明

- 原 Tensor 为组合视图，且本次调用保留原布局时，返回结果仍为组合视图；返回 Tensor 的 `stride`、`physical_format` 和 `data_format` 不支持读取，读取会报错；`physical_layout`、`physical_shape` 和 `physical_stride` 仍可读取。

- `layout` 和 `physical_layout` 必须为 `make_layout()` 创建的显式布局。
- 显式指定布局、格式或运行时偏移时，仅支持片上存储；GM 仅支持不指定布局和格式、偏移为静态值的形式。
- SSBUF 只使用 ND；NZ/ZN 的逻辑形状必须为二维或三维。
- 新视图必须满足底层存储容量和地址对齐要求。静态可确定的越界会报错，动态访问范围由调用者保证。

## 调用示例

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def example_kernel(y: cbd.Tensor):
    t = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((64,)))
    u = t.reinterpret(layout=cbd.make_layout((2, 8), stride=(16, 1)), offset=32)
    y[0] = u.shape[0]
    y[1] = u.shape[1]
    y[2] = u.stride[0]
    y[3] = u.stride[1]


@cbd.host
def run(y):
    example_kernel[1](y)


y = torch.empty(4, dtype=torch.int64, device="npu")
run(y)
torch.npu.synchronize()
print("shape:", y[:2].cpu().tolist())
print("stride:", y[2:].cpu().tolist())
```

输入：

```text
原 shape=(64,), dtype=float32
新 layout=make_layout((2, 8), stride=(16, 1)), offset=32
```

输出：

```text
shape: [2, 8]
stride: [16, 1]
```
