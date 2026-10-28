# Storage.view

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

在当前句柄引用的存储上绑定 Tensor 视图。不传参数时返回默认视图；传入布局或类型时生成相应的新视图。

## 函数原型

```python
view(
    self,
    layout=None,
    *,
    dtype=None,
    physical_layout=None,
    data_format=None,
    offset=0,
) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| `layout` | 输入 | 目标逻辑布局。默认 `None`，使用默认视图的逻辑布局；也可传入 `make_layout()` 返回的 Layout。 |
| `dtype` | 输入 | 目标元素类型。默认 `None`，沿用默认视图的类型。可选值：`dtypes.float16`、`dtypes.float32`、`dtypes.float64`、`dtypes.bfloat16`、`dtypes.float8_e4m3fn`、`dtypes.float8_e5m2`、`dtypes.float8_e8m0`、`dtypes.hifloat8`、`dtypes.int8`、`dtypes.int16`、`dtypes.int32`、`dtypes.int64`、`dtypes.uint8`、`dtypes.uint16`、`dtypes.uint32`、`dtypes.uint64`、`dtypes.fp4x2_e1m2`、`dtypes.fp4x2_e2m1`、`dtypes.int4x2`。具体说明见[预定义数据类型](../../data-types.md)。 |
| `physical_layout` | 输入 | 目标物理布局。默认 `None`；指定新逻辑布局时按目标布局和格式推导。也可传入 Layout。 |
| `data_format` | 输入 | 目标格式。默认 `None`，沿用默认视图的格式；可指定 `"nd"`、`"nz"`、`"zn"`。 |
| `offset` | 输入 | 相对默认视图起点的字节偏移。默认 `0`，可为非负 Python 整数或运行时整数值。 |

## 返回值说明

返回访问同一存储的 Tensor，类型和布局由本次参数及默认视图确定；

## 约束说明

- 新视图须满足底层存储的容量和地址对齐要求。静态可确定的越界会被拒绝；运行时范围由调用者保证。
- SSBUF 只使用 ND；NZ/ZN 的逻辑形状为二维或三维。

## 调用示例

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def example_kernel(y: cbd.Tensor):
    t = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((32,)))
    s = t.bare()
    u = s.view(cbd.make_layout((2, 8)), offset=32)
    y[0] = s.size_bytes
    y[1] = s.alignment
    y[2] = u.shape[0]
    y[3] = u.shape[1]


@cbd.host
def run(y):
    example_kernel[1](y)


y = torch.empty(4, dtype=torch.int64, device="npu")
run(y)
torch.npu.synchronize()
print("存储字节数:", y[0].item())
print("对齐字节数:", y[1].item())
print("视图 shape:", y[2:].cpu().tolist())
```

输入：

```text
Buffer: mem_loc=UB, dtype=float32, layout=make_layout((32,))
新视图 layout=make_layout((2, 8)), offset=32
```

输出：

```text
存储字节数: 128
对齐字节数: 32
视图 shape: [2, 8]
```
