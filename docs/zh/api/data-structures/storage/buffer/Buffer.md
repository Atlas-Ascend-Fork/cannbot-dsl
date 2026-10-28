# Buffer 构造函数

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

根据存储层级、元素类型和布局分配片上存储，返回访问该存储的 Tensor。分配容量由物理布局确定，包含跨度和填充占用的空间。

## 函数原型

```python
Buffer(mem_loc, dtype, layout, *, physical_layout=None, data_format=None) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| `mem_loc` | 输入 | 存储层级。必填，可选值：`MemLoc.UB`、`MemLoc.L1`、`MemLoc.L0A`、`MemLoc.L0B`、`MemLoc.L0C`、`MemLoc.BIAS`、`MemLoc.SSBUF`、`MemLoc.FBUF`。具体说明见[存储位置枚举类](../memory-spaces.md)。 |
| `dtype` | 输入 | 元素类型。必填，可选值：`dtypes.float16`、`dtypes.float32`、`dtypes.float64`、`dtypes.bfloat16`、`dtypes.float8_e4m3fn`、`dtypes.float8_e5m2`、`dtypes.float8_e8m0`、`dtypes.hifloat8`、`dtypes.int8`、`dtypes.int16`、`dtypes.int32`、`dtypes.int64`、`dtypes.uint8`、`dtypes.uint16`、`dtypes.uint32`、`dtypes.uint64`、`dtypes.fp4x2_e1m2`、`dtypes.fp4x2_e2m1`、`dtypes.int4x2`。具体说明见[预定义数据类型](../../data-types.md)。 |
| `layout` | 输入 | 逻辑形状和坐标跨度。必填，传入 [make_layout()](../../layout/make_layout.md) 返回的 `Layout`。 |
| `physical_layout` | 输入 | 物理存储的形状和跨度。默认 `None`，根据逻辑布局、元素类型和格式推导；也可传入 `make_layout()` 返回的 `Layout`，指定预留空间或物理跨度。 |
| `data_format` | 输入 | 存储格式。可选值为 `None`、`"nd"`、`"nz"`。默认 `None`：L1、L0A、L0B、L0C 使用 NZ，其余支持的层级使用 ND。 |

## 返回值说明

返回新分配存储上的 Tensor。`dtype`、`memloc` 取自参数；逻辑形状由 `layout` 指定，物理形状和跨度由 `physical_layout` 或默认推导结果确定。

## 约束说明

- 物理分配容量必须能静态确定；逻辑形状包含运行时长度时，应提供足够大的静态物理布局，并保证运行时访问不越界。
- BIAS、SSBUF、FBUF 只使用 ND；BIAS 使用一维、32 位元素。
- 对同一个存储层级，普通分配不能与[片上存储空间](../on-chip-memory.md)混用。

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
