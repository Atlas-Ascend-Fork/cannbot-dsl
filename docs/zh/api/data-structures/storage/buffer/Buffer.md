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
| `mem_loc` | 输入 | 存储层级。必填，可选值：`MemLoc.UB`、`MemLoc.L1`、`MemLoc.L0A`、`MemLoc.L0B`、`MemLoc.L0C`、`MemLoc.BIAS`、`MemLoc.SSBUF`、`MemLoc.FBUF`、`MemLoc.FBUF_ACT`。具体说明见[存储位置](../../tensor/properties/memloc.md)。 |
| `dtype` | 输入 | 元素类型，必填。可指定的类型见[数据类型](../../data-types.md)；元素类型须为具有完整字节存储宽度的标量类型，或为支持的低位打包类型。 |
| `layout` | 输入 | 逻辑形状和坐标跨度。必填，传入 [make_layout()](../../tensor/make_layout.md) 返回的 `Layout`。 |
| `physical_layout` | 输入 | 物理存储的形状和跨度。默认 `None`，根据逻辑布局、元素类型和格式推导；也可传入 `make_layout()` 返回的 `Layout`，指定预留空间或物理跨度。 |
| `data_format` | 输入 | 存储格式。可选值为 `None`、`"nd"`、`"nz"`。默认 `None`：L1、L0A、L0B、L0C 使用 NZ，其余支持的层级使用 ND。 |

## 返回值说明

返回新分配存储上的 Tensor。`dtype`、`memloc` 取自参数；逻辑形状由 `layout` 指定，物理形状和跨度由 `physical_layout` 或默认推导结果确定。

## 约束说明

- 分配的存储容量须在编译时确定。例如，逻辑形状为`(n,)`且`n`在执行时确定时，可用`physical_layout=make_layout((128,))`预留128个元素的空间；调用方须保证`n`不超过128。
- BIAS、SSBUF、FBUF 只使用 ND；BIAS 使用一维、32 位元素。
- 对同一个存储层级，普通分配不能与[片上存储空间](./memory-spaces.md)混用。

## 调用示例

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def example_kernel(y: cbd.Tensor):
    t = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((32,)))
    u = cbd.from_tensor(t, layout=cbd.make_layout((2, 8)), offset=32)
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
print("视图 shape:", y[:2].cpu().tolist())
print("视图 stride:", y[2:].cpu().tolist())
```

输入：

```text
Buffer: mem_loc=UB, dtype=float32, layout=make_layout((32,))
新视图 layout=make_layout((2, 8)), offset=32
```

输出：

```text
视图 shape: [2, 8]
视图 stride: [8, 1]
```
