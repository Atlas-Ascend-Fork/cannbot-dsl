# RingBuffer 构造函数

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

按相同元素类型和布局分配 `depth` 个片上存储槽位，建立独立的读写游标。每个槽位的分配容量由物理布局确定。

## 函数原型

```python
RingBuffer(
    mem_loc,
    dtype,
    layout,
    *,
    depth,
    physical_layout=None,
    data_format=None,
) -> RingBuffer
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| `mem_loc` | 输入 | 槽位所在的存储层级。必填，可选值：`MemLoc.UB`、`MemLoc.L1`、`MemLoc.L0A`、`MemLoc.L0B`、`MemLoc.L0C`、`MemLoc.BIAS`。具体说明见[存储位置枚举类](../memory-spaces.md)。 |
| `dtype` | 输入 | 槽位的元素类型。必填，可选值：`dtypes.float16`、`dtypes.float32`、`dtypes.float64`、`dtypes.bfloat16`、`dtypes.float8_e4m3fn`、`dtypes.float8_e5m2`、`dtypes.float8_e8m0`、`dtypes.hifloat8`、`dtypes.int8`、`dtypes.int16`、`dtypes.int32`、`dtypes.int64`、`dtypes.uint8`、`dtypes.uint16`、`dtypes.uint32`、`dtypes.uint64`、`dtypes.fp4x2_e1m2`、`dtypes.fp4x2_e2m1`、`dtypes.int4x2`。具体说明见[预定义数据类型](../../data-types.md)。 |
| `layout` | 输入 | 逻辑形状和坐标跨度。必填，传入 [make_layout()](../../layout/make_layout.md) 返回的 `Layout`。 |
| `depth` | 输入 | 槽位数量。必填，为 Python 正整数。 |
| `physical_layout` | 输入 | 物理存储的形状和跨度。默认 `None`，根据逻辑布局、元素类型和格式推导；也可传入 `make_layout()` 返回的 `Layout`，指定预留空间或物理跨度。 |
| `data_format` | 输入 | 存储格式。可选值为 `None`、`"nd"`、`"nz"`。默认 `None`：L1、L0A、L0B、L0C 使用 NZ，UB、BIAS使用 ND。 |

## 返回值说明

返回 RingBuffer 对象。所有槽位采用指定的元素类型和布局，`depth` 等于参数值，读写游标均从第一个槽位开始。

成功构造后，各公开属性的读取范围如下：

| 属性 | 是否可读取 |
| --- | --- |
| `dtype`、`memloc`、`depth`、`shape`、`layout`、`physical_layout`、`physical_shape`、`physical_stride`、`layout_map_kind`、`logical_format`、`physical_format`、`data_format` | 可以直接读取。 |
| `stride` | ND 布局下可以读取；NZ 布局下不支持读取逻辑跨度，读取时会报错，可通过 `physical_stride` 查看物理跨度。 |

## 约束说明

- 物理分配容量必须能静态确定；动态逻辑长度须在预留物理空间内。
- BIAS 使用一维、32 位元素及 ND 格式。
- 同一层级不能同时使用普通分配和片上存储空间。

## 调用示例

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def example_kernel(y: cbd.Tensor):
    ring = cbd.RingBuffer(cbd.MemLoc.UB, cbd.dtypes.int64, cbd.make_layout((4,)), depth=2)
    w0 = ring.next_write()
    w0[0] = 11
    w1 = ring.next_write()
    w1[0] = 22
    r0 = ring.next_read()
    r1 = ring.next_read()
    r2 = ring.next_read()
    y[0] = r0[0]
    y[1] = r1[0]
    y[2] = r2[0]


@cbd.host
def run(y):
    example_kernel[1](y)


y = torch.empty(3, dtype=torch.int64, device="npu")
run(y)
torch.npu.synchronize()
print(y.cpu().tolist())
```

输入：

```text
depth=2, 单槽 shape=(4,), dtype=int64
第 0、1 槽首元素分别写入 11、22，连续选槽三次
```

输出：

```text
[11, 22, 11]
```
