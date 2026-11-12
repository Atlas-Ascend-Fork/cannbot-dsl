# tile_slice_multi

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

按固定块形状、重复次数和块起点间距描述多个完整切块。接口返回视图，不立即搬运数据；可将该视图传给`mem_copy`。

## 函数原型

```python
tile_slice_multi(input: Tensor, block_shape, *, repeats, step)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `input` | 输入 | 源Tensor或目标Tensor。 |
| `block_shape` | 输入 | 每个块的形状，维数与`input`一致，各维为编译期正整数。 |
| `repeats` | 输入 | 重复次数，为正整数或包含一至两个正整数的序列。 |
| `step` | 输入 | 相邻块起点在各维的元素间距；每个重复维度对应一组非负整数间距。源端的0间距表示重复读取同一块。 |

## 返回值说明

返回描述多个块的Tensor视图。返回形状由重复次数和块形状组成。

## 约束说明

- 重复次数、块形状和间距须为编译期整数，所有完整块须位于输入Tensor的访问范围内。
- 用于`mem_copy`时，源和目标的块形状、重复次数须一致；目标块能否重叠由调用方保证，并须满足相应搬运通路的要求。

## 调用示例

以下示例将32个元素分成两个块，从GM搬到UB后写回，并校验结果。

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def kernel(x: cbd.Tensor, y: cbd.Tensor):
    ub = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float16, cbd.make_layout((32,)))
    src = cbd.tile_slice_multi(x, (16,), repeats=2, step=(16,))
    dst = cbd.tile_slice_multi(ub, (16,), repeats=2, step=(16,))
    cbd.mem_copy(dst, src)
    cbd.mem_copy(y, ub)


@cbd.host
def run(x, y):
    kernel[1](x, y)


x = torch.arange(32, dtype=torch.float16, device="npu")
y = torch.empty_like(x)
run(x, y)
torch.testing.assert_close(y.cpu(), x.cpu())
```
