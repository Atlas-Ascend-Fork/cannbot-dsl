# cv_valid_extent

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

查询Tensor按指定轴进行两路分区时，某一路分区的有效长度。可与相同轴和对齐设置的`Tensor.partition`配合使用。

## 函数原型

```python
cv_valid_extent(tensor, *, axis, alignment=1, part_id)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `tensor` | 输入 | 需要查询分区有效长度的Tensor。 |
| `axis` | 输入 | 分区轴，取值为`0`或`1`。 |
| `alignment` | 输入 | 分区对齐粒度，默认为`1`。 |
| `part_id` | 输入 | 分区编号，取值为`0`或`1`，可为整数或运行时整数。 |

## 返回值说明

返回指定分区沿`axis`方向的有效元素数量。

## 约束说明

- `tensor`须为二维Tensor。
- `alignment`须为正整数；`axis=1`时只能为`1`。
- 与`Tensor.partition`配合时，查询和分区须使用相同的轴与对齐粒度。

## 调用示例

以下示例查询长度为16的轴在两路分区中的有效长度。

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def kernel(out: cbd.Tensor):
    tile = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float16, cbd.make_layout((16, 32)))
    out[0] = cbd.cv_valid_extent(tile, axis=0, part_id=0)
    out[1] = cbd.cv_valid_extent(tile, axis=0, part_id=1)


@cbd.host
def run(out):
    kernel[1](out)


out = torch.zeros((2,), dtype=torch.int64, device="npu")
run(out)
assert out.cpu().tolist() == [8, 8]
```
