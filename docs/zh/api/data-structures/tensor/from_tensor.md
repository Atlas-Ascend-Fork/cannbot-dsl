# from_tensor

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

返回与源Tensor共享存储的新视图，可指定目标元素类型、逻辑布局、物理布局和字节偏移。新视图起点为源Tensor起点加上`offset`。

## 函数原型

```python
from_tensor(
    input,
    dtype=None,
    layout=None,
    *,
    physical_layout=None,
    offset: int = 0,
    data_format=None,
) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| `input` | 输入 | 源Tensor，新视图共享该Tensor的存储。 |
| `dtype` | 输入 | 目标元素类型。默认 `None`，保留源类型；可指定的类型见[数据类型](../data-types.md)。元素类型须为具有完整字节存储宽度的标量类型，或为支持的低位打包类型。 |
| `layout` | 输入 | 目标逻辑布局。默认 `None`，沿用原逻辑布局；可传入 `make_layout()` 返回的 Layout，指定静态或运行时的形状、跨度。 |
| `physical_layout` | 输入 | 目标物理布局。默认 `None`；提供新逻辑布局时按目标格式推导，逻辑/物理布局与格式全部省略时保留原布局。可显式传入 Layout。 |
| `offset` | 输入 | 相对当前 Tensor 起点的字节偏移。默认 `0`，可为 `0`～`2^63-1` 的 Python 整数，或运行时非负整数值。 |
| `data_format` | 输入 | 目标格式。默认 `None`，保留源格式；可指定 `"nd"`、`"nz"`、`"zn"`。 |

## 返回值说明

返回新的 Tensor。`dtype`、逻辑/物理布局和格式由参数及原 Tensor 确定；`shape` 和 `rank` 取自目标逻辑布局，存储层级保持不变。原 Tensor 的属性不变。

## 约束说明

- 将`t.layout`作为`layout`或`physical_layout`参数时，`t`须支持读取[data_format](./properties/data-format.md)。通过[make_layout](./make_layout.md)创建的布局和`t.physical_layout`不受此限制。
- 调用方必须保证通过新视图访问的范围位于底层存储内；接口不检查容量。

## 调用示例

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def example_kernel(y: cbd.Tensor):
    t = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((64,)))
    u = cbd.from_tensor(t, layout=cbd.make_layout((2, 8), stride=(16, 1)), offset=32)
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
