# Tensor.expand

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

根据指定的目标形状广播当前 Tensor，返回新的 Tensor 视图。支持以下两种扩展方式，可同时使用：

- 扩展原有轴：将原形状中长度为 `1` 的任意轴扩展到目标长度，不限于最外层轴。
- 新增轴：在原形状的最前面增加一个或多个轴，各新增轴的长度由目标形状指定。

原有轴与目标形状从最后一轴开始对应；原有轴的长度不为 `1` 时，对应的目标长度必须保持不变。沿扩展轴或新增轴访问不同位置时，访问的是原 Tensor 中的同一份数据。

## 函数原型

```python
expand(self, *sizes, size=None) -> Tensor
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| `sizes` | 输入 | 目标形状，可逐维传入整数或运行时整数值，也可传入一个元组或列表。已有轴可使用 `-1` 保留原长度。 |
| `size` | 输入 | 可选，使用关键字传入目标形状，与 `sizes` 不能同时指定。 |

## 返回值说明

返回具有目标形状的新Tensor视图，元素类型不变，与原Tensor共享存储。原数据的存放方式保持不变。

例如，一个[NZ格式](./properties/data-format.md)的Tensor形状为`(1, 32)`，执行`expand(4, 32)`后，新视图的4行都访问原来的那一行，数据仍按原格式存放。

## 约束说明

- 对[NZ、ZN格式](./properties/data-format.md)的Tensor，或原本不支持读取`stride`的视图执行扩展后，返回结果不支持读取`stride`和`data_format`。需要查看数据的实际存储布局时，读取`physical_layout`。
- 不支持包含嵌套元组的形状，例如`((2, 3), 8)`。
- 目标维数不得小于 `self.rank`。
- 新增轴不能使用 `-1`。运行时输入也必须满足形状扩展条件。

## 调用示例

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def example_kernel(y: cbd.Tensor):
    t = cbd.Buffer(cbd.MemLoc.UB, cbd.dtypes.float32, cbd.make_layout((1, 8)))
    u = t.expand((4, 8))
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
原 shape=(1, 8), 目标 shape=(4, 8)
```

输出：

```text
shape: [4, 8]
stride: [0, 1]
```
