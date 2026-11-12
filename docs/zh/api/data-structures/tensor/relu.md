# Tensor.relu

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

为L0C中的Tensor指定ReLU操作，返回描述该操作的Tensor视图。该视图可作为`mem_copy`的源，在搬出时对数据应用ReLU。

## 函数原型

```python
relu(self)
```

## 参数说明

无。

## 返回值说明

返回带ReLU操作的Tensor视图。

## 约束说明

视图的使用须满足L0C搬出通路的要求，具体见[mem_copy](../../data-movement/mem-copy.md)。

## 调用示例

以下示例计算矩阵乘，对结果应用ReLU后搬出，并校验结果。

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def example_kernel(a: cbd.Tensor, b: cbd.Tensor, out: cbd.Tensor):
    layout = cbd.make_layout((128, 128))
    a_l1 = cbd.Buffer(cbd.MemLoc.L1, cbd.dtypes.float16, layout)
    b_l1 = cbd.Buffer(cbd.MemLoc.L1, cbd.dtypes.float16, layout)
    a_l0 = cbd.Buffer(cbd.MemLoc.L0A, cbd.dtypes.float16, layout)
    b_l0 = cbd.Buffer(cbd.MemLoc.L0B, cbd.dtypes.float16, layout)
    acc = cbd.Buffer(cbd.MemLoc.L0C, cbd.dtypes.float32, layout)
    cbd.mem_copy(a_l1, a)
    cbd.mem_copy(b_l1, b)
    cbd.mem_copy(a_l0, a_l1)
    cbd.mem_copy(b_l0, b_l1)
    cbd.matmul(acc, a_l0, b_l0, init=True)
    cbd.mem_copy(out, acc.relu())


@cbd.host
def run(a, b, out):
    example_kernel[1](a, b, out)


a = torch.full((128, 128), -1, dtype=torch.float16, device="npu")
b = torch.eye(128, dtype=torch.float16, device="npu")
out = torch.empty((128, 128), dtype=torch.float32, device="npu")
run(a, b, out)
torch.testing.assert_close(out.cpu(), torch.zeros((128, 128), dtype=torch.float32))
```
