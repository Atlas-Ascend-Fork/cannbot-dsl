# @host

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

`@host`用于声明Host函数，组织核函数调用，并作为Python调用入口。Host函数的返回值应为`None`，计算结果通过传入的Tensor写回。

## 定义与调用Host函数

在函数定义前添加`@cbd.host`，然后使用运行时参数调用该函数。以下示例通过`run`启动核函数，并读取输出检查结果。

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.jit
def current_index():
    return cbd.get_block_idx()


@cbd.kernel
def write_index(out: cbd.Tensor):
    index = current_index()
    cbd.scalar.vec_store_bypass(out.ptr(index), index)


@cbd.host
def run(out):
    write_index[1](out)


out = torch.full((1,), -1, dtype=torch.int64, device="npu")
run(out)
assert out.cpu().tolist() == [0]
```

核函数启动语句直接写在Host函数体中。

## 提前编译Host函数

使用`cannbotdsl.compile`和参数规格，可以提前编译Host函数，再使用实际Tensor调用编译结果。

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.jit
def current_index():
    return cbd.get_block_idx()


@cbd.kernel
def write_index(out: cbd.Tensor):
    index = current_index()
    cbd.scalar.vec_store_bypass(out.ptr(index), index)


@cbd.host
def run(out):
    write_index[1](out)


out = torch.full((1,), -1, dtype=torch.int64, device="npu")
spec = cbd.TensorSpec((1,), cbd.dtypes.int64)
program = cbd.compile(run, spec)
program(out)
program.close()
assert out.cpu().tolist() == [0]
```

Tensor参数规格的写法见[TensorSpec](../compilation/host/specs/tensor_spec.md)。
