# @jit

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

`@jit`用于声明可复用的DSL辅助函数。辅助函数可由Host函数、核函数或其他`@jit`函数调用，并沿用调用者的执行环境；函数体中可使用的接口取决于调用位置。

## 定义与调用辅助函数

在函数定义前添加`@cbd.jit`，即可在DSL函数中调用。以下示例使用辅助函数计算索引偏移，并在核函数中写回结果。

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.jit
def shifted_index(index, offset):
    return index + offset


@cbd.kernel
def write_index(out: cbd.Tensor):
    index = cbd.get_block_idx()
    value = shifted_index(index, 7)
    cbd.scalar.vec_store_bypass(out.ptr(index), value)


@cbd.host
def run(out):
    write_index[1](out)


out = torch.full((1,), -1, dtype=torch.int64, device="npu")
run(out)
assert out.cpu().tolist() == [7]
```

辅助函数可以返回计算结果。

## 在辅助函数中复用其他辅助函数

一个`@jit`函数可以调用另一个`@jit`函数。以下示例由`current_value`调用`shifted_index`，核函数调用`current_value`获取待写入的值。

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.jit
def shifted_index(index, offset):
    return index + offset


@cbd.jit
def current_value(index):
    return shifted_index(index, 7)


@cbd.kernel
def write_index(out: cbd.Tensor):
    index = cbd.get_block_idx()
    value = current_value(index)
    cbd.scalar.vec_store_bypass(out.ptr(index), value)


@cbd.host
def run(out):
    write_index[1](out)


out = torch.full((1,), -1, dtype=torch.int64, device="npu")
run(out)
assert out.cpu().tolist() == [7]
```

## 调用位置

`@jit`函数须由DSL函数调用，不支持直接从普通Python代码调用。
