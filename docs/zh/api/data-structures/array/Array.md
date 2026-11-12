# Array 构造函数

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

创建包含 `length` 个同类型标量的数组，可通过 `init` 指定初始值。

## 函数原型

```python
Array(dtype, length, *, init=None)
```

## 参数说明

| 参数名 | 输入/输出 | 说明 |
| --- | --- | --- |
| `dtype` | 输入 | 元素类型。必填，可选值：`dtypes.bool`、`dtypes.int8`、`dtypes.uint8`、`dtypes.int16`、`dtypes.uint16`、`dtypes.int32`、`dtypes.uint32`、`dtypes.int64`、`dtypes.uint64`、`dtypes.float16`、`dtypes.bfloat16`、`dtypes.float32`、`dtypes.float64`。具体说明见[数据类型](../data-types.md)。 |
| `length` | 输入 | 元素数量。必填，为 `1`～`2^63-1` 的 Python 整数。 |
| `init` | 输入 | 初始值。默认 `None`，不初始化元素；也可传入长度等于 `length` 的 list 或 tuple，每项为符合 dtype 的 Python 常量。 |

## 返回值说明

返回可按下标读取和写入的 Array，`length` 为声明长度。

## 约束说明

- 数组须在 Kernel 或其调用的 jit 中创建，不能在 VF/SIMT 区域中创建或访问。
- 在分支或循环外仍需访问的数组，应在进入分支或循环前创建。

## 调用示例

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def example_kernel(y: cbd.Tensor):
    a = cbd.Array(cbd.dtypes.int64, 4, init=[1, 2, 3, 4])
    a[1] = 9
    for i in cbd.range(4):
        y[i] = a[i]


@cbd.host
def run(y):
    example_kernel[1](y)


y = torch.empty(4, dtype=torch.int64, device="npu")
run(y)
torch.npu.synchronize()
print(y.cpu().tolist())
```

输入：

```text
dtype=int64, length=4, init=[1, 2, 3, 4]
a[1]=9
```

输出：

```text
[1, 9, 3, 4]
```
