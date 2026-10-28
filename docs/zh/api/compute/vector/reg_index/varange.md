# varange
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

以传入的`start`为起始值，生成递增/递减的索引，将生成的索引作为返回值返回，`Vector Length (VL)`表示矢量数据寄存器的位宽，`VL_T`表示该寄存器可存储的元素数量。算法逻辑表示如下：

```python
# 递增
increased = [start + i for i in range(VL_T)]
# 递减
decreased = [start + VL_T - 1 - i for i in range(VL_T)]
```

以上代码仅说明元素排列顺序，不模拟具体元素数据类型的溢出和舍入行为。

以`dtypes.int16`数据类型，起始值`start=10`为例：
递增索引为{10, 11, 12, 13, ... 135, 136, 137}，递减索引为{137, 136, 135, 134, ... 12, 11, 10}。

## 函数原型

```python
varange(start, dtype, *, order_mode: 'str | OrderMode' = OrderMode.INCREASED) -> RawVReg
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `start` | 输入 | 源操作数（标量），数据类型须与`dtype`一致。作为递增序列的起点时，序列第0个元素等于`start`，后续元素按1递增；作为递减序列的起点时，序列第0个元素等于`start + VL_T - 1`，后续元素按1递减。取值范围为该dtype的可表示范围。 |
| `dtype` | 输入 | 生成索引的元素类型。 |
| `order_mode` | 输入 | 索引排列方向，可传入`OrderMode`枚举成员，或字符串`"increased"`（递增）、`"decreased"`（递减），默认值为`OrderMode.INCREASED`。 |

**支持的元素数据类型：**

参数`dtype`指定生成索引的元素数据类型，支持`dtypes.int8`、`dtypes.int16`、`dtypes.float16`、`dtypes.int32`、`dtypes.float32`。

## 返回值说明

返回保存生成索引的矢量数据寄存器值，类型为`RawVReg`，其中元素的数据类型由`dtype`指定。

## 约束说明

- 本接口需在`vf(mode="simd")`作用域内调用。

- 整型dtype（`dtypes.int8`、`dtypes.int16`、`dtypes.int32`）结果在超出该dtype可表示范围时回绕（wrap-around），不触发异常。例如`dtypes.int8`取`start=127`时，序列前128个元素依次为127、−128、−127、…、-2；`start=−128`时，序列前128个元素依次为−128、−127、…、−1。

## 调用示例

以下示例演示`varange`的调用，并在Host侧校验结果。

```python
import torch
import torch_npu

from cannbotdsl import Buffer, dtypes, host, make_layout, mem_copy
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.ops.reg import full_mask, varange, vstore
from cannbotdsl.tensor import MemLoc

@kernel
def _varange_kernel(dst):
    out = Buffer(MemLoc.UB, dst.dtype, make_layout((64,)))

    res = out
    with vf(mode="simd"):
        mask = full_mask()
        vstore(res, 0, varange(5, dtypes.int32), mask)

    mem_copy(dst, out)

@host
def run(dst):
    _varange_kernel[1](dst)

def main():
    dst = torch.empty((64,), dtype=torch.int32, device="npu:0")

    run(dst)
    torch.npu.synchronize()

    torch.testing.assert_close(dst.cpu(), torch.arange(5, 69, dtype=torch.int32))
    print("varange example passed")
    print(f"first={int(dst.cpu()[0])}, last={int(dst.cpu()[-1])}")

if __name__ == "__main__":
    main()
```

预期结果：

```text
varange example passed
first=5, last=68
```
