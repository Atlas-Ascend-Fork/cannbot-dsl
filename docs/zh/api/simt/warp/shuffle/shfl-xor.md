# simt.shfl_xor

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

读取当前分组内Lane编号为`LaneId ^ lane_mask`的线程持有的`value`，其中`LaneId`为当前线程在Warp内的编号。`width`将Warp划分为大小相同的分组；在`0 <= lane_mask < width`的范围内，交换只发生在当前分组中。

例如，Warp内32个活跃线程调用`simt.shfl_xor(LaneId, 1, 16)`时，每个线程读取同组内Lane编号为`LaneId ^ 1`的线程持有的值，即相邻两个线程交换数据。

![simt.shfl_xor结果示意图](figures/shfl-xor-example.svg)

## 函数原型

```python
simt.shfl_xor(
    value: ScalarValue | float16x2 | bfloat16x2 | int | float,
    lane_mask: int32 | int,
    width: int = 32,
) -> ScalarValue | float16x2 | bfloat16x2
```

`ScalarValue`为公开标量类型，支持的数据类型见参数说明。

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| value | 输入 | 待交换的值。标量支持`int32`、`int64`、`float16`、`bfloat16`、`float32`；短向量支持`float16x2`（两个`float16`分量）、`bfloat16x2`（两个`bfloat16`分量），两个分量来自同一个源线程。也接受Python整数或浮点数，分别按`int32`、`float32`处理。 |
| lane_mask | 输入 | 与当前线程Lane编号进行异或的掩码，类型为`int32`标量或Python整数，取值范围为`0 <= lane_mask < width`。 |
| width | 输入 | 分组的线程数，必须是编译期Python整数，可选`1`、`2`、`4`、`8`、`16`、`32`，默认`32`。 |

## 返回值说明

返回指定源线程的`value`，数据类型与输入相同。输入为短向量时，返回相同类型的短向量。

## 约束说明

- 如果目标线程是非活跃状态，获取到寄存器中未初始化的值。
- `width`不满足参数说明中的类型或取值要求时，抛出`ValueError`。

## 调用示例

在一个Warp中调用`simt.shfl_xor`，校验结果并打印前8个线程的输出。

```python
import torch
import torch_npu
from cannbotdsl import dim3, host, simt
from cannbotdsl.lang.kernel import kernel
from cannbotdsl.lang.vf import vf
from cannbotdsl.tensor import Tensor


@kernel
def example_kernel(src: Tensor, dst: Tensor):
    with vf(mode='simt'):
        i = simt.thread_idx()[0]
        value = src[i] * 2
        dst[i] = simt.shfl_xor(value, 1)


@host
def run(src, dst):
    example_kernel[dim3(1), dim3(32)](src, dst)


def main():
    src = torch.arange(32, dtype=torch.int32, device='npu')
    dst = torch.empty((32,), dtype=torch.int32, device='npu')
    run(src, dst)
    torch.npu.synchronize()
    expected = torch.tensor([2 * (i ^ 1) for i in range(32)], dtype=torch.int32)
    assert torch.equal(dst.cpu(), expected)
    print(dst.cpu().tolist()[:8])


if __name__ == "__main__":
    main()
```

预期结果：

```text
[2, 0, 6, 4, 10, 8, 14, 12]
```
