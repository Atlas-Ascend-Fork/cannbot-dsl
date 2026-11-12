# 链式比较

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

链式比较支持`<`、`<=`、`>`、`>=`、`==`、`!=`。例如`a < b < c`等价于
`a < b and b < c`，但中间操作数`b`只计算一次。在`a < x[0] < b`中，只读取一次
`x[0]`，再用同一个结果完成前后两次比较。

参与比较的操作数可以是Python数值或DSL标量。相邻操作数的数据类型必须兼容。需要明确比较类型时，可先显式转换，例如
`cbd.dtypes.float32(a) < b < c`，其中`b`、`c`也使用`float32`。

## 语法

```python
result = a < b <= c < d
```

## 约束说明

参与比较的运行时值必须是单个数值，不能直接使用整个Tensor。需要比较Tensor中的数据时，先取出元素，例如`0 < x[0] < 10`。

## 调用示例

以下示例在Host侧校验三组输入的判断结果，真输出1，假输出0。

```python
import cannbotdsl as cbd
from cannbotdsl.tensor import Tensor
import torch
import torch_npu  # noqa: F401


@cbd.kernel
def predicate_kernel(out: Tensor, a, b):
    # 计算条件，并将真、假分别转换为1、0写入输出。
    result = 0 < a < 10
    out[0] = 1 if result else 0


class PredicateOp:
    @cbd.host
    def run(self, out, a: cbd.dtypes.int64, b: cbd.dtypes.int64):
        predicate_kernel[1](out, a, b)


def main():
    cases = ((1, 0, [1]), (5, 2, [1]), (0, 0, [0]))
    for a, b, expected in cases:
        out = torch.zeros(1, dtype=torch.int64, device="npu:0")
        PredicateOp().run(out, a, b)
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), torch.tensor(expected, dtype=torch.int64))
        print(f"a={a}, b={b} -> {out.cpu().tolist()}")


if __name__ == "__main__":
    main()
```

预期结果：

```text
a=1, b=0 -> [1]
a=5, b=2 -> [1]
a=0, b=0 -> [0]
```
