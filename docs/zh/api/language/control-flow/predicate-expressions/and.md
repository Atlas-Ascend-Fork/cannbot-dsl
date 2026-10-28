# and

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

`and`左侧为真时继续计算右侧，左侧为假时短路。

真假判断规则见[判断条件概览](../predicate-expressions.md#真假判断规则)。

## 语法

```python
result = a and b
```

编译期已确定短路时，右侧不参与追踪；短路条件需在运行时判断时，右侧仍需追踪和编译，但设备只在需要时执行右侧。

## 返回值说明

仅涉及Python值时，按Python规则返回选中的操作数；涉及运行时值时，返回Python值或运行时布尔值。

## 约束说明

Python值须可按Python真假规则判断；运行时操作数须是可转换为条件的DSL标量。

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
    result = a != 0 and b != 0
    out[0] = 1 if result else 0


class PredicateOp:
    @cbd.host
    def run(self, out, a: cbd.dtypes.int64, b: cbd.dtypes.int64):
        predicate_kernel[1](out, a, b)


def main():
    cases = ((1, 0, [0]), (5, 2, [1]), (0, 0, [0]))
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
a=1, b=0 -> [0]
a=5, b=2 -> [1]
a=0, b=0 -> [0]
```
