# any

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

`any()`在任一条件为真时返回真，对各元素的真假值进行逻辑或运算。

## 函数原型

```python
any(iterable: Iterable[Any], /) -> bool
```

`Iterable`、`Any`来自`typing`。以上为Python内建函数的签名；DSL动态形式还可能返回运行时布尔值，具体见返回值说明。

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `iterable` | 输入 | 必选，无默认值。动态形式使用固定tuple/list表达式，元素为Python值或运行时布尔、整数、浮点标量；不支持生成器、运行时可迭代对象或非标量运行时元素。普通Python调用不受此限制。 |

真假判断规则见[判断条件概览](../predicate-expressions.md#真假判断规则)。

先计算tuple/list中的所有元素，再进行真假判断。即使某个元素已经决定结果，其他元素表达式也会求值；这与`and`/`or`逐项短路求值不同。

## 返回值说明

全部元素为Python值时返回Python `bool`。涉及运行时条件时，根据短路结果返回Python布尔值或运行时布尔值。例如，`any((True, runtime_condition))`的结果为Python `True`。

## 约束说明

在DSL运行时分支或循环内，`any()`即使仅包含纯Python元素，也须满足参数说明中的固定容器表达式要求。

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
    result = any((a != 0, b != 0))
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
