# or

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

`or`表示至少一个条件成立。执行时先判断左侧，左侧为真时不再计算右侧。

## 语法

```python
result = a or b
```

例如，`x`为一维Tensor时，`x[0] > 0 or x[1] > 0`只有在第一个条件为假时才读取`x[1]`。

## 返回值说明

两侧均为常量时，左侧为真则返回左侧的值，否则返回右侧的值。数值0为假，非零数值为真。

| 写法 | 返回值 |
| --- | --- |
| `0 or 5` | `5` |
| `4 or 5` | `4` |
| `4 or 0` | `4` |

条件需要根据Tensor数据等运行时值判断时，返回`True`或`False`。

## 约束说明

运行时操作数须为单个布尔值、整数或浮点数，例如`x[0] > 0`。

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
    result = a != 0 or b != 0
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
