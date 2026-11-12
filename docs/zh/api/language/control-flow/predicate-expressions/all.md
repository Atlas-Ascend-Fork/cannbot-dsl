# all

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

`all()`仅在全部条件为真时返回真，对各元素的真假值进行逻辑与运算。

## 函数原型

```python
all(iterable, /)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `iterable` | 输入 | 由布尔值、数值或标量比较结果组成的元组或列表。 |

### 支持的调用形式

| 使用场景 | 写法 |
| --- | --- |
| 判断多个条件是否全部成立 | `all((a > 0, b > 0))`或`all([a > 0, b > 0])` |
| 判断Tensor指定元素是否都大于0 | `all((x[0] > 0, x[1] > 0))` |
| 判断Tensor指定元素是否都非零 | `all((x[0], x[1]))` |

## 返回值说明

返回`True`或`False`。

## 约束说明

条件取决于Tensor数据（如`x[0]`），或调用位于运行时分支、循环内时，须在调用处直接列出条件：

```python
result = all((x[0] > 0, x[1] > 0))  # 支持，也可使用列表。
```

此时不支持先将条件保存到变量，再将该变量传给`all()`：

```python
conditions = (x[0] > 0, x[1] > 0)
result = all(conditions)  # 不支持。
```

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
    result = all((a != 0, b != 0))
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
