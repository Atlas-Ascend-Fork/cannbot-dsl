# 遍历元组和列表

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

`for`可按固定元组或列表中的元素展开循环。元素本身也是元组或列表时，可用多个循环变量分别接收其中的值。

## 语法形式

```python
for left, right in ((1, 2), (3, 4)):
    ...
```

## 约束说明

循环中的`break`、`continue`遵守编译期循环规则。在运行时控制流内嵌套这种循环时，它们只能控制内层循环。其他作用域限制见[控制流约束](../control-flow.md#约束)。

## 调用示例

以下示例分别遍历嵌套元组和列表，计算其中数值的和。

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def sequence_kernel(out: cbd.Tensor):
    total = 0
    for left, right in ((1, 2), (3, 4)):
        total = total + left + right
    out[0] = total

    total = 0
    for value in [1, 2, 3]:
        total = total + value
    out[1] = total


@cbd.host
def run(out):
    sequence_kernel[1](out)


def main():
    out = torch.zeros(2, dtype=torch.int64, device="npu")
    run(out)
    result = out.cpu().tolist()
    assert result == [10, 6]
    print(result)


if __name__ == "__main__":
    main()
```

预期结果：

```text
[10, 6]
```
