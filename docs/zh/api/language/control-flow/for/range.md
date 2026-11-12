# range

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

为`for`循环指定起点、终点和步长。

## 函数原型

参数支持Python整数或运行时整数，调用形式如下：

```python
range(stop, /)
range(start, stop, step=1, /)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `start` | 输入 | 循环起点，支持Python整数或运行时整数，单位为循环计数；省略时为0。 |
| `stop` | 输入 | 循环终点，不包含`stop`，必选，无默认值。支持Python整数或运行时整数，单位为循环计数。 |
| `step` | 输入 | 步长，支持Python整数或运行时整数，必须为正值，单位为循环计数；默认值为1。 |

## 返回值说明

在DSL函数中用于指定`for`循环的起点、终点和步长。

## 约束说明

变量初始化、类型、作用域及提前退出的错误示例见[控制流约束](../control-flow.md#约束)。

## 调用示例

以下示例分别使用终点、起止点和步长指定循环范围，计算各范围内整数的和。

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def range_kernel(out: cbd.Tensor):
    total = 0
    for i in range(4):
        total = total + i
    out[0] = total

    total = 0
    for i in range(2, 5):
        total = total + i
    out[1] = total

    total = 0
    for i in range(0, 6, 2):
        total = total + i
    out[2] = total


@cbd.host
def run(out):
    range_kernel[1](out)


def main():
    out = torch.zeros(3, dtype=torch.int64, device="npu")
    run(out)
    result = out.cpu().tolist()
    assert result == [6, 9, 6]
    print(result)


if __name__ == "__main__":
    main()
```

预期结果：

```text
[6, 9, 6]
```
