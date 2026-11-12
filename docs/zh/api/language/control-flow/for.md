# for

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

`for`按给定范围或固定序列重复执行循环体。

## 语法形式

```python
for i in iterable:
    ...
else:
    ...
```

循环正常结束后执行可选的`else`语句块。编译期循环通过`break`退出时，不执行`else`。

## 迭代方式

| 写法 | 用途 |
| --- | --- |
| [range](./for/range.md) | 指定循环起点、终点和步长。 |
| [cannbotdsl.range](./for/dsl-range.md) | 指定循环范围，并可设置`unroll`或`unroll_full`。 |
| [range_constexpr](./for/range-constexpr.md) | 指定编译期展开的循环范围。 |
| [元组和列表](./for/sequences.md) | 遍历固定的元素序列。 |

## 约束说明

- 运行时循环不支持`break`、`continue`，循环体内不支持`return`、`raise`。嵌套的编译期循环可使用`break`、`continue`控制自身循环。
- 运行时循环体不支持推导式、嵌套函数或类、`lambda`、`try`、`yield`、异步语法、`del`和`match`。

变量初始化、类型和作用域的错误示例见[控制流约束](./control-flow.md#约束)。

## 调用示例

以下示例对比普通`for`循环和`for ... else`，计算0至3的和。带`else`的循环在求和后再加100。

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def for_kernel(out: cbd.Tensor):
    total = 0
    for i in range(4):
        total = total + i
    out[0] = total

    total = 0
    for i in range(4):
        total = total + i
    else:
        total = total + 100
    out[1] = total


@cbd.host
def run(out):
    for_kernel[1](out)


def main():
    out = torch.zeros(2, dtype=torch.int64, device="npu")
    run(out)
    result = out.cpu().tolist()
    assert result == [6, 106]
    print(result)


if __name__ == "__main__":
    main()
```

预期结果：

```text
[6, 106]
```
