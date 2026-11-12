# cannbotdsl.range

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

用于设备执行的计数循环，可通过`unroll`或`unroll_full`设置循环展开方式。

## 函数原型

```python
cannbotdsl.range(
    *args,
    unroll: int | None = None,
    unroll_full: bool = False,
)
```

**普通循环**：不设置展开参数。

```python
cannbotdsl.range(stop)
cannbotdsl.range(start, stop)
cannbotdsl.range(start, stop, step)
```

**按展开因子展开**：设置`unroll=N`，向编译器提供每组展开`N`次迭代的优化提示。例如，`unroll=4`指定展开因子为4。该参数不保证消除循环。

```python
cannbotdsl.range(stop, unroll=N)
cannbotdsl.range(start, stop, unroll=N)
cannbotdsl.range(start, stop, step, unroll=N)
```

**请求完全展开**：设置`unroll_full=True`，向编译器提供完全展开循环的优化提示。该参数不保证消除循环。

```python
cannbotdsl.range(stop, unroll_full=True)
cannbotdsl.range(start, stop, unroll_full=True)
cannbotdsl.range(start, stop, step, unroll_full=True)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `*args` | 输入 | 一至三个位置参数，依次对应`stop`、`start, stop`或`start, stop, step`，单位为循环计数。`start`默认为0，`stop`为循环终点（不包含），无默认值，`step`默认为1。接受Python整数或运行时整数，步长必须为正。 |
| `unroll` | 输入 | 可选，默认`None`。通过关键字`unroll=N`指定每组迭代的展开因子，`N`为编译期已知的正Python `int`。使用`unroll_full=True`时，不得显式传入此参数。 |
| `unroll_full` | 输入 | 可选，默认`False`，类型为Python `bool`；设为`True`请求完全展开，不保证最终完全消除循环。使用`unroll=N`时保持为`False`。 |

## 返回值说明

仅用于`for ... in cannbotdsl.range(...)`指定迭代范围，不提供可独立使用的返回对象。

## 约束说明

- 只能直接用于`@host`、`@jit`或`@kernel`及其调用范围内的`for`循环。
- `start`、`stop`、`step`中的运行时整数类型须能统一；无法统一时，须先显式转换为相同的整数类型。
- `unroll`与`unroll_full=True`不能同时指定。

## 调用示例

以下示例分别使用普通循环、`unroll=4`和`unroll_full=True`计算整数的和。

```python
import torch
import torch_npu
import cannbotdsl as cbd


@cbd.kernel
def dsl_range_kernel(out: cbd.Tensor):
    total = 0
    for i in cbd.range(4):
        total = total + i
    out[0] = total

    total = 0
    for i in cbd.range(8, unroll=4):
        total = total + i
    out[1] = total

    total = 0
    for i in cbd.range(4, unroll_full=True):
        total = total + i
    out[2] = total


@cbd.host
def run(out):
    dsl_range_kernel[1](out)


def main():
    out = torch.zeros(3, dtype=torch.int64, device="npu")
    run(out)
    result = out.cpu().tolist()
    assert result == [6, 28, 6]
    print(result)


if __name__ == "__main__":
    main()
```

预期结果：

```text
[6, 28, 6]
```
