# for

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

`for`按给定范围或固定序列重复执行循环体。运行时计数循环可使用Python内建`builtins.range()`或CANNBotDSL提供的`cannbotdsl.range()`；编译期展开循环可使用`range_constexpr()`或固定tuple/list。需要设置运行时循环的展开方式时，可使用`cannbotdsl.range()`的`unroll`或`unroll_full`参数。

## 语法形式

`for`语句无返回值，`else`可选：

```python
for i in iterable:
    ...
else:
    ...
```

循环正常结束时执行可选的`else`语句块；编译期循环通过`break`退出时，不执行`else`。

## 迭代范围

`for`的`iterable`支持下列范围调用或固定tuple/list。

### `builtins.range()`

在`for`中使用时建立设备执行的计数循环，不提供循环展开参数。

#### 函数原型

参数支持Python整数或运行时整数，调用形式如下：

```python
builtins.range(stop: Any, /) -> range
builtins.range(start: Any, stop: Any, step: Any = 1, /) -> range
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `start` | 输入 | 循环起点，支持Python整数或运行时整数，单位为循环计数；省略时为0。 |
| `stop` | 输入 | 循环终点，不包含`stop`，必选，无默认值。支持Python整数或运行时整数，单位为循环计数。 |
| `step` | 输入 | 步长，支持Python整数或运行时整数，必须为正值，单位为循环计数；默认值为1。 |

#### 返回值说明

普通Python调用返回`range`对象；在DSL的`for`中使用时按设备计数循环处理。

#### 约束说明

无。

### `cannbotdsl.range()`

用于设备执行的计数循环，可通过`unroll`或`unroll_full`设置循环展开方式。

#### 函数原型

```python
cannbotdsl.range(
    *args: Any,
    unroll: Optional[int] = None,
    unroll_full: bool = False,
)
```

**普通循环**：不设置展开参数。

```python
cannbotdsl.range(stop)
cannbotdsl.range(start, stop)
cannbotdsl.range(start, stop, step)
```

**按展开因子展开**：设置`unroll=N`，按每组`N`次迭代请求展开；实际生成代码中可能仍保留循环。

```python
cannbotdsl.range(stop, unroll=N)
cannbotdsl.range(start, stop, unroll=N)
cannbotdsl.range(start, stop, step, unroll=N)
```

**完全展开**：设置`unroll_full=True`，请求完全展开运行时循环；实际生成代码中仍可能保留循环。

```python
cannbotdsl.range(stop, unroll_full=True)
cannbotdsl.range(start, stop, unroll_full=True)
cannbotdsl.range(start, stop, step, unroll_full=True)
```

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `*args` | 输入 | 一至三个位置参数，依次对应`stop`、`start, stop`或`start, stop, step`，单位为循环计数。`start`默认为0，`stop`为循环终点（不包含），无默认值，`step`默认为1。接受Python整数或运行时整数，步长必须为正。 |
| `unroll` | 输入 | 可选，默认`None`。通过关键字`unroll=N`指定每组迭代的展开因子，`N`为编译期已知的正Python `int`。使用`unroll_full=True`时，不得显式传入此参数。 |
| `unroll_full` | 输入 | 可选，默认`False`，类型为Python `bool`；设为`True`请求完全展开，不保证最终完全消除循环。使用`unroll=N`时保持为`False`。 |

#### 返回值说明

仅用于`for ... in cannbotdsl.range(...)`指定迭代范围，不提供可独立使用的返回对象。

#### 约束说明

- 只能直接用于`@host`、`@jit`或`@kernel`及其调用范围内的`for`循环。
- `start`、`stop`、`step`中的运行时整数不能混用有符号和无符号类型；位宽不同时，统一为其中最宽的类型。
- `unroll`与`unroll_full=True`不能同时指定。

### `range_constexpr()`

范围规则与Python `range`相同；包含它的循环在编译期展开，不保留运行时循环。展开次数会增加生成代码的体积，适合较小且固定的迭代范围。

#### 函数原型

```python
range_constexpr(*args: int) -> range
```

支持以下调用形式：`range_constexpr(stop)`、`range_constexpr(start, stop)`和`range_constexpr(start, stop, step)`。

#### 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `*args` | 输入 | 一至三个编译期已知的Python整数位置参数，依次对应`stop`、`start, stop`或`start, stop, step`，单位为循环计数。`start`默认为0，`stop`为循环终点（不包含），无默认值，`step`默认为1；范围规则与Python `range`相同。 |

#### 返回值说明

返回Python `range`对象。

#### 约束说明

- 循环中的`break`、`continue`遵守编译期执行的Python循环语义。

### 固定tuple/list

按固定元素展开循环。元素为tuple/list时支持循环目标解包。

语法示例：

```python
for left, right in ((1, 2), (3, 4)):
    ...
```

#### 函数原型

无。

#### 参数说明

无。

#### 返回值说明

无。

#### 约束说明

循环中的`break`、`continue`遵守编译期执行的Python循环语义。

## 约束说明

- 运行时循环不支持`break`、`continue`，循环体内不支持`return`、`raise`；嵌套的编译期循环可使用`break`、`continue`控制自身循环。
- 运行时循环体不支持推导式、嵌套函数或类、`lambda`、`try`、`yield`、异步语法、`del`和`match`。

## 调用示例

以下示例对比各类循环求和及`for ... else`的结果，并在Host侧逐项校验。

```python
import builtins
import cannbotdsl as cbd
from cannbotdsl.tensor import Tensor
import torch
import torch_npu  # noqa: F401


@cbd.kernel
def for_kernel(out: Tensor):
    # 对比不同迭代形式的求和结果，分别写入输出。
    total = 0
    for i in builtins.range(4):
        total = total + i
    out[0] = total

    total = 0
    for i in cbd.range(4):
        total = total + i
    out[1] = total

    total = 0
    for i in cbd.range(8, unroll=4):
        total = total + i
    out[2] = total

    total = 0
    for i in cbd.range(4, unroll_full=True):
        total = total + i
    out[3] = total

    total = 0
    for i in cbd.range_constexpr(4):
        total = total + i
    out[4] = total

    pair_total = 0
    for left, right in ((1, 2), (3, 4)):
        pair_total = pair_total + left + right
    out[5] = pair_total

    total = 0
    for i in builtins.range(4):
        total = total + i
    else:
        total = total + 100
    out[6] = total


@cbd.host
def run(out):
    for_kernel[1](out)


def main():
    out = torch.zeros(7, dtype=torch.int64, device="npu:0")
    run(out)
    torch.npu.synchronize()
    assert out.cpu().tolist() == [6, 6, 28, 6, 6, 10, 106]
    print(out.cpu().tolist())


if __name__ == "__main__":
    main()
```

示例输入：

```text
builtins.range(4)：0、1、2、3
cannbotdsl.range(4)：0、1、2、3
cannbotdsl.range(8, unroll=4)：0至7
cannbotdsl.range(4, unroll_full=True)：0、1、2、3
range_constexpr(4)：0、1、2、3
固定tuple：元素为(1, 2)和(3, 4)
for ... else：循环范围为0、1、2、3
```

预期结果：

```text
[6, 6, 28, 6, 6, 10, 106]
```
