# aot.compile

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

`cannbotdsl.aot.compile` 将 `@host` 函数或绑定的 `@host` 方法提前编译为可复用的调用对象。传给 `compile` 的参数用于描述编译时规格；返回对象调用时再传入真实 Tensor 或其他运行时参数。

## 函数原型

```python
cannbotdsl.aot.compile(func, /, *args, **kwargs)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `func` | 输入 | `@host` 函数或绑定的 `@host` 方法。 |
| `args` | 输入 | 用于描述编译时参数规格的位置参数，例如 [TensorSpec](../host/specs/tensor_spec.md) 或 [Dim](../host/specs/dim.md)。 |
| `kwargs` | 输入 | 用于描述编译时参数规格的关键字参数。 |

## 返回值说明

返回可复用的已编译调用对象。

## 约束说明

- `compile` 应在普通 Python 调用层使用，不能在 `@host` 函数体内调用。
- `func` 必须是 `@host` 函数或绑定的 `@host` 方法。

## 调用示例

下面的示例使用 Tensor 规格提前编译 Host 入口：

```python
import cannbotdsl as cb


@cb.kernel
def add_kernel(x, y, out):
    ...


@cb.host
def run(x, y, out):
    add_kernel[1](x, y, out)


program = cb.aot.compile(run, cb.TensorSpec((1024,), cb.dtypes.float16),
                         cb.TensorSpec((1024,), cb.dtypes.float16),
                         cb.TensorSpec((1024,), cb.dtypes.float16))
```
