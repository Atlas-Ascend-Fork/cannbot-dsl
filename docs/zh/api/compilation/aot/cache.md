# AOT 编译缓存

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

`compile_cache` 将一个调用 `cannbotdsl.compile` 的普通 Python 函数包装为可复用的缓存调用。使用相同参数和编译选项重复调用时，会复用已有的编译结果；参数或相关编译选项发生变化时，会生成新的编译结果。

## 函数原型

```python
cannbotdsl.aot.compile_cache(func)
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `func` | 输入 | 负责调用 `cannbotdsl.compile` 并返回编译结果的普通 Python 函数。 |

## 返回值说明

返回带编译缓存功能的调用对象。

## 约束说明

- `func` 必须使用普通 `def` 定义，不能使用 `async def`、`*args` 或 `**kwargs`。
- `func` 每次调用都必须返回 `cannbotdsl.compile` 返回的已编译调用对象。

## 调用示例

```python
import cannbotdsl
from cannbotdsl import aot


@aot.compile_cache
def get_program(spec):
    return cannbotdsl.compile(run, spec, spec, spec)
```
