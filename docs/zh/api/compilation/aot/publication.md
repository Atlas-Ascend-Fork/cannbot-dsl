# Native 导出与构建

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

`aot.export` 声明 Native 发布入口，`aot.build` 找到并执行这些入口，将其中编译得到的结果整理为 Native 产物目录。产物目录可随 Python 包发布，运行时通过 `aot.register_directory` 注册。

## 函数原型

```python
cannbotdsl.aot.export(name)
cannbotdsl.aot.build(*, collectors, target, output, operators=None, pattern="*.py")
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `name` | 输入 | Native 发布任务的名称。 |
| `collectors` | 输入 | 包含 `@aot.export` 发布入口的模块或目录。 |
| `target` | 输入 | 目标设备型号。 |
| `output` | 输入 | Native 产物输出目录。 |
| `operators` | 输入 | 可选的收集任务名称列表。 |
| `pattern` | 输入 | 扫描 Python 文件的匹配模式，默认为 `*.py`。 |

## 返回值说明

`export` 返回装饰后的发布入口；`build` 返回生成的 Native 产物目录。

## 约束说明

- 发布入口必须是无参数的普通 Python 函数，并且不能使用 `async def`。
- 发布入口需要调用编译接口，使 `build` 能够把编译结果加入 Native 产物目录。
- 输出目录必须是新的发布目录，不能覆盖已有产物。

## 调用示例

```python
import cannbotdsl
from cannbotdsl import aot


@aot.export("my_op")
def publish_my_op():
    program = cannbotdsl.compile(run, input_spec, output_spec)
    program.close()


native_dir = aot.build(collectors="my_project", target="dav-3510",
                       output="build/native")
```
