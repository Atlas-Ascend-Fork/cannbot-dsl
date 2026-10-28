# @aicpu_kernel

## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

`@aicpu_kernel`用于声明一个AI CPU 函数。函数读取参数类中的标量和GM字段，完成轻量控制计算，并通过GM输出字段写回结果。

## 函数原型

```python
aicpu_kernel(fn: Callable) -> Kernel
```

`aicpu_kernel`通常以装饰器形式使用。`Kernel`表示装饰器返回的函数句柄，用户不需要直接构造该类型。

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `fn` | 输入 | AI CPU 函数。函数只接收一个参数，该参数使用参数类作为类型注解；参数类的声明方式见[参数声明](./types/parameters.md)。 |

## 返回值说明

返回装饰后的AI CPU 函数句柄。该句柄可以传给[compile_aicpu_kernel](../build/compile.md)，也可以调用[run_host](../validation/run-host.md)进行Host 侧验证。

## 约束说明

- 入口函数只能接收一个参数。
- 入口参数必须使用参数类作为类型注解，字段类型使用[标量类型](./types/scalars.md)、[GM输入字段](./types/gm-in.md)或[GM输出字段](./types/gm-out.md)中的声明方式。
- 函数通过GM输出字段返回计算结果；返回值用于表示函数执行状态，通常返回`0`。
- 函数体中不能调用`run_host`、`compile_aicpu_kernel`、`current_raw_stream`或`launch`。

## 调用示例

以下示例使用`@aicpu_kernel`声明整数数组复制函数，在Host 侧验证后编译并启动该函数。

```python
import tempfile

import numpy as np
import torch
import torch_npu  # noqa: F401

from cannbotdsl.aicpu import aicpu_kernel, current_raw_stream, gm_in, gm_out, I64
from cannbotdsl.aicpu.toolchain import compile_aicpu_kernel


class CopyArgs:
    source: gm_in(I64)
    output: gm_out(I64)
    count: I64


@aicpu_kernel
def copy_values(args: CopyArgs):
    for i in range(args.count):
        args.output[i] = args.source[i]
    return 0


def main():
    host_source = np.array([2, 5, 9], dtype=np.int64)
    host_output = np.zeros_like(host_source)
    assert copy_values.run_host(
        source=host_source,
        output=host_output,
        count=host_source.size,
    ) == 0
    np.testing.assert_array_equal(host_output, host_source)

    torch.npu.set_device(0)
    source = torch.tensor([2, 5, 9], dtype=torch.int64, device="npu:0")
    output = torch.empty_like(source)

    with tempfile.TemporaryDirectory() as workdir:
        compiled = compile_aicpu_kernel(
            copy_values,
            workdir=workdir,
            launch_mode="interface",
        )
        compiled.launch(
            current_raw_stream(0),
            source=source.data_ptr(),
            output=output.data_ptr(),
            count=source.numel(),
        )
        torch.testing.assert_close(output.cpu(), source.cpu())

    print("aicpu decorator example passed")


if __name__ == "__main__":
    main()
```

预期结果：

```text
aicpu decorator example passed
```
