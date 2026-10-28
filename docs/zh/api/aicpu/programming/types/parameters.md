# 参数声明

AI CPU 函数使用一个普通 Python 类声明参数。字段名称用于Host验证和算子启动，字段声明顺序用于确定参数顺序。

## 参数类

参数类只声明字段，不需要添加装饰器：

```python
from cannbotdsl.aicpu import gm_in, gm_out, I64

class CopyArgs:
    source: gm_in(I64)
    output: gm_out(I64)
    count: I64
```

一个字段只能使用以下三种声明方式：

| 声明方式 | 含义 |
| --- | --- |
| `I32`、`I64`、`U32`、`U64` | 标量参数，或GM字段的元素类型 |
| `gm_in(T)` | 元素类型为`T`的只读GM输入 |
| `gm_out(T)` | 元素类型为`T`的可读写GM输出 |

`T`只能取上述四个标量类型。具体取值和适用场景见[标量类型](./scalars.md)、[GM输入字段](./gm-in.md)和[GM输出字段](./gm-out.md)。

## 字段名称

字段名称必须是有效的Python标识符，并在调用时保持一致：

```python
copy_values.run_host(
    source=host_source,
    output=host_output,
    count=host_source.size,
)

compiled.launch(
    stream,
    source=source.data_ptr(),
    output=output.data_ptr(),
    count=source.numel(),
)
```

Host验证使用Host缓冲区，设备启动使用设备地址；两种调用方式使用相同的字段名称。

## 字段顺序

字段按照参数类中的声明顺序传递。修改字段顺序会改变启动时的参数顺序，因此定义好参数类后应保持字段顺序稳定。

## 缓冲区准备

调用方负责准备GM输入和输出缓冲区：

- GM输入缓冲区必须包含函数读取的元素。
- GM输出缓冲区必须预先分配，并允许函数写入。
- 需要长度、步长或其他元数据时，应使用独立的标量字段传入。
- 设备启动后，输入和输出缓冲区需要保持有效，直到对应stream完成。

## 调用示例

以下示例声明一个包含标量、只读输入和可写输出的参数类，并使用同一个参数类完成Host验证和设备启动。

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

    print("parameter declaration example passed")


if __name__ == "__main__":
    main()
```

预期结果：

```text
parameter declaration example passed
```
