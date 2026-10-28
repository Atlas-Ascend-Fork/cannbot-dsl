# Kernel.run_host
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

`run_host`在Host 侧执行`@aicpu_kernel`函数，用于在启动设备算子前检查计算结果。结果写入传入的Host输出缓冲区。

## 函数原型

```python
Kernel.run_host(**kwargs: object) -> int | None
```

`Kernel`表示`@aicpu_kernel`返回的函数句柄。

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `kwargs` | 输入/输出 | 按参数类字段名称传入参数。标量字段传入Python整数；GM输入传入可读取的Host缓冲区；GM输出传入可写的Host缓冲区。 |

## 返回值说明

返回函数的状态值。计算结果写入传入的GM输出缓冲区，不通过返回值返回数组。

## 约束说明

- 参数名称必须与参数类字段名称一致，并提供函数执行时会访问的字段。
- Host输入和输出缓冲区必须覆盖函数访问的元素范围，输出缓冲区必须允许原地写入。

## 调用示例

以下示例先使用`run_host`生成参考结果，再启动设备算子并比较两条执行路径的输出。

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
    golden = np.zeros_like(host_source)
    assert copy_values.run_host(
        source=host_source,
        output=golden,
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
        torch.testing.assert_close(output.cpu(), torch.from_numpy(golden))

    print("run_host example passed")


if __name__ == "__main__":
    main()
```

预期结果：

```text
run_host example passed
```
