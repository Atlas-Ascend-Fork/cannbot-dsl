# CompiledAicpuKernel.launch
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

`CompiledAicpuKernel.launch`在Host 侧启动已经编译的AI CPU 算子，并将参数传递给对应的AI CPU 函数。

## 函数原型

```python
CompiledAicpuKernel.launch(
    stream_handle: int,
    **kwargs: object,
) -> int
```

`CompiledAicpuKernel`由[compile_aicpu_kernel](../build/compile.md)返回，不需要直接构造。

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `stream_handle` | 输入 | 当前NPU stream的整数句柄，通常由[current_raw_stream](./current-raw-stream.md)获取。 |
| `kwargs` | 输入 | 按参数类字段名称传入参数。标量字段传入Python整数，GM字段传入设备缓冲区地址。字段名称、数量和类型必须与参数类声明一致。 |

## 返回值说明

返回`0`，表示启动请求已经提交。设备侧后续算子可以在同一NPU stream上直接使用输出缓冲区；如果Host需要立即读取结果，应通过数据拷贝或同步接口等待相关任务完成。

## 约束说明

- 只能在Host 侧调用，不能在`@aicpu_kernel`函数体或其他设备函数体中调用。
- `stream_handle`、输入地址和输出地址必须来自同一设备。
- 输入和输出缓冲区必须保持有效，直到对应stream完成。

## 调用示例

以下示例获取当前NPU stream，启动整数数组复制函数，并将设备输出复制到Host 侧检查。

```python
import tempfile

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
    torch.npu.set_device(0)
    source = torch.tensor([2, 5, 9], dtype=torch.int64, device="npu:0")
    output = torch.empty_like(source)

    with tempfile.TemporaryDirectory() as workdir:
        compiled = compile_aicpu_kernel(
            copy_values,
            workdir=workdir,
            launch_mode="interface",
        )
        status = compiled.launch(
            current_raw_stream(0),
            source=source.data_ptr(),
            output=output.data_ptr(),
            count=source.numel(),
        )
        assert status == 0
        torch.testing.assert_close(output.cpu(), source.cpu())

    print("launch example passed")


if __name__ == "__main__":
    main()
```

预期结果：

```text
launch example passed
```
