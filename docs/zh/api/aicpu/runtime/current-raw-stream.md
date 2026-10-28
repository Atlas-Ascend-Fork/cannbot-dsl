# current_raw_stream
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

`current_raw_stream`获取当前NPU stream的整数句柄，供[CompiledAicpuKernel.launch](./launch.md)提交AI CPU 算子。

## 函数原型

```python
current_raw_stream(device_id: int = 0) -> int
```

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `device_id` | 输入 | NPU设备编号，默认值为`0`。 |

## 返回值说明

返回当前设备stream的整数句柄。

## 约束说明

- 只能在Host 侧调用。
- `device_id`必须对应当前运行环境中可用的NPU设备。

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
        compiled.launch(
            current_raw_stream(0),
            source=source.data_ptr(),
            output=output.data_ptr(),
            count=source.numel(),
        )
        torch.testing.assert_close(output.cpu(), source.cpu())

    print("current stream example passed")


if __name__ == "__main__":
    main()
```

预期结果：

```text
current stream example passed
```
