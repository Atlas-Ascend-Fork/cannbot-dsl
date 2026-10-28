# compile_aicpu_kernel
## 产品支持情况

- Ascend 950PR&950DT系列产品：支持

## 功能说明

`compile_aicpu_kernel`将`@aicpu_kernel`声明的函数编译为可在NPU上执行的AI CPU 算子，并返回可用于启动该算子的对象。

## 函数原型

```python
compile_aicpu_kernel(
    kernel: Kernel,
    *,
    workdir: str,
    launch_mode: str,
    npu_arch: str = "dav-3510",
    build_x86: bool = False,
) -> CompiledAicpuKernel
```

从`cannbotdsl.aicpu.toolchain`导入。`Kernel`和`CompiledAicpuKernel`表示框架返回的对象，用户不需要直接构造。

## 参数说明

| 参数 | 输入/输出 | 说明 |
| --- | --- | --- |
| `kernel` | 输入 | `@aicpu_kernel`返回的函数句柄。 |
| `workdir` | 输入 | 保存编译结果的目录。目录不存在时自动创建，调用方需要具有写权限。 |
| `launch_mode` | 输入 | 参数传递方式。当前用户流程固定传入`"interface"`，启动时按参数类字段名称传入参数。 |
| `npu_arch` | 输入 | 目标NPU架构，默认值为`"dav-3510"`。应与实际运行设备和编译环境匹配。 |
| `build_x86` | 输入 | 是否额外准备Host 侧验证产物，默认值为`False`。普通设备运行不需要设置为`True`。 |

## 返回值说明

返回已编译的AI CPU 算子对象，可传入stream和参数调用[launch](../runtime/launch.md)启动。

## 约束说明

无

## 调用示例

以下示例编译一个整数数组复制函数，在当前NPU stream上启动并检查结果。

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

    print("compile example passed")


if __name__ == "__main__":
    main()
```

预期结果：

```text
compile example passed
```
