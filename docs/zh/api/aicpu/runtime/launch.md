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
| `stream_handle` | 输入 | 有效的 NPU stream 整数句柄，可通过 [current_raw_stream](./current-raw-stream.md) 获取。 |
| `kwargs` | 输入 | 按参数类字段名称传入参数。标量字段传入Python整数，`interface` 模式的 GM 字段可传入 NPU `torch.Tensor` 或整数设备地址；`legacy` 模式仅传入已打包参数结构的设备地址 `param_addr`。字段名称及值须与所选模式和参数声明对应。 |

### Tensor 参数与 profiling

以下增量能力已在 Ascend 950PR 验证，其他产品需按安装版本确认。

在 `interface` 模式下，传入 Tensor 对象可保留 shape、dtype 和 format，例如 `source=source`。只传 `source.data_ptr()` 时仅保留地址，不能推导 Tensor 描述。

- Tensor 必须在同一 NPU 上、连续且 dtype 与 GM 元素类型匹配：`I32`/`U32`/`I64`/`U64` 分别对应 `torch.int32`/`torch.uint32`/`torch.int64`/`torch.uint64`。
- 当前 NPU 必须与 Tensor 设备一致。`stream_handle` 必须等于该设备当前 TorchNPU stream 的句柄；非默认 stream 的获取和提交需在同一个 `torch.npu.stream(stream)` 上下文内完成。
- Tensor 最多 8 维，各维大小须能用 uint32 表示。非 NPU、非连续、超维或 stream/device 不匹配等情况在下发前报错；dtype 不匹配抛出 `TypeError`。
- 默认按 `gm_in` / `gm_out` 上报 Tensor 角色；若声明了 `ProfileSpec`，只上报其选择的字段。显式选择的字段传入裸地址会抛出 `TypeError`。
- 元数据上报不复制 Tensor 数据，也不为上报额外同步设备。调用方仍需保证缓冲区在任务执行期间有效。

需要重新编译 `interface` 动态库才能获得 Tensor profiling 上报能力；旧动态库可以沿用原启动入口，但不能因此获得缺失的上报信息。`legacy` 模式不解析参数块中的 Tensor。完整采集示例见 [AICPU Tensor 参数与 Profiling](../../../guide/aicpu-profiling.md)。

## 返回值说明

返回 `0` 表示 Host 侧启动调用已返回。如需排查任务下发异常，可参考 CANN 运行时日志。

AI CPU 算子异步执行。已下发的任务可以在同一 NPU stream 上按顺序衔接。Host 读取结果前应通过数据拷贝或同步接口等待相关任务完成。

## 约束说明

- 只能在Host 侧调用，不能在`@aicpu_kernel`函数体或其他设备函数体中调用。
- `stream_handle` 须由 CANN 或 TorchNPU 运行时获取。传裸句柄时，调用方须确保句柄有效，并与输入、输出地址属于同一设备。
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
        compiled.launch(
            current_raw_stream(0),
            source=source.data_ptr(),
            output=output.data_ptr(),
            count=source.numel(),
        )
        torch.testing.assert_close(output.cpu(), source.cpu())

    print("launch example passed")


if __name__ == "__main__":
    main()
```

预期结果：

```text
launch example passed
```
