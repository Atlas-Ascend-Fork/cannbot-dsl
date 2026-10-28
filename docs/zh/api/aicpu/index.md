# AI CPU

本目录介绍用于运行期轻量控制计算的 AI CPU 接口，例如根据输入数据生成元数据、调度信息或任务划分结果。

当前 AI CPU 接口属于Phase-1能力。本文档只描述当前代码已经实现的接口；后续设计文档中提到的MLIR 方案不属于当前 API。

一个完整的使用流程如下：

1. 使用[@aicpu_kernel](./programming/aicpu-kernel.md)声明 AI CPU 函数。
2. 在[类型与参数](./programming/types/index.md)中为函数声明标量参数和GM输入、输出。
3. 使用[run_host](./validation/run-host.md)在Host 侧检查结果。
4. 使用[compile_aicpu_kernel](./build/compile.md)得到可启动的AI CPU 算子。
5. 使用[current_raw_stream](./runtime/current-raw-stream.md)和[CompiledAicpuKernel.launch](./runtime/launch.md)在Host 侧启动。

## 调用位置

| 内容 | 调用位置 | 说明 |
| --- | --- | --- |
| `@aicpu_kernel` | Host 侧 Python 代码 | 将函数声明为AI CPU 函数 |
| `@aicpu_kernel`修饰的函数体 | AI CPU | 执行函数中的整数计算和GM访问 |
| `run_host` | Host 侧 Python 代码 | 使用Host缓冲区执行参考计算 |
| `compile_aicpu_kernel` | Host 侧 Python 代码 | 准备可执行的AI CPU 算子产物 |
| `current_raw_stream` | Host 侧 Python 代码 | 获取当前NPU stream |
| `CompiledAicpuKernel.launch` | Host 侧 Python 代码 | 启动已准备好的AI CPU 算子 |

`@aicpu_kernel`是装饰器，函数体是被装饰的计算逻辑；两者不是两个需要分别调用的设备接口。`run_host`、`current_raw_stream`和`launch`属于Host 侧调用，不能写在AI CPU 函数体中。

## 导入方式

```python
from cannbotdsl.aicpu import (
    aicpu_kernel,
    gm_in,
    gm_out,
    I32,
    I64,
    U32,
    U64,
    current_raw_stream,
)
from cannbotdsl.aicpu.toolchain import compile_aicpu_kernel
```

## 相关页面

| 目的 | 页面 |
| --- | --- |
| 定义AI CPU 函数 | [@aicpu_kernel](./programming/aicpu-kernel.md) |
| 声明参数 | [类型与参数](./programming/types/index.md) |
| 选择标量类型 | [标量类型](./programming/types/scalars.md) |
| 声明GM输入 | [GM输入字段](./programming/types/gm-in.md) |
| 声明GM输出 | [GM输出字段](./programming/types/gm-out.md) |
| Host 侧验证 | [run_host](./validation/run-host.md) |
| 编译算子 | [compile_aicpu_kernel](./build/compile.md) |
| 启动算子 | [launch](./runtime/launch.md) |
