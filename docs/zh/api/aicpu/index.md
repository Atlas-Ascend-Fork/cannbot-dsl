# AI CPU

本目录介绍用于运行期轻量控制计算的 AI CPU 接口，例如根据输入数据生成元数据、调度信息或任务划分结果。

一个完整的使用流程如下：

1. 使用[@aicpu_kernel](./programming/aicpu-kernel.md)声明 AI CPU 函数。
2. 在[类型与参数](./programming/types/index.md)中为函数声明标量参数和GM输入、输出。
3. 使用[run_host](./validation/run-host.md)在Host 侧检查结果。
4. 使用[compile_aicpu_kernel](./build/compile.md)得到可启动的AI CPU 算子。
5. 使用[current_raw_stream](./runtime/current-raw-stream.md)和[CompiledAicpuKernel.launch](./runtime/launch.md)在Host 侧启动。

## 调用位置

AI CPU函数体中编写整数计算和GM访问逻辑。`run_host`、`compile_aicpu_kernel`、`current_raw_stream`和`CompiledAicpuKernel.launch`在Host侧调用，不能写在AI CPU函数体中。

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
