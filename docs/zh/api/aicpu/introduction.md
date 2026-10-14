# AI CPU API


AI CPU API 使用 `cannbotdsl.aicpu` 独立命名空间，面向 AI CPU kernel 的定义、参数描述、Host 启动和诊断。

有关 `@aicpu_kernel` 的基本用法，请参见[装饰器](/zh/api/decorators)。

## 接口分类

| 分类 | 主要内容 | 接口 |
| --- | --- | --- |
| Kernel 定义 | 定义 AI CPU kernel 入口 | `aicpu_kernel` |
| 参数与 ABI 类型 | 描述 GM 输入、输出和固定宽度整数参数 | `GmIn`、`GmOut`、`I32`、`I64`、`U32`、`U64` |
| 运行与诊断 | 准备 Host 缓冲、stream 和结构体参数并报告 trace 错误 | `X86Buffer`、`current_raw_stream`、`pack_struct_bytes`、`AicpuTraceError` |

## 导入方式

```python
from cannbotdsl.aicpu import aicpu_kernel
```

AI CPU API 的公开范围以 `cannbotdsl.aicpu.__all__` 为准。
