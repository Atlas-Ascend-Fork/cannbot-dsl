# Host API


Host API 在普通 Python 代码或 Host 编排函数中使用，负责定义 kernel、准备编译参数、编译和加载产物、启动设备任务以及查询平台信息。

## 接口分类

| 分类 | 主要内容 | 接口 |
| --- | --- | --- |
| 装饰器与程序定义 | 定义 Host 入口、DSL 辅助函数和设备 Kernel | [`host`、`jit`、`kernel` 及其他装饰器](/zh/api/decorators) |
| 编译与运行 | 编译、加载、调用和复用编译产物 | `compile`、`load`、`ProviderCallable`、`clear_compile_cache` |
| 参数与数据描述 | 描述编译期常量、动态维度和 Kernel 参数结构 | `Constexpr`、`Dim`、`TensorSpec`、`TensorListSpec`、`StructSpec` |
| [Host API：平台信息](/zh/api/host/platform-profiling/) | 查询设备及存储能力并配置 profiling | `PlatformInfo`、[`get_platform_info`](/zh/api/host/platform-profiling/get-platform-info)、[`get_mem_size`](/zh/api/host/platform-profiling/get-mem-size)、`ProfileSpec`、`profiler.report_tensor_info` |
| 异常与诊断 | 表达 Host 构图、编译和运行阶段错误 | `CANNBotError`、`DiagnosticCode` 等 |
