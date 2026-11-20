# 调试与性能分析

本目录介绍调试输出、运行时异常 Tensor 落盘和性能分析配置。

## 接口与诊断功能

| 接口 | 用途 |
| --- | --- |
| [debug](./debug.md) | 为 kernel 和设备端 jit 配置硬件流水消融。 |
| [dump_tensor](./dump_tensor.md) | 将GM或支持的片上存储区间写入CANN调试流。 |
| [dump_reg](./dump_reg.md) | 打印SIMD VF中的一个矢量寄存器。 |
| [异常 Tensor 落盘](./exception_dump.md) | 在AIC/AIV异常时导出CANN标准Tensor文件，供故障解析。 |
| [log](./log.md) | 分级诊断日志、ACLNN错误上报和输出配置。 |
| [统一错误模板](./error_message.md) | 38种统一格式错误日志。 |
| [print](./print.md) | 按上下文分派Python、Host、Kernel格式化打印，并兼容原有采集模式。 |
| [print_time_stamp](./print_time_stamp.md) | 记录设备周期和PC，并识别执行打点的核。 |
| [Profiling 名称与类型](./profiling.md) | 配置性能分析中的Kernel名称、类型和参数信息。 |

## 使用说明

各接口支持的数据类型、存储位置、长度及执行域见对应页面。格式化print、两个dump接口和采集式print的长度限制、输出获取方式、同步与AOT能力不同。当前不支持 FP4 可读数值输出，可先数值转换再调测。

异常 Tensor 落盘由运行时异常监测触发，使用独立的CANN异常Dump配置；其Tensor范围和文件状态见对应页面，不沿用设备打印接口的类型及缓冲限制。
