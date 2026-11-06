# 调试

本目录说明Tensor、矢量寄存器及格式化打印接口，覆盖Python、Host和Kernel的适用范围。

## 接口

| 接口 | 用途 |
| --- | --- |
| [dump_tensor](./dump_tensor.md) | 将GM或支持的片上存储区间写入CANN调试流。 |
| [dump_reg](./dump_reg.md) | 打印SIMD VF中的一个矢量寄存器。 |
| [print](./print.md) | 按上下文分派Python、Host、Kernel格式化打印，并兼容原有采集模式。 |
| [print_time_stamp](./print_time_stamp.md) | 记录设备周期和PC，并识别执行打点的核。 |

## 使用说明

各接口支持的数据类型、存储位置、长度及执行域见对应页面。格式化print、两个dump接口和采集式print的长度限制、输出获取方式、同步与AOT能力不同。当前不支持 FP4 可读数值输出，可先数值转换再调测。
