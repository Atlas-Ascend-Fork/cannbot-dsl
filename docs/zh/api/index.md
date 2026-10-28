# CANNBot-DSL API 参考

CANNBot-DSL API 参考按用户要完成的工作组织：先了解语言入口，再选择数据结构、数据搬运和计算接口，最后查看设备执行、调试、编译发布以及 AI CPU 的专用接口。

| 主题 | 适合查找的内容 |
| --- | --- |
| [语言](./language/index.md) | 装饰器、控制流、VF、编译期表达式和 Kernel 启动 |
| [数据结构](./data-structures/index.md) | 数据类型、Tensor、Layout、Tiler、片上存储、Buffer、RingBuffer 和 Array |
| [数据搬运](./data-movement/index.md) | `mem_copy` 及其搬运配置 |
| [计算](./compute/index.md) | 标量、矢量、矩阵、原子和张量计算 |
| [设备执行](./execution/index.md) | 设备执行上下文、流水类型和已公开的同步接口 |
| [调试](./debug/index.md) | Kernel 调试接口 |
| [编译与发布](./compilation/index.md) | Host 参数规格、JIT/AOT 编译、Native 发布和 ACLNN 工程接入 |
| [AI CPU](./aicpu/index.md) | AI CPU Kernel 的声明、参数、编译、验证和启动 |

各接口页面中的产品支持、函数原型、参数说明、约束说明和调用示例以页面自身为准；本入口只负责导航，不重复接口定义。

页面标题和导航使用 API 名称本身，不预设统一的 Python 模块前缀；实际导入方式以对应页面的调用示例为准。
