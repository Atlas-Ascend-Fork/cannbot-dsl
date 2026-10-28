# AOT 编译与 Native 发布

`cannbotdsl.aot` 提供两类能力：用规格描述提前编译 Host 入口，以及把编译结果发布为可注册的 Native 产物。算子开发时可以先使用 JIT 快速迭代；需要独立发布时再使用 AOT 产物。

## 目录

- [AOT 编译](./compile.md)
- [参数规格](./specs.md)
- [编译缓存](./cache.md)
- [Native 导出](./publication.md)
- [运行时注册](./runtime-registration.md)

Native 发布面向 CANNBot-DSL 的 Python 运行入口；ACLNN 发布面向 ACLNN 工程。两者是不同的发布方式，请根据算子的调用入口选择。
