# Native 与 ACLNN 的选择

根据算子的调用入口选择发布方式：只需要通过 CANNBot-DSL 的 Python 入口调用时，使用 AOT Native 发布；需要接入 ACLNN 工程时，使用 OpKit / ACLNN 发布。

| 需求 | 选择 |
| --- | --- |
| 复用已编译的 Host/Kernel 程序 | [`aot.compile`](../aot/compile.md) |
| 生成可随 Python 包发布的 Native 资源 | [`aot.export` / `aot.build`](../aot/publication.md) |
| 在运行时加载 Native 资源 | [`aot.register_directory`](../aot/runtime-registration.md) |
| 生成 ACLNN 设备二进制和 C++ 构建输入 | [`opkit.build`](./aclnn-build.md) |

Native 目录的注册不会替代 ACLNN 的安装和 CMake 构建；两者的产物格式和运行时入口不同。
