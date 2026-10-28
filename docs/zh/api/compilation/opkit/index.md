# OpKit / ACLNN 发布

OpKit 用于将 CANNBot-DSL 算子接入 ACLNN 工程。它提供 `@opkit.aclnn` 声明和 `opkit.build` 构建接口，生成 ACLNN 工程继续使用的设备二进制和 C++ 源码。

OpKit 生成的是 ACLNN 工程输入，不是已经安装到系统中的最终库文件。算子工程需要使用自己的 C++/CMake 流程继续完成编译和部署。

## 目录

- [ACLNN 构建](./aclnn-build.md)
- [Native 与 ACLNN 的选择](./native-or-aclnn.md)
