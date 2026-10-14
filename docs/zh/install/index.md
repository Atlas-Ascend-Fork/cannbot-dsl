# 开始使用

CANNBot-DSL 是面向 Ascend NPU 算子开发的开源项目。当前仓库收录多个复杂算子样例，并提供与样例对应的测试代码。

## 快速上手

### 1. 准备开发环境

先查看[环境要求与安装步骤](/zh/guide/programming-model/overview#环境)，完成 CANN toolkit、950 ops 包和 CANNBot-DSL 的安装。完整的软硬件配套说明及样例特定版本要求见[仓库 README](https://gitcode.com/cann/cannbot-dsl#软硬件配套说明)。

### 2. 运行已有样例

从[样例导航](/zh/project/examples/)选择目标算子，按照对应 README 准备输入、依赖并运行。随后参考[运行测试](/zh/project/examples/testing)验证结果。

### 3. 编写自己的算子

先阅读[硬件与执行模型](/zh/guide/programming-model/hardware-model)，再跟随[写出第一个算子](/zh/guide/programming-model/first-operator)理解向量加、多核切块和流水组织。教程中的验证状态以页面说明为准；需要可复现的实现时，请结合仓库样例及其测试。

## 文档导航

- [项目介绍](/zh/guide/project/overview)：了解项目定位和内容边界。
- [编程模型](/zh/guide/programming-model/overview)：按语言机制、数据与硬件、动手实践和工程化的顺序理解 DSL。
- [仓库结构](/zh/guide/project/repository)：快速找到样例、测试和工程脚本。
- [样例导航](/zh/project/examples/)：按场景进入已有算子目录。
- [API 文档](/zh/api/)：查看公共接口的签名、参数和使用示例；可从[装饰器](/zh/api/decorators)开始了解程序入口。
- [参与贡献](/zh/project/community/contributing)：了解提交代码前的基本流程。

## 推荐阅读顺序

首次访问时，建议先阅读项目介绍，再按[编程模型的推荐路径](/zh/guide/programming-model/overview#几条推荐路径)了解 DSL，或根据目标进入样例目录。准备修改代码时，请同时阅读对应样例的 README 和测试文件。

::: info 当前阶段
本站正在分阶段建设。当前版本提供项目概览、编程模型、样例导航和首批 API 文档。
:::
