# 编程模型

本章介绍 CANNBot-DSL 的编程模型：它如何把一段 Python 代码变成跑在昇腾 AI Core 上的算子，为什么要这样设计，以及怎样利用这套模型写出高性能算子。

[API 文档](../api/index.md)回答「某个接口怎么用」，本章回答「为什么是这些接口、它们怎么配合」。本章按语言机制、数据与硬件、动手实践和工程化组织内容；接口细节一律以 API 文档和 `samples/` 为准。

## 写在前面

CANNBot-DSL 是一个**显式**的 DSL：它不替你决定数据放在哪一级存储、什么时候搬、怎么切块、哪段代码跑在 Cube 上哪段跑在 Vector 上。这些决定都由你在 Python 里写出来。

这听起来比「自动调优」的框架麻烦，但它换来了三件事：

- **性能可预期**。代码写成什么样，设备上大致就执行成什么样，没有需要猜测的黑盒重写。
- **调优有抓手**。性能不达标时，你能定位到具体是哪一级搬运、哪个 tile 大小、哪条流水线的问题。
- **对 Agent 友好**。显式、局部、可组合的写法更容易被程序生成和校验，这也是 CANNBot 仓群把 DSL 单独拆出来的原因。

代价是：你需要先理解一点硬件。本章第二节就是为此准备的。

「显式」不等于「没有编译器优化」。VF 融合、Hardware Loop、`Channel` 的依赖同步仍由后端插入。显式的是存储层级、搬运方向、tile 和程序顺序，不是每一条硬件指令。

## 阅读顺序

本章共 22 节加一个附录，按「语言机制 → 数据与硬件 → 动手 → 工程化」四段组织。

### 第一段：语言机制（1～5）

| 顺序 | 文档 | 读完之后你会知道 |
| --- | --- | --- |
| 1 | [简介](./introduction.md) | CANNBot-DSL 是什么、六个装饰器各自管什么、一个最小算子长什么样 |
| 2 | [硬件与执行模型](./hardware-model.md) | AIC/AIV、存储层级与容量、block 与 subblock、数据通路、关键规格数字 |
| 3 | [代码生成](./code-generation.md) | 构图和设备执行两个阶段分别跑什么、编译期值和运行期值的区别 |
| 4 | [类型系统与宿主语言边界](./type-system.md) | 值分几类、`self` 上能放什么、Python 的哪些能力能跨过构图期 |
| 5 | [控制流](./control-flow.md) | 哪些 `if`/`for` 在编译期消失、哪些保留到设备上 |

### 第二段：数据与硬件（6～11）

| 顺序 | 文档 | 读完之后你会知道 |
| --- | --- | --- |
| 6 | [数据、布局与切块](./data-and-layout.md) | Shape/Stride/Layout/Tensor、逻辑布局与物理布局、`tile_slice` |
| 7 | [片上存储、流水与 GM 协作](./onchip-memory.md) | `Buffer` 与 `Channel`、double buffer、UB 预算、bank 冲突、workspace |
| 8 | [矢量寄存器与 lane 模型](./vector-registers.md) | VL = 256 字节、lane 怎么排、寄存器预算、掩码与 lane 的关系 |
| 9 | [三类计算单元](./compute.md) | Cube 矩阵乘流程、VF 寄存器级矢量计算、SIMD + SIMT 混合与纯 SIMT 的启动方式、Scalar 与原子操作 |
| 10 | [数据类型与量化](./data-types.md) | HiF8、MXFP8/MXFP4、scale 存在哪里、谁负责搬 |
| 11 | [同步、Cache 与跨核交接](./synchronization.md) | 框架替你同步什么、剩下四类同步怎么写 |

### 第三段：动手（12～15）

| 顺序 | 文档 | 读完之后你会知道 |
| --- | --- | --- |
| 12 | [写出第一个算子](./first-operator.md) | 从向量加一路加到多核 + double buffer 的完整过程 |
| 13 | [第一个 Cube 算子与第一个 Mix 算子](./cube-and-mix.md) | 矩阵乘的四步流程、K 向累加、AIC 与 AIV 协作 |
| 14 | [AI CPU 与调度计划](./aicpu.md) | 变长与稀疏场景怎么把分核计划外置 |
| 15 | [融合算子的设计方法论](./fusion-design.md) | 怎么把一个数学式拆成能跑的阶段、在线归约与状态递推 |

### 第四段：工程化（16～22 + 附录）

| 顺序 | 文档 | 读完之后你会知道 |
| --- | --- | --- |
| 16 | [JIT 参数与编译缓存](./jit-arguments.md) | 什么会触发重新编译、怎么用 `Dim` 控制编译爆炸、缓存怎么落盘 |
| 17 | [torch 接口与 stream 语义](./torch-interop.md) | 怎么接到框架上、为什么必须 synchronize |
| 18 | [编译选项与产物观察](./compiler-options.md) | 哪些开关影响生成代码、怎么把 MLIR 和 AscendC 源码 dump 出来 |
| 19 | [高性能算子编写指南](./performance.md) | 五个层次的优化手段、roofline 怎么算 |
| 20 | [调试与精度验证](./debugging.md) | 设备侧打印、寄存器转储、精度比对、性能采集 |
| 21 | [AOT 与 Native 算子包](./aot-packaging.md) | 怎么把算子编译成 wheel 发布 |
| 22 | [当前限制、迁移与常见问题](./limitations.md) | 哪些写法现在不支持、报错了怎么查 |
| 附 | [概念对照、命名速查与术语表](./appendix.md) | Ascend C 概念映射、读样例用的命名表、术语 |

::: tip 四段的分界线是「依赖关系」，不是「难度」
动手段（12～15）只依赖前两段：写一个算子需要知道硬件长什么样、数据怎么切、寄存器怎么用，不需要先懂编译缓存和 AOT。

工程化段（16～22）讲的是**算子写完之后**的事——怎么控制重编译、怎么看产物、怎么接到框架上、怎么发布。它们和「怎么写出第一个算子」没有依赖关系，所以放在动手之后。
:::

### 几条推荐路径

| 你的情况 | 建议路径 |
| --- | --- |
| 赶时间，先跑通一个算子 | 1 → 2 → 12 → 19 |
| 要写矩阵乘或 attention | 1 → 2 → 8 → 9 → 13 → 15 |
| 要写量化算子 | 1 → 2 → 10 → 13 → 19 |
| 有 Ascend C 背景 | 附录第一节 → 7 → 11 → 13 |
| 结果不对，来查问题 | 20 → 22 → 11 |
| 要上线部署 | 16 → 18 → 21 |

中间遇到不懂的概念再回查 3～11。

## 前置知识

- Python 3.10+，尤其是装饰器、`with` 语句和类。
- PyTorch 基本用法（`torch.Tensor`、`dtype`、`contiguous`）。
- 不要求有 Ascend C 经验，但有的话第 2 节会读得更快，也可以先看[附录](./appendix.md)的概念对照表。

## 环境

| 维度 | 要求 |
| --- | --- |
| 昇腾产品 | Ascend 950PR / Ascend 950DT（NPU ARCH 3510），950PR 的加速卡形态为 Atlas 350 |
| CPU 架构 | x86_64 / aarch64 |
| Python | 3.10、3.11、3.12 |
| CANN | 建议 9.2.0-beta.2(SIMT 必须 9.2.0 及以上版本) |
| CANNBot-DSL | 0.7.0 |
| torch | 2.7.1 / 2.9.0 / 2.10.0 / 2.11.0 / 2.12.0 |

### 安装

CANN 这一侧需要**两个**软件包：toolkit 之外，950 还要单独装 ops 包。只装 toolkit 会在运行时报缺算子。

```bash
CANN_BASE=https://ascend.devcloud.huaweicloud.com/artifactory/cann-run/software/9.2.0-beta.2/x86_64

wget ${CANN_BASE}/Ascend-cann-toolkit_9.2.0-beta.2_linux-x86_64.run
wget ${CANN_BASE}/Ascend-cann-950-ops_9.2.0-beta.2_linux-x86_64.run     # 950 必装

bash ./Ascend-cann-toolkit_9.2.0-beta.2_linux-x86_64.run --install --force --install-path=${install_path}
bash ./Ascend-cann-950-ops_9.2.0-beta.2_linux-x86_64.run --install --force --install-path=${install_path}

source ${install_path}/ascend-toolkit/set_env.sh
```

然后装 DSL：

```bash
python -m pip install cannbot-dsl
python -c 'import cannbotdsl; print(cannbotdsl.__version__)'
```

发行包名是 `cannbot-dsl`，导入名是 `cannbotdsl`。编译器后端已包含在 whl 包内，安装后即可使用，无需从源码构建。`aarch64` 机器把上面 URL 里的 `x86_64` 换掉即可。完整步骤与 HDK 版本要求见[仓库 README](https://gitcode.com/cann/cannbot-dsl)。

## 本章的约定

- 代码里统一使用 `import cannbotdsl as cbd`。从子模块逐个导入（例如 `from cannbotdsl.ops.reg import vadd`）也等价。
- 「构图期 / 编译期」指 Python 解释器执行你的 DSL 函数、生成设备代码的阶段；「设备执行期 / 运行期」指编译产物在 NPU 上跑的阶段。
- 出现 ✓ / ✗ 的表格表示「支持 / 不支持」。
- 指向 API 文档的链接一律带 `.html` 后缀。站点启用了 `cleanUrls`，但托管层未做 rewrite，无后缀的叶子页冷启动会 404。

::: warning `cbd.print` 覆盖了内建 `print`
`cannbotdsl` 把设备侧打印命名为 `print` 并在包顶层导出。`import cannbotdsl as cbd` 时没问题（内建 `print` 还在），但 `from cannbotdsl import *` 会把内建 `print` 换掉。本章一律写 `cbd.print(...)` 表示设备打印、`print(...)` 表示构图期打印。
:::

::: info 两种标注的区别
官方 API 文档站没有完全覆盖 0.7.0 的接口面，所以本章有些内容的依据是 `samples/` 和 `cannbotdsl` 包源码。按可信度分两档标注：

| 标注 | 含义 |
| --- | --- |
| **包内可用 · 文档站未收录** | 包源码里能找到确切定义（标注块会给出模块路径），可以直接用；但文档站未收录，API 稳定性没有承诺 |
| **待核实** | 只在样例里见过用法，接口语义没有权威出处。**写法可以参考，不要当作接口承诺。** |

两种都不要写进对兼容性敏感的生产代码。
:::

## 下一步

从[简介](./introduction.md)开始。
