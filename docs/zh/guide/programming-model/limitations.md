# 当前限制、迁移与常见问题

这一节把前面各章散落的「现在不能这么写」收在一起，并对照官方约定标出容易误读的地方。CANNBot-DSL 0.7.0 是尝鲜版本，API 不保证兼容；这里记录的是**当前文档和样例能核对上的事实**，不是稳定承诺。

## 950 新特性的覆盖边界

Ascend 950 的对外宣传里有几项能力，当前**没有**由 DSL 0.7.0 暴露，或者方向与直觉相反。先把边界划清楚，省掉找接口的时间。

| 特性 | DSL 0.7.0 的状态 |
| --- | --- |
| SIMD 矢量编程（RegBase） | ✓ 完整支持，`with vf(mode="simd")` |
| **SIMD + SIMT 混合编程** | ✓ **支持**。`with vf(mode="simt", thread=N)` 加 `cannbotdsl.ops.simt` 的线程 / warp / 原子接口，见[三类计算单元](/zh/guide/programming-model/compute#simt-模式) |
| **纯 SIMT 编程** | ✓ **支持**。设备侧使用 `vf(mode="simt")`，Host 侧使用 `kernel[dim3(block), dim3(thread), dyn_ub_buf](...)`，见[Host 启动规则](/zh/guide/programming-model/compute#host-启动规则) |
| HiF8 / FP8 / MXFP8 / MXFP4 矩阵乘 | ✓ 支持，见[数据类型与量化](/zh/guide/programming-model/data-types) |
| HF32 | ✓ 支持（`enable_hf32` / `set_hf32_round_mode` / `set_fp32_mode`；**没有** `enable_hf32_trans`） |
| **4:2 结构化稀疏** | **硬件已取消**。3510 不支持 4:2 稀疏矩阵乘，需要用 Vector 自己做稠密↔稀疏转换 |
| **int4 矩阵乘** | **硬件已取消**。需先 `cast` 成 int8 |
| 块稀疏 / Token 稀疏注意力 | ✓ 支持，但靠的是**通用能力**：离散访存、128 字节访问粒度、Cube↔Vector 融合通路、MX 低精度。**没有专用的稀疏注意力数据通路** |
| 跨卡 / 分布式通信 | 包里有 `cannbotdsl.distributed`（顶层导出）与 `hcomm_init` / `write_nbi` / `read_nbi` / `hcomm_drain` 等同步接口，但官方文档站没有详情页，本章不涉及。当前把 DSL 当作**单算子**编程模型使用 |
| 卷积 | ✓ 有一组专用接口（`Conv2dSpec`、`conv2d_load_fmap` / `load_filter` / `load_im2col` / `store_output`、`make_conv2d_spec`），顶层导出，本章未展开 |
| HiF4 | 属于下一代（Ascend 960），950 不支持 |

::: warning 「样例名字里有 sparse」不等于有硬件稀疏加速
仓库里 `sparse_flash_attention`、`quant_block_sparse_attn`、`qwen_sparse_attn`、QSLI 都带 sparse 字样，但它们是**算法层面的稀疏**（只访问候选块 / 候选 token），不是权重结构化稀疏。后者在这一代已经取消。
:::

跨代的完整变更对照见[附录](/zh/guide/programming-model/appendix)。

SIMD + SIMT 混合编程和纯 SIMT 编程都需要 CANN 9.2.0 及以上版本。

## 入口与装饰器

| 写法 | 现状 |
| --- | --- |
| 普通 Python 直接调 `@kernel` / `@jit` | 不支持。外部入口应是 `@host`，再由 `@host` 启动 Kernel |
| `@jit` 里启动 Kernel（包括 `kernel[block_dim](...)` 和纯 SIMT 的 `kernel[dim3(block), dim3(thread)](...)`） | 不支持。启动语句必须直接位于 `@host` 函数体 |
| 用 `@jit` 包一层 `run()` 当编译入口 | 官方装饰器页不允许；部分 API 示例仍这样写，不要当模板抄 |
| `@host` 里用 `const_expr()` 做设备分支 | 官方控制流页只承认 `@jit` / `@kernel`。`@host` 上条件已是 Python 布尔值时，直接写 `if` |
| `@kernel` 之间互相启动 | 不支持 |
| `@host` 返回 Tensor | 应返回 `None`；输出由调用方准备好传入 |

## 构图与设备执行

同一份 DSL 代码会经历**构图**和**设备执行**两个阶段，不是「Python 跑两遍」。构图阶段 Python 解释器会执行函数体以收集控制流和接口调用；设备阶段跑的是编译产物。

因此：

- `print()` 只在构图期有效；设备侧打印用 `cbd.print()`，且目前只保证 UB / L1 / L0C。
- `__init__`、普通 Python 算术、编译期 `if` 的结果会固化进设备代码。
- 列表推导、`lambda`、`try`、嵌套函数、`yield`、`match` 等可以出现在 `__init__` 或编译期展开的循环里，**不能写在设备循环体里**。

## 控制流

设备上保留下来的 `if` / `for` / `while` 有一组硬约束：

- 不能 `break` / `continue` / `return` / `raise`。
- 循环变量不能直接带出循环。
- 分支里创建的 `Buffer` / `Channel` / Tensor 不能带到分支外。
- `cannbotdsl.range()` 是设备循环标记，不是 Python 的 `range` 对象。Host 侧遍历用 `builtins.range()`。
- `unroll` / `unroll_full` 不会把 `cbd.range()` 变成编译期循环。
- Hardware Loop 是编译器约束（从 0 起步长 1、无运行时跳转、边界不变），不是 Python 类型。不要在 DSL 里写 `uint16_t`。

## 存储与搬运

| 限制 | 说明 |
| --- | --- |
| `produce()` / `consume()` 不等待 | 只选槽并推进游标。同步由 lowering 按配对插入。配对写错时不会在这两个调用上报「等不到」 |
| VF 内调用 `produce()` / `consume()` | 不支持。进 `with vf(...)` 之前取好 Tensor |
| VF 内的 UB 重叠 / 跨流水 | `Channel` 不管；按接口约束自己插 `vmem_bar` |
| CrossCore `depth` | `1 <= depth <= 8` |
| `dsl.UB.view()` 之后 | 当前 Kernel 的 UB `Buffer` 必须走 `make_buffer()`，不能再混用普通 UB `Channel` |
| `n1_pad` | 仅 NZ。额外 N1 stride，不自动补零，也不是 bank 冲突一键开关 |
| `get_mem_size` | 可查 `"ub"` / `"l1"` / `"l0a"` / `"l0b"` / `"l0c"` / `"bt"` / `"fb0"`。**L2 和 SSBUF 查不到**，传进去抛 `ValueError`。`"ub"` 返回 253952（248 KB），预留已扣除 |
| `UB.view()` | 一旦调用，当前 Kernel 的 UB `Buffer` 必须走 `make_buffer()`，普通 UB `Channel` 也不能再创建。两种风格不能混用 |
| 尾块 `mem_copy` | `tile_slice` 与 Channel 槽位 shape 必须一致。不要把尾块缩成 `(actual,)` 再拷进 `(tile,)` 的槽；GM 先对齐，多余 lane 用掩码 |
| Scalar 直接读写 GM 的 Cache 一致性 | 框架不会自动插入 `dcci` / `dci`；调用方必须显式处理 Cache 可见性，或使用 `*_bypass` |

## 矩阵乘

- 接口数学语义是 \(C = A[M,K] \times B[K,N]\)。
- 样例里 GM 右矩阵常为 `(N, K)`，搬入 L0B 时转置，golden 用 `A @ B.T`。读代码时先分清「官方公式」和「这份 GM 怎么存」。
- `matmul` 只在 AIC 上生效。从 L0C 搬到 ND 的 GM/UB 时，样例通常不写 `engine`；需要显式 NZ2ND、量化或 ReLU 时再配 `make_copy_engine`。

## JIT 参数

- `Dim` 的 `min` / `max` / `multiple_of` 是运行时长度必须满足的约束，不是「声明了 `multiple_of=8` 编译器就替你去掉尾块」的承诺。
- 同一次 `compile(...)` 里，同名 `Dim` 的约束必须完全一致。
- 维度表达式（如 `M * 2`）不是新的 `Dim`；同一次编译里必须至少有一个位置直接使用 `M`。

## Mix / 多核

- **只有 Mix 算子**才有 block（AIC）和 subblock（AIV）。纯 Cube、纯 Vector 没有 subblock。
- Mix 的 1：N 目前文档写的是 1：1 或 1：2，不要假设任意比例。
- 1：2 时逻辑 AIV 编号常用 `get_block_idx() * 2 + get_subblock_id()`。

## 报错了先查什么

| 现象 | 先看 |
| --- | --- |
| 普通 Python 调 Kernel / `@jit` 报错 | 入口是不是 `@host` |
| `const_expr` / `target_version` 报错 | 是不是写在 `@jit` / `@kernel` 里；传入的是不是编译期值 |
| `cbd.range` 不能迭代 | 是不是当成 Python `range` 用在了 Host 普通循环里 |
| 同步相关的数据错乱 | `produce` / `consume` 是否成对；有没有在 VF 里选槽；VF 是否缺 `vmem_bar` |
| 最后几个元素错 | 尾块掩码、GM 是否对齐到 tile、`tile_slice` 有没有被缩 shape |
| 换卡容量不够 / 莫名变慢 | 有没有硬编码 UB/L1；L2 是否误用了 `get_mem_size` |
| 矩阵乘结果像转置错了 | GM 上 B 是 `(K, N)` 还是 `(N, K)`，golden 该不该 `A @ B.T` |

更细的排查步骤见[调试与精度验证](/zh/guide/programming-model/debugging)。完整接口约束以 [API 文档](/zh/api/) 和 `samples/` 为准。

## 已知的破坏性变更

0.7.0 是尝鲜版本，接口不保证兼容。目前能从 [CHANGELOG](https://gitcode.com/cann/cannbot-dsl/blob/master/CHANGELOG.md) 核对到的破坏性变更：

| 变更 | 影响 |
| --- | --- |
| `from_torch_npu` 改为 `from torch import as_tensor` | 适配 cannbotdsl 0.0.3 时的调整，涉及 flash_attn、flash_kda、matmul、pointnet_sa、rms_norm、voxel_conv 多个样例 |
| `flash_kda()` 新增**必选**参数 `metadata` | 算子改为「调度与计算解耦」，不再自行生成调度信息、也不再启动 AI CPU。旧调用方式不可用 |
| `samples/matmul/` 拆成 `matmul/` 与 `quant_matmul/` | 导入路径与测试路径变化 |
| `flash_kda_metadata` 与 `fused_recurrent_kda_snapshot` 撤并 | 前者实现并入 `samples/flash_kda/`，后者删除 |

升级版本时的建议：先跑一遍自己算子的精度测试，再看 CHANGELOG 里涉及到的接口。

## 文档本身的缺口

**缺口清单统一维护在[编译选项与产物观察 · 真正还缺的能力](/zh/guide/programming-model/compiler-options#真正还缺的能力)**，这里不另立一张会与之脱节的表。

划分标准是：**包源码能证实的就写清楚并标明「包内可用 · 文档站未收录」，真查不到的才叫缺口。** 文档站没收录不等于能力缺失——IR 转储、日志级别、缓存落盘、`dump_reg`、`RegLayout`、`PIPE` 都属于前者。

下面这几项按「文档站未覆盖」而非「能力缺失」处理：

| 项 | 现状 |
| --- | --- |
| AI CPU 接口详情 | 官方 `/zh/api/aicpu/` 只有名字索引。本章内容据样例与包源码整理，见 [AI CPU 与调度计划](/zh/guide/programming-model/aicpu) |
| 分布式通信 | `cannbotdsl.distributed` 存在，无详情页 |
| L1 / SSBUF / FBUF 容量 | API 文档不给数值；L1 用 `get_mem_size("l1")` 可查，**SSBUF 查不到**（传 `"ssbuf"` 抛 `ValueError`） |

这些可以在 [GitCode 仓库](https://gitcode.com/cann/cannbot-dsl)的 Issues 里反馈。

## 下一步

[概念对照、命名速查与术语表](/zh/guide/programming-model/appendix)：查 Ascend C 概念映射、样例命名习惯和术语。
