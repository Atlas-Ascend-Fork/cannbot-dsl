# 附录：概念对照、命名速查与术语表

本附录是查询用的，不需要通读。两张对照表分别面向 **Ascend C 背景**和**读样例代码**两种场景。

## 一、Ascend C 概念对照

从 Ascend C 迁移过来的读者最需要这张表。左列是 Ascend C 的写法，右列是 DSL 里的对应物。

| Ascend C | CANNBot-DSL | 说明 |
| --- | --- | --- |
| `__aicore__` 核函数 | `@kernel` | — |
| `__simd_vf__` | `with vf(mode="simd"):` | VF 作用域 |
| `__simt_vf__` / `__launch_bounds__(N)` | `with vf(mode="simt", thread=N):` | SIMT VF；`thread` 同时决定每线程寄存器配额 |
| `__simd_callee__` | VF 内调用的 `@jit` 方法 | — |
| `TPipe` + `TQue` | `Channel(..., depth=N)` | **同步不用手写**，这是最大的差别 |
| `TBuf` | `Buffer(...)` | 单块 scratch |
| `InitBuffer` / `AllocTensor` / `FreeTensor` | `produce()` / `consume()` | 只选槽并推进游标，不阻塞 |
| `EnQue` / `DeQue` | 由 lowering 自动插入 | — |
| `SetFlag` / `WaitFlag` | 核内由 Channel 负责；跨流水用 pipe 级同步 | — |
| `DataCopy` | `mem_copy(dst, src, ...)` | 方向表见 API 文档 |
| `DataCopyPad` | `mem_copy(..., left_padding=, right_padding=, pad_value=)` | 随路填充 |
| `LoadData` / `LoadData2D` | `mem_copy` 到 L0A / L0B | — |
| `LoadDataWithTranspose` | `mem_copy(..., transpose=True)` | — |
| `Mmad` | `matmul(dst, lhs, rhs, init=...)` | `init` 控制是否累加 |
| `MmadWithSparse` | **不支持** | 3510 取消了 4:2 结构化稀疏 |
| `Fixpipe` 随路量化 | `mem_copy(..., deq_scale=...)` / `make_copy_engine(relu=True)` | — |
| `RegTensor<T>` | `vload` 返回的寄存器值 | 生命周期限于单个 VF |
| `MaskReg` | `Mask`（`full_mask` / `update_mask` 等） | — |
| `AddrReg` | 由编译器从地址表达式自动生成 | 不需要手写 |
| `Reg::Add` 等 | `vadd` 等 `v*` 接口 | 完整列表见 API 文档 |
| `Cast` | `vcast`（矢量）/ `cast`（标量） | — |
| `GetBlockIdx()` | `get_block_idx()` | — |
| `GetSubBlockIdx()` | `get_subblock_id()` | 只有 Mix 算子有 subblock |
| `dcci` / `dci` | `dcci_*` / `dci` | DCache 维护 |
| `__NPU_ARCH__ == 3510` | `get_platform_info().npu_arch == "dav-3510"` | 配 `target_version()` 用 |
| AIC↔AIV 经 GM 传标量 | `MemLoc.SSBUF` | 3510 新增，不用再绕 GM |

### 跨代变更速查（NPU ARCH 2201 → 3510）

从上一代迁移时，这些是会导致代码改不动或行为变化的点：

| 项 | 2201（A2 / A3） | 3510（950PR / 950DT） |
| --- | --- | --- |
| UB | 192 KB | **256 KB**（每 AIV；`get_mem_size("ub")` 返回 253952 = 248 KB，顶部 8 KB 预留已扣除） |
| L0C | 128 KB | **256 KB** |
| 矢量编程模式 | 仅 SIMD | **SIMD + SIMT 混合或纯 SIMT**（`vf(mode=...)`；混合编程以 SIMD 为主） |
| BiasTable | 1 KB | **4 KB** |
| Fixpipe Buffer | 2 KB | **4 KB** |
| L1 / L0A / L0B | 512 KB / 64 KB / 64 KB | 不变 |
| L0A 分形 | `FRACTAL_ZZ` | **`FRACTAL_NZ`**（省掉 L1→L0A 的一次格式转换） |
| 矢量操作数来源 | UB（MemBase） | **寄存器（RegBase）** |
| `L1 → GM` 通路 | 支持 | **取消**（需经单位矩阵 MMAD 落 L0C 再从 Fixpipe 出） |
| `GM → L0A/L0B` | 支持 | **取消**（必须经 L1 两跳） |
| `UB → L1` / `L1 → UB` / `L0C → UB` | 不支持 | **新增** |
| NDDMA（GM→UB 多维重排） | 无 | **新增** |
| 4:2 结构化稀疏 | 支持 | **取消** |
| int4 矩阵乘 | 支持 | **取消**（需先 `cast` 成 int8） |
| 核间同步模式 | 0 / 1 / 2 | **新增模式 4**（AIC 与单个 AIV） |
| AIC↔AIV 标量交接 | 经 GM | **SSBuffer** |
| 低精度格式 | — | **新增 HiF8 / MXFP8 / MXFP4 / FP8** |
| UB bank 组织 | 16 组 × 3 bank × 4 KB | **8 组 × 2 bank × 16 KB** |

## 二、样例命名速查

仓库样例的变量名有一套稳定习惯。读代码前过一遍这张表，能省很多来回。

### 存储层级前缀

| 前缀 | 含义 | 例子 |
| --- | --- | --- |
| `gm_` | GM 上的张量（通常是 Kernel 形参） | `gm_a`、`gm_x`、`gm_out` |
| `l1_` | L1 上的 Channel 或其槽位 | `l1_a`、`l1_b`、`l1_a_tensor` |
| `l0a` / `l0b` / `l0c` | 对应 L0 Buffer | `l0c_acc` |
| `ub_` | UB 上的资源 | `ub_out` |
| 无前缀 | 多为编译期常量或循环变量 | `base_m`、`k_l1_idx` |

### 后缀

| 后缀 | 含义 | 例子 |
| --- | --- | --- |
| `_ch` | 一个 `Channel` 对象（**不是** Tensor，要 `produce` / `consume`） | `x_ch`、`qk_ch` |
| `_buf` | 一个 `Buffer`（直接就是 Tensor） | `gamma_buf`、`softmax_max_buf` |
| `_t` / `_tensor` | 从 Channel 取出的槽位 Tensor | `l1_a_tensor` |
| `_tile` | 一个 tile 视图（`tile_slice` 的结果） | `gm_a_tile` |
| `_slice` | 第二级 tile 视图 | `l1_a_slice` |
| `_acc` | 累加器（整个循环共用一块） | `l0c_acc` |
| `_idx` | tile 序号（**不是元素下标**） | `m_idx`、`k_l1_idx`、`tile_idx` |
| `_num` / `_tiles` | 数量 | `used_core_num`、`n_tiles` |
| `_engine` / `_eng` | `make_copy_engine` 的结果 | `nd2nz_engine_a`、`fixpipe` |

### tiling 参数

| 名字 | 含义 |
| --- | --- |
| `base_m` / `base_n` / `base_k` | 最内层（L0 级）tile 尺寸 |
| `m_l1` / `n_l1` / `k_l1` | L1 级 tile 尺寸 |
| `k_l0_per_l1` | 一个 L1 块里有几个 L0 块 |
| `l1_buffer_num` | L1 Channel 的 depth（通常 2 或 4） |
| `l0c_db` | L0C 是否双缓冲（2 或 1） |
| `used_core_num` | 实际启动的核数 |
| `row_factor` / `col_factor` | 一次处理多少行 / 列（Vector 侧） |
| `VL` | 矢量 lane 数（fp32 下 64），见[矢量寄存器](/zh/guide/programming-model/vector-registers) |

### 常见缩写

| 缩写 | 全称 |
| --- | --- |
| `SFA` | Sparse Flash Attention |
| `MLA` | Multi-head Latent Attention |
| `MQSMLA` | Mixed Quant Sparse Flash MLA |
| `QLI` / `QSLI` | Quant (Sparse) Lightning Indexer |
| `KDA` | Kimi Delta Attention |
| `DP` / `SK` | Data Parallel / Split-K（Stream-K 的两类 tile） |
| `FD` | FlashDecode |
| `GQA` | Grouped Query Attention |
| `PA` | Page Attention（分页 KV Cache） |
| `TND` / `BSND` / `BNSD` | 张量轴顺序（Token-N-D 等） |

## 三、产品与架构命名对照

同一个硬件在不同文档里有五种叫法，这里一次对齐：

| 口径 | 名字 |
| --- | --- |
| 芯片（Prefill / 推荐场景） | **Ascend 950PR** |
| 芯片（Decode / 训练场景） | **Ascend 950DT** |
| 950PR 的加速卡形态 | **Atlas 350 加速卡** |
| NPU 架构版本 | **3510**（`__NPU_ARCH__ == 3510`） |
| 编译目标 | **`dav-3510`** |
| 上一代对应关系 | `dav-2201` ↔ Atlas A2 / A3（910B / 910C 级） |
| 超节点 | Atlas 950 SuperPoD |

查询方式：

```python
info = cbd.get_platform_info()
info.npu_arch          # "dav-3510"
info.soc_version       # 完整 SoC 版本
info.cube_core_num     # Cube 核数（分档出货，可能是 32 或 28 等）
info.vector_core_num   # Vector 核数
info.ai_cpu_num        # AI CPU 数
```

::: warning 核数与容量是分档的
同一型号按良率分档出货：950PR 为 32 或 28 个 Cube Core，950DT 为 36 / 32 / 28；内存容量也分档。**所以一定要用 `get_platform_info()` 和 `get_mem_size()` 查，不要硬编码。**
:::

## 四、术语表

| 术语 | 含义 |
| --- | --- |
| **构图期 / 编译期** | Python 解释器执行 DSL 函数体、生成设备代码的阶段 |
| **设备执行期 / 运行期** | 编译产物在 NPU 上执行的阶段 |
| **编译期值** | 构图期可知的 Python 值，会固化成设备代码里的常量 |
| **运行期值** | 只有设备执行时才有的值（标量形参、`get_block_idx()`、Tensor 元素） |
| **AIC / AIV** | AI Cube / AI Vector，AI Core 内的两类计算单元 |
| **block / subblock** | 一次启动的 group 编号 / Mix 算子中 AIV 在 group 内的编号 |
| **Mix 算子** | 一个 group 内同时使用 AIC 与 AIV 的算子 |
| **VF（Vector Function）** | `with vf(...)` 圈出的一段矢量计算。每个 VF 独立选择 SIMD 或 SIMT 实现 |
| **SIMD 模式** | `vf(mode="simd")`，寄存器级单指令多数据，lane 由掩码控制。950 上的主力范式 |
| **SIMT 模式** | 单指令多线程，每线程独立计算地址并可走不同控制流。混合编程用 `vf(mode="simt", thread=N)`，纯 SIMT 用 `vf(mode="simt")` 并从 Host 传入线程维度；用于不规则控制流与 gather/scatter，需 CANN 9.2.0 及以上版本 |
| **VF 融合** | 编译器把控制流等价的相邻 VF 合并，消除中间结果落 UB |
| **Hardware Loop** | 满足编码规范的循环被降成硬件循环；否则退化为软件循环 |
| **lane** | 矢量寄存器里的一个元素槽 |
| **VL** | 矢量寄存器位宽，Ascend 950 上为 256 字节 |
| **分形（Fractal）** | Cube 要求的分块存储格式，如 NZ、ZN |
| **NZ / ZN** | 「大 Y 小 x」记法：大写表示分形之间的排列，小写表示分形内部 |
| **nd2nz** | ND → NZ 的随路格式转换 |
| **K0** | Cube 分形的 K 方向粒度，= 32 字节 / 元素字节数 |
| **随路（on-the-fly）** | 由搬运单元在搬运过程中完成的转换，不占用计算单元 |
| **UnitFlag** | 让乘加与搬出细粒度并行的机制 |
| **Fixpipe** | L0C 搬出路径上的处理单元，可做量化、激活、格式转换 |
| **double buffer** | `depth=2` 的 Channel 加循环外预取，使搬运与计算重叠 |
| **在途（in-flight）** | 已发出但未完成的搬运 |
| **延迟线（DelayLine）** | 记录「若干拍之前那次搬运对应哪个 tile」的标量历史 |
| **workspace** | 算子自用的 GM 暂存区，由调用方分配 |
| **Stream-K** | 把尾部 tile 按 K 维切开以缓解长尾的调度策略 |
| **在线归约** | 边扫边修正统计量的归约形式，如 online softmax |
| **MX（MicroScaling）** | 每 32 个元素共享一个 E8M0 缩放因子的块量化格式 |
| **E8M0** | 8 位纯指数类型，用作 MX 的 scale |
| **HiF8** | 华为自研的变长指数 8 位浮点，不需要额外 scale |
| **HF32** | 用精度换 FP32 矩阵乘吞吐的模式（相当于 TF32 档） |
| **FTZ** | Flush-To-Zero，非规格化数直接当零。3510 默认开启 |
| **AOT** | 提前编译，把算子固化成可发布的二进制 |
| **Native 算子包** | AOT 产物打成的 wheel |
| **roofline** | 用带宽上限与算力上限判断瓶颈类型的模型 |
| **计算访存比** | FLOP / Byte，判断访存受限还是计算受限的首要指标 |

## 下一步

回到[编程模型总览](/zh/guide/programming-model/overview)，或直接查 [API 文档](/zh/api/)。
