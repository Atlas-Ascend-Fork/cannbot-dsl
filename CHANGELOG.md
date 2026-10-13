<!-- changelog-cursor: ba3d3b5e58ebcc072f14d32e00dc6d403eb1c8cb -->

## 🔥 更新日志

### 【2026-10-13】
#### 特性增强 Feature Enhancement
- 【indexer_prologue_qw / indexer_prologue_k】接入 `net/native_package` Native 打包：改用 `cannbotdsl.aot.export` 声明导出入口并登记到 `operator_groups.toml` 的 `ds41` 组，可通过 `./run-build.sh --operator indexer_prologue_qw indexer_prologue_k` 编译为 Native 算子包。`indexer_prologue_k` 新增默认导出 profile（H=5120/512 × mode0 带/不带 scale、mode1 G=8，共 6 个二进制），支持通过 `CANNBOTDSL_DS41_PROFILES` 的 `indexer_prologue_k` 键覆盖。同时移除 `indexer_prologue_qw` 的 `_CompatChannel` 及 `matmul`/`mem_copy` 包装层，直接使用 0.7 的 `Channel.produce()/consume()`，编译产物二进制逐字节不变。

### 【2026-10-08】
#### 新特性 New Features
- 【block_attn_res_update_rms_norm】迁入 Block Attention Residual Update 与 RMSNorm 融合算子：将本层增量 `delta` 加到当前残差块，结合 `pseudo_query` 分数与 online-softmax 累积结果合并，再沿隐藏维做 RMSNorm 并乘缩放权重 `gamma`；同步注册 profiling identity（`op_type=BlockAttnResUpdateRMSNorm`）。
- 【fused_recurrent_kda_snapshot】迁入 KDA decode 算子 `fused_recurrent_kda()`：计算 1～8 token 的 KDA decode，并把每个 token 的递归状态写入 state-pool 槽位，用于为候选分支保存完整状态的 Snapshot 协议。
- 【mega_recurrent_kda】迁入多 batch decode 融合算子：将 QKV、decay、beta、output-gate 投影、causal Conv1D + SiLU、Q/K 归一化、recurrent KDA、RMSNorm、output gating 和输出投影融合为一次 mixed AIC/AIV kernel 调用。
- 【mega_recurrent_kda_replayssm】迁入 ReplaySSM verify 算子：将 QKV、decay、beta、gating 和输出投影融合为一次 mixed AIC/AIV kernel 调用。
- 【commit_recurrent_kda_replayssm】迁入 ReplaySSM 提交算子：根据 Mega ReplaySSM 保存的 replay record，将每个 batch 已接受的 token 依次提交到递归状态。
- 【quant_mla_prolog】迁入 AW1 MLA Prolog 算子，支持 BF16 非量化输出（C0）与 MXFP8 量化输出。

#### 特性增强 Feature Enhancement
- 【flash_kda】同步最新实现：适配 CANNBotDSL 0.7 前向路径，重构 packed32 `_cube_raw_l1_to_l0b` 为 reinterpret+mem_copy，并新增 `cube_raw_l1_to_l0a/b` 辅助路径。

### 【2026-09-30】
#### 新特性 New Features
- 【attn_epilogue】新增 Attention 输出投影融合算子 `attn_epilogue()`：融合 inverse RoPE、两次 OCP MXFP8 激活量化与 WoA/WoB 两级矩阵乘，支持 BF16 输入/输出、E4M3FN 权重与 E8M0 scale，按平台核数和输入规模自适应分块，内部管理工作区。测试覆盖接口契约、分块边界、中间量化结果、输出独立性及图重放一致性；样例文档补充精度验证与清 L2 条件下的小算子拼接性能对比记录。

### 【2026-09-29】
#### 新特性 New Features
- 【attn_prologue】迁入基于 CANNBotDSL 的 MXFP8 attention prologue 融合算子，包含 QA/KV 投影、RMSNorm、QR 动态量化、QB 投影、尾部 RoPE 和 KV cache 原地更新；同步迁入独立 CPU golden、FP64 精度复核与完整用例 runner，新增 4 个 decode/prefill NPU 冒烟用例及 6 个 CPU 精度判据用例并接入 CI 算子清单。

### 【2026-09-23】
#### 新特性 New Features
- 【batch_matmul】新增非量化批量矩阵乘算子 `batch_matmul()`：$C[c\_batch, M, N] = A[a\_batch, M, K] @ B[b\_batch, N, K]^T + bias$，batch 维按 numpy/torch.matmul 语义右对齐广播（size-1 维重复、多维同时广播、混合秩），支持 rank 2~6（双 2-D 输入自动升维并压回 2-D）；fp16/bf16/fp32（核内 fp32 累加），fp32 可选 HF32 快速模式（TF32 档）。转置由 stride 自动推导：canonical `transpose(-1,-2)` 视图（K-major 存储）零拷贝翻转到对应 kernel 路径，其余非连续布局拒绝；bias 经 BT 折入 init MMAD，支持 `[N]` 共享与 `[*c_batch, N]` 逐 batch。面向 NPU ARCH 3510（Ascend 950PR / Ascend 950DT），host tiling 对齐主线 `batch_mat_mul_v3`（ResetBaseDav3510 + CalL1TilingDefault + GetBaseK），性能对比 `torch.bmm/baddbmm`（msprof Task Duration）。测试覆盖 fp16/bf16/fp32/HF32、bias（shared/per-batch）、K-major 转置视图、多维 batch 广播（rank 2~6）、非对齐尾块、2-D 输入与不可广播拒绝共 18 个 NPU 用例。

- 【matmul_streamk】新增 Stream-K（DPSK）矩阵乘算子 `matmul_streamk()`：$C[M,N] = A[M,K] \times B[K,N]$，采用 DP + SK 混合调度——DP tiles 各核独立计算完整 K 维并直写输出 GM，SK tiles 按 K 维 split 写入 workspace，AIV 从 workspace 做 fp32 累加 reduce；纯 DP 路径启用 slide window + 行反转提升 L2 命中率。A/B/C 支持 float16、bfloat16（核内仅支持 transpose_a=False、transpose_b=False），host 侧 tiling 推导 tile/缓冲深度/L2 cache 参数。面向 NPU ARCH 3510（Ascend 950PR / 950DT），5 组代表性 shape × 2 dtype 共 10 个精度用例，性能对比 CANN 包内置 torch_npu matmul（100 个纯 SK 用例）。

- 【quant_batch_matmul_mxa8w4】新增混合精度量化矩阵乘算子 `matmul_mix_quant()`（MXA8W4）：$C[M,N] = (A \cdot sa)[M,K] \times (B \cdot sb)[N,K]^T + bias$，激活为 MXFP8（float8_e4m3fn + E8M0 paired-scale），权重为 MXFP4（uint8 打包 float4_e2m1 + E8M0 paired-scale），输出 float16；AIV 权重 prologue 将 fp4 → fp8 做纯位置换转换（2^-6 因子由 fixpipe `deq_scale=64` 与 host 侧 `bias/64` 补偿），经 CrossCore 通道交接到 AIC MX matmul。K 为 32 的倍数且无上限（片上 K 状态全部窗化），M/N/K 尾块由引擎在搬运途中补零。面向 NPU ARCH 3510（Ascend 950PR / 950DT），11 个精度用例（5 组代表性形状 × 含 bias + 输入校验），性能对比 CANN 内置 `aclnnQuantMatmulV5`。

### 【2026-09-21】
#### 文档 Documentation
- 【API 文档】扩充设备侧接口文档并重组 Kernel API 分类：Host API 新增平台信息（`get_platform_info`、`get_mem_size`）；Kernel API 划分为数据搬运、系统变量访问、同步与缓存控制三类，新增系统变量访问 9 个与同步与缓存控制 4 个接口页，原 `api/operations/` 下的接口页迁至 `api/kernel/`，侧边栏同步更新。

### 【2026-09-15】
#### 新特性 New Features
- 【flash_kda_metadata】新增 FlashKDA 配套的独立 AICPU 调度算子 `flash_kda_metadata()`：按 64 个 token 划分 chunk，生成各轮 batch、任务前缀和、核间任务区间与有效序列长度等调度信息，供 `flash_kda` 消费；不读取 Q/V/state 数值，也不计算 attention 输出，返回一维 int32 张量。支持 BNSD / BSND / TND 布局（TND 必传 `cu_seqlens`），约束 D = 128、Nv % Nqk == 0、Nv <= WORKSPACE_SLOTS（当前 861）；调度配置不变时 metadata 可在层间复用。共 8 个 NPU 用例，覆盖三种布局、变长尾块与 858/861/864 容量边界。

- 【fused_recurrent_kda_snapshot】新增 KDA decode 算子 `fused_recurrent_kda()`：计算 1～8 token 的 KDA decode，并把每个 token 的递归状态写入 state-pool 指定槽位，用于为候选分支保存完整状态的 Snapshot 协议。支持 BSND / BNSD / packed TND，D = 128、N <= 96、单序列长度不超过 8；`state` 为 `[pool_slots, N, 128, 128]`（BF16 或 FP32），返回 `(state, out)`，**state 原地更新**。测试覆盖三种布局与 BF16/FP32 state，并与 PyTorch golden 对照。

#### 特性增强 Feature Enhancement
- 【flash_kda】接口升级为「调度与计算解耦」：新增必选参数 `metadata` 与布局 `TND`，`flash_kda()` 不再生成调度信息、也不启动 AICPU，改为消费配套 `flash_kda_metadata` 的输出，可在调度配置相同的层间复用；序列长度不再要求 64 对齐，尾块由内核处理。**注意：`metadata` 为必选参数，属破坏性接口变更，旧调用方式不再可用。**

#### 测试框架 Test Framework
- 【测试】新增 `test/_samples_path.py` 共享加载器，为每个样例分配私有包前缀按路径加载，避免与环境中已安装的同名外部模块冲突；flash_kda 测试重写为 12 个 NPU 用例（布局精度、非对齐尾块、跨轮调度与 K 快照补零回归，容差 atol=rtol=5e-3），并内联 PyTorch golden 做输出与最终 state 的三方对照。

### 【2026-09-14】
#### 文档 Documentation
- 【文档站点】文档站从框架推进到可对外发布：新增 5 个内容分区（开始使用 / 样例 / API 文档 / 参与贡献 / 关于），覆盖项目定位与支持范围（NPU ARCH 3510 / Ascend 950PR / 950DT）、样例导航、测试与贡献流程与文档分批发布方式；首页改为 Hero + 特性卡片，落地亮/暗两套品牌主题与中文全文搜索。

- 【API 文档】确立公共接口文档的分类与编写模板：接口按调用位置与执行模型分为 Host API、Kernel API、AI CPU API 三类，新增通用接口文档模板 `docs/api/api-template.md`（要求以公开接口和验证结果为准，未经验证的产品不得标记为「支持」），并按模板落地首批接口页——数据搬运 `api/operations/data-movement/mem-copy` 与系统变量访问 `api/operations/system/get-core-id`。

### 【2026-09-11】
#### 文档 Documentation
- 【文档站点】搭建基于 VitePress 的文档站点框架（`docs/`），配置站点标题、描述、base 路径、`cleanUrls`、`lastUpdated`、页脚与 sitemap，并声明 `package.json` / `package-lock.json` 依赖。

### 【2026-09-10】
#### 新特性 New Features
- 【grouped_matmul】新增非量化分组矩阵乘算子 `group_matmul()`：$y_i[m_i,n_i] = x_i[m_i,k_i] \times weight_i[k_i,n_i]$，覆盖 S1–S6 六种场景，支持 group_type=-1（不分组）/ 0（M 轴分组 SPLIT_M）/ 2（K 轴分组 SPLIT_K），group_list_type 支持 0（cumsum 累积和）与 1（count 计数），group_list 为 None 时按各组张量首维隐式分组，并支持零大小组；x/weight/y 支持 float16、bfloat16，返回值恒为 `List[Tensor]`。面向 NPU ARCH 3510（Ascend 950PR / Ascend 950DT），性能对比 9.1.0 CANN 包内置 GroupedMatmul。

- 【quant_batch_matmul_mxfp8】新增 MXFP8 全量化矩阵乘算子 `npu_quant_matmul()`（QBMM MXFP8）：$C[M,N] = Dequant(A)[M,K] \times Dequant(B)[N,K]^T$，MXFP8 沿 K 轴每 32 个数据使用一个 E8M0 Scale，公开 Scale 接口将相邻两个 Scale 组成一个 pair（分组轴长 `ceil(K/64)`、末维 2）；a/b 支持 float8_e4m3fn 与 float8_e5m2，输出支持 float16、bfloat16、float32；四种 transposeA/transposeB 组合由 A/B 与 Scale 的 view shape 推导。面向 NPU ARCH 3510，性能对比 CANN 包内置 `npu_quant_matmul`。

#### 架构重构 Architecture Refactoring
- 【matmul】将 `samples/matmul/` 拆分为 `matmul/`（非量化 A16W16）与 `quant_matmul/`（MXFP8 全量化）两个子目录，测试同步迁移至 `test/matmul/matmul/` 与 `test/matmul/quant_matmul/`。

### 【2026-09-09】
#### 新特性 New Features
- 【kv_compress_epilog】新增 KV Cache 压缩更新算子 `kv_compress_epilog()`：将 bfloat16 激活值量化压缩后按 slotMapping 散写到 cache 对应行并**原地更新**，输出行布局为 `[rope bf16 128B][nope fp8 (d-64)B][scale][pad]`；nope 段每 64 个元素独立计算 scale 并量化为 FP8(e4m3)，rope 段保留 bfloat16；scale 段支持 bf16（quant_mode=0）与 e8m0（quant_mode=1）；约束 64 < d ≤ 8192 且 d % 64 == 0。面向 NPU ARCH 3510，测试覆盖 quant_mode × round_scale × 2 种 shape 共 8 个用例并逐字节对比 golden。

### 【2026-09-04】
#### 缺陷修复 Bug Fixes
- 【兼容性】适配 cannbotdsl 0.0.3：flash_attn、flash_kda、matmul、pointnet_sa、rms_norm、voxel_conv 各样例的 `from_torch_npu` 改由 `from torch import as_tensor` 提供，并同步更新相关测试用例。

### 【2026-08-07】
#### 特性增强 Feature Enhancement
- 【rms_norm】UB 缓冲深度由固定双缓冲改为按 UB 容量自适应计算（在 `DOUBLE_BUFFER_NUM` 与单缓冲之间择优），并同步更新算例图与测试用例。

### 【2026-08-05】
#### 新特性 New Features
- 【flash_attn】新增 Flash Attention 算子 `flash_attn()`：$O = softmax(QK^T \cdot scale)V$，D=128，Q/K/V/O 支持 float16、bfloat16；支持 mask_mode 0（全注意力）与 3（right-context causal），支持 GQA（N1 须为 N2 整数倍），layout_q / layout_kv / layout_out 可分别指定为 BNSD / BSND。面向 NPU ARCH 3510（Ascend 950PR / Ascend 950DT），测试覆盖 prefill、decode（S1=1）、MTP（1<S1≤4）与 causal 场景共 12 个代表性用例，并与 CANN 内置 FIA 做 msprof 性能对比。

- 【matmul】新增非量化矩阵乘算子 `matmul()`（A16W16）：$C[M,N] = A[M,K] \times B[N,K]^T$，A/B/C 支持 float16、bfloat16，输入为二维 contiguous 矩阵；L2 cache 按矩阵复用情况自适应开关；核内仅支持 transpose_a=False、transpose_b=True。面向 NPU ARCH 3510（Ascend 950PR / Ascend 950DT），性能对比 CANN 包内置 matmul 算子基础模板。

- 【rms_norm】新增 RmsNorm 归一化算子 `rms_norm()`：$rstd = 1/\sqrt{mean(x^2)+\epsilon}$，$y = x \cdot rstd \cdot \gamma$；x/gamma/y 支持 bfloat16、float16、float32，rstd 恒为 float32，归一化轴为 x 的最后 K 维且须与 gamma 形状一致，epsilon 默认 1e-6，返回 `(y, rstd)`；核内按 UB 容量与数据类型自适应选择 UB 全载或沿归一化轴列切分。面向 NPU ARCH 3510（Ascend 950PR / Ascend 950DT），性能对比 CANN BuiltIn 实现。

- 【flash_kda】新增 Kimi Delta Attention prefill 融合算子 `flash_kda()`：融合原始 gate 与 beta 的激活、Q/K 的 L2 normalize 以及完整的 Chunk KDA 计算，按 64-token chunk 切分并在 chunk 间递推状态，返回 `(out, final_state)`；q/k/v/g 为 BF16，initial_state / A_log / dt_bias 为 FP32，Dk = Dv = 128，S 为 64 的整数倍，支持 GQA（Nv % Nk == 0）与 BNSD/BSND layout。面向 Ascend 950，性能对比 H800 上的 FlashKDA 代码。

- 【voxel_conv】新增 VoxelConv 卷积算子 `voxel_conv()`：$C[N,Co,Ho,Wo] = \text{Conv2D}(x[N,Ci,Hi,Wi], filter[Co,CiG,Kh,Kw])$，来自 VoxelNet 点云 3D 目标检测的 Convolutional Middle Layers；数据格式为 NCHW 输入/输出、OIHW 权重，x 与 weight 支持 float16、bfloat16，支持 groups>1 分组卷积，stride ∈ [1,63]、padding 四侧独立 ∈ [0,255]、dilation ∈ [1,255] 且 H/W 独立。面向 NPU ARCH 3510（Ascend 950DT / Ascend 950PR），Conv1D 与 Conv3D 为后续扩展范围。

- 【pointnet_sa】新增 PointNet Set Abstraction 算子 `pointnet_sa()`：$feat[K, D_{out}] = \max_{j} \text{MLP}(points[K,j,D_{in}])$，对应 PointNet++ 点云层次化特征学习 SA 层中的 shared MLP + max-pool 模块（不含 Farthest Point Sampling 与 Ball Query）；输入 points (K,N_per_group,D_in) 与 weight (D_out,D_in)，输出 feat (K,D_out)，支持 float16、bfloat16。面向 NPU ARCH 3510（Ascend 950DT / Ascend 950PR）。

#### 架构重构 Architecture Refactoring
- 【voxel_conv】算子由 `conv2d` 更名为 `voxel_conv`，样例、测试与文档引用路径同步重命名。
