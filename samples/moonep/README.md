# moonep

MoonEP（动态冗余专家均衡）训练全链路的 CANNBot-DSL 样例：无论路由分布如何倾斜，
每个 rank 接收并处理的 token 数固定为 `S × K`。面向 Ascend 950PR/950DT（ARCH 3510）。

## 设计文档与双链路

通信（dispatch/combine）全部复用 moe_ep 四件套（零 kernel 改动，
ops-transformer 含 torch_extension 零修改）；
DSL 自研三个四件套之外的节点：`moonep_plan`（规划）、`prefetch_weight`（副本权重预取）、
`reduce_grad`（副本梯度归并）。语义基准为 `test/moonep/moonep_golden.py`。

moe_ep 的 expert-order + `num_recv_tokens_per_expert[2E]` 与原生 MoonEP 的
NvS 分组 + `cu_seqlens` 天然对应（前缀和即 cu_seqlens），无需任何装配层。

## 算子介绍

参考 MoonEP（Moonshot AI 专家并行通信库）的设计：少量冗余专家由路由输出在线规划，
其权重在专家计算前预取到本地预取槽，计算结果与梯度再按规划归并回各自的 home rank。

```text
dispatch: [S, H] token + 路由信息 → 按专家分组、负载均衡的 [NvS, H]
combine : [NvS, H] 专家输出      → 按路由权重加权回收为 [S, H]
```

| 特性与约束 | 说明 |
| :--- | :--- |
| 负载均衡 | 每个 rank 固定接收 `S × K` 个 token，与路由倾斜程度无关 |
| 在线规划 | 由路由输出在设备侧规划冗余专家放置，产出 `cu_seqlens` 供分组 GEMM 使用 |
| 静态形状 | 通信缓冲区形状静态已知，消除逐层 MoE 的 host 同步 |
| 数据类型 | hidden 为 bfloat16，路由权重为 float32，专家索引为 int32 |
| 支持架构 | NPU ARCH 3510（Ascend 950PR/Ascend 950DT） |

记号约定：`S` 为每 rank 输入 token 数，`K` 为 top-k，`N = S × K` 为 dispatch 前
的 (token, k) 份数，`E` 为 EP 组内专家总数，`R` 为 EP rank 数，`epn = E / R`
为每 rank 本地专家数，`NvS` 为 dispatch 后每 rank 的 token 槽数（v1 实现 `NvS = S × K`，
无 padding），`H` 为 hidden size，`H'` 为专家 FFN intermediate size。

## 与上游 MoonEP 的关系

本样例参考 [MoonshotAI/MoonEP](https://github.com/MoonshotAI/MoonEP)（MIT License）
移植：上游以 CuTe DSL 实现，本样例以 CANNBot-DSL 重写为
NPU ARCH 3510 形态。上游接口与本样例的对应关系：

| 上游 | 作用 | 本样例 |
| :--- | :--- | :--- |
| planning kernel（融在 dispatch 内部） | 均衡规划：冗余专家放置、token 落位、`cu_seqlens` | 独立为 `moonep_plan` |
| `dispatch` | 前向分发，token 直写目标 rank 的专家分组位置 | `ElasticBuffer.dispatch`（复用） |
| `prefetch_weight` | 冗余专家权重从 home rank 预取到目标 rank 预取槽 | `prefetch_weight` |
| `combine` | 前向回收，纯 sum 归并回 token 主序（路由权重由 GMM 侧施加） | `ElasticBuffer.combine`（复用；dispatch 权重随车供 GMM 加权） |
| `reduce_grad` | 训练反向：home rank 归并副本专家在其上产生的权重梯度 | `reduce_grad` |

与上游的差异：上游将规划融在首次 dispatch 内、以不可变的 plan 句柄承载结果；
规划只依赖路由计数与 topk 索引、与 token 载荷无关，故本样例将其独立为
`moonep_plan` 接口——plan 输出为纯张量，可跨调用、跨前后向显式复用。

## 前向与反向视图

```text
传统:   router ─→ dispatch ─────────────→ GMM1 ─→ SwiGLU ─→ GMM2 ─→ combine
MoonEP: router ─→ plan ─→ dispatch ─→ prefetch_weight ─→ GMM1 ─→ SwiGLU ─→ GMM2 ─→ combine

反向:   combine⁻¹ = dispatch（d(out) 散到各副本行，rw 随车）
        GMM 反向：dW 写入 [2epn, W] 视图（前 epn 本地专家 + 后 epn 副本槽）
        dispatch⁻¹ = combine（副本行梯度裸和回 token 主序）
        reduce_grad：各副本槽 dW 归并回 home 真实行并清零槽位
        （plan/vtopk 复用前向同一份，反向免重新规划、免重新预取）
```

GMM 权重视图为 `[2*epn, H, H']`（本地专家权重 + 预取槽拼接），分组 GEMM 按
`cu_seqlens` 分段使用。反向与前向镜像复用：dispatch 的反向等价于 combine
（K 份梯度归和回 token 主序），combine 的反向等价于复用 plan 的 dispatch
（免重新规划、免预取）。训练反向与前向镜像复用（dispatch⁻¹ = ep combine 裸和，combine⁻¹ = ep dispatch 散梯度）。

## 接口与参数

### ep 复用链（主线）

规划复用 DSL 的 `moonep_plan`，通信由 `cann_ops_transformer` 的 `ElasticBuffer`
dispatch/combine 承载（无需任何适配层）：

```python
from cann_ops_transformer.ops import ElasticBuffer

dst_n, cu, etc, src_n, vtopk = moonep_plan(ctx, tokens_per_expert, topk_experts_sk)
buf = ElasticBuffer(ep_group, num_max_tokens_per_rank=nmtr, hidden=H, num_topk=K)
recv_x, _, recv_w, handle = buf.dispatch(
    hidden_sh, topk_idx=vtopk.view(S, K), topk_weights=rw_sk,
    num_experts=2 * E, num_max_tokens_per_rank=nmtr, do_cpu_sync=False,  # nmtr = max(S, int(S*K/R*1.25)+64)
)  # vtopk 虚拟专家路由 + 权重随车展开（原生 MoonEP 的 NvS/route_weights 对应物）
# ... GMM 以 num_recv_tokens_per_expert 前缀和（=cu_seqlens）分组计算，
#     并以 recv_w 施加路由权重（moe_ep 契约：权重归 GMM） ...
out, out_w = buf.combine(y_weighted, handle, topk_weights=recv_w)  # 纯 sum + 权重回收
```

### DSL 自研节点

| 接口 | 实现文件 | 使用场景 |
| :--- | :--- | :--- |
| `moonep_plan` | moonep_plan.py | 规划：由路由计数与 topk 产出 plan 五件套 |
| `prefetch_weight` | prefetch_weight.py | 权重预取：按 plan 将冗余专家权重搬到本地预取槽 |
| `reduce_grad` | reduce_grad.py | 训练反向：归并副本专家权重梯度回 home rank 并清零槽位 |

通信分发/回收无 DSL 算子——由 `ElasticBuffer` dispatch/combine（复用）承载。

#### moonep_plan

```python
dst_n, cu_seqlens, experts_to_copy, src_n, vtopk = moonep_plan(
    context, tokens_per_expert, topk_experts_sk,
)
```

| 参数 | shape | dtype | 说明 |
| :--- | :--- | :--- | :--- |
| `context` | - | MoonepContext | 通信上下文，见「通信上下文」 |
| `tokens_per_expert` | `[E]` | `torch.int32` | 本 rank 各专家 token 计数 |
| `topk_experts_sk` | `[S, K]` | `torch.int32` | top-k 专家索引 |
| 返回 `dst_n` | `[N]` | `torch.int32` | **零占位**（vtopk-only 快路径不计算；原语义：落位编码 `rank × NvS + offset`） |
| 返回 `cu_seqlens` | `[2*epn]` | `torch.int32` | 分组 GEMM 分段偏移（本地专家 + 预取槽，均按专家 id 升序） |
| 返回 `experts_to_copy` | `[R, epn]` | `torch.int32` | 各 rank 需预取的远程专家清单，空槽 -1 |
| 返回 `src_n` | `[N]` | `torch.int32` | **零占位**（vtopk-only 快路径不计算；原语义：slot 反查表） |
| 返回 `vtopk` | `[N]` | `torch.int32` | 虚拟专家 id（扩展空间 `E' = 2E`），供 ep 复用链路由改写 |

#### prefetch_weight / reduce_grad

```python
prefetch_weight(context, experts_to_copy, real_weights, local_weights, epn)
reduce_grad(context, experts_to_copy, grad_view, epn)
```

| 参数 | shape | dtype | 说明 |
| :--- | :--- | :--- | :--- |
| `experts_to_copy` | `[R, epn]` | `torch.int32` | `moonep_plan` 的预取/归并清单 |
| `real_weights` | `[epn, H*H']` | `torch.bfloat16` | 真实本地专家权重（普通张量，home 直推的数据源；多组用 list） |
| `local_weights` | `[2*epn, H*H']` | `torch.bfloat16` | 对称区权重视图（`create_weight_view` 创建）：前 `epn` 行本地专家（调用前就位），后 `epn` 行预取槽（本函数填充） |
| `grad_view` | `[2*epn, H*H']` | `torch.float32` | 梯度视图：前 `epn` 行真实专家，后 `epn` 行副本槽 |
| `epn` | 标量 | int | 每 rank 本地专家数（行宽由张量 shape 自取） |

两接口均原地生效、无返回值；gate / up / down 三个 projection 各调一次
（或传 list 共享清单）。每次调用末尾含一次**kernel 内联的**对称堆自旋完成屏障
（barrier_tail：epoch 直推 + flag 收齐，融合在 push/recv kernel 尾段，不走
HCCL 集合通信、无独立 launch）；调用末尾为发起流级等待（不等无关流）。
`reduce_grad` 语义：`grad_view[e_local] += Σ 副本梯度`，随后副本槽清零。
对称堆容量在 `create_moonep_context` 与 `create_weight_view` 处前置校验，
不足即报所需/现有容量。对称窗权重视图进 grouped_matmul 前先
`materialize_weights(view)` 落地普通显存（GMM host 路径限制）。
两接口的发送/接收清单按 `experts_to_copy` 身份单槽缓存——同一 plan 重复调用
无 GM→CPU 回读与清单重建，新 plan 自动重建。

### 使用约束

- 所有张量须连续且位于同一 NPU；通信形态为 EP 组内 all-to-all（UB 链路 URMA）。
- v1 `NvS = S × K`（无 padding），`E ≤ 1024`、`R ≤ 64`、`E % R == 0`；
  `S × K` 不设上限且任意 N 原生整除（topk host 侧直拷 workspace、
  分块粒度降档到 N 的因子，无尾补机制）。
- ⚠️ **通信域分离**：`ElasticBuffer`（dispatch/combine）与 `create_moonep_context`
  （plan/prefetch/reduce）**必须各用独立通信器**（各自 `dist.new_group`，且均不可用
  `dist.group.WORLD`）。共用同一 HCCL 通信器即共用同一块对称堆，双方各自圈地会
  互相踩踏，dispatch 自旋死锁（507899）。`create_moonep_context` 与首次
  `moonep_plan` 会对本约束做运行期校验，违例在创建期秒级报错而非挂死。
  专用通信域约束与官方 MC2 算子同级（`MoeDistributeDispatchV2`/`CombineV2`
  同样要求"该通信域中不允许有其他算子"）。
- **多槽窗口池**：`create_weight_view(..., group=g, groups=G, slot=s, slots=S)`——
  槽 s 组 g 视图基址 = 发布区 + (s*G+g)*整视图步长。双槽轮转（层 L 用 L%2）
  使 GEMM 可直读当前槽视图、下一层 prefetch 写另一槽，免集成侧快照拷贝
  （单槽并发覆写实测 SIGSEGV）。缺省 slot=0/slots=1 与单槽布局字节一致；
  槽位数由首个调用定型。内存代价：S 倍发布区（K3 双槽 3 组约 +1.2GB）。
- **plan 返回路径零同步**：尾部 D2H 会阻塞紧随的 dispatch 发射
  （多层训练实测 941ms 级）；副本槽溢出检查延迟到 prefetch/reduce 尾部
  （sym_barrier 后微秒级）执行。契约：**链路内算子未报错 ⟹ 所有已执行
  plan 的 cu_seqlens[-1] == num_rows**（写前保护仍即时生效，极端倾斜的
  报错从 plan 内移到同层 prefetch/reduce 尾部）。
- ⚠️ **`nmtr = max(S, int(S×K/R×1.25)+64)`**：moe_ep per-rank 容量同时约束
  发送侧（x.dim0 ≤ nmtr）与接收侧（per-peer 期望 S×K/R + 泊松余量）。
- ⚠️ **topk 每 token 的 K 个专家须互不相同**（标准 router 语义）。
- combine 用对应 dispatch 的 handle；反向 dispatch 复用前向同一 plan 的 vtopk。
- 反向已提供；上游的边缘开关（`zero_copy` 视图、
  `async_finish` 事件流）v1 未实现。

### DSL 编译器已知约束（踩坑实证，写 kernel 前必读）

- `cannbotdsl.range` 只在 @kernel/@host/@jit 的 AST 边界内可用。
- constexpr 循环内含 `mem_copy` 会撞编译器 SIGSEGV；同 kernel 须单 vf 区。
- **tile 搬运三种边界**：多行 2D tile + 运行时首坐标 = 静默垃圾数据（编译器已知限制）；
  **多行跨步 tile（如 `tile_slice(t, (e_per,R), (e_lo,0))`）= 设备异常或静默错数据**
  ——只允许整段拷贝，或"外层 extent=1 的 tile + 循环 IV"（`(1,R)@(e,0)` 逐行推进）；
  单行 tile 的坐标用运行时值在多 block 下已验证可靠。
- 原子累加的目标 tile 有 **32B 粒度要求**（不足须 padding 到 32B 倍数，
  见 plan 的 per-block 局部 loads 行方案——本仓直接避免了原子路径）。
- UB 标量点读只在 vf(simt) 或完整 S↔MTE2 双向握手（notify+wait 成对）后进行。
- **流序写入纪律**：跨调用复用的设备缓冲只允许 kernel 全量覆写或设备侧
  `copy_`，禁止 host 侧 H2D 直写复用缓冲（会与上一轮仍在读的 kernel 竞争，
  表现为轮询/握手永久等待）；epoch 类小标量每次新分配。
- 已验证形态：单机 4 卡与 8 卡全机（Ascend 950PR）——默认门禁 13 用例
  （前向/反向/预取各 3 代表 shape + 多组 + 多槽 + 归并 + bf16）4P/8P 全 PASS，
  `--full` 全量矩阵 27 用例按需执行（倾斜/中间档 shape）；推荐配置
  （S=4096/K=8/E=256/H=7168）三档 maxvio（0.1/1/8）性能基准全通过。

## 通信上下文

各接口共享一份 EP 组级通信上下文（`MoonepContext`，对应上游 `Buffer` 对象），
由 `create_moonep_context()` 创建：内含全 rank 窗口地址视图、通道句柄与容量参数
（`ep_rank_id` / `R` / `S` / `K` / `E` / `NvS`）。上下文以显式入参传入各接口，
不依赖全局状态；EP 组初始化时创建一次、全程复用。

## 目录结构

```text
samples/moonep/
├── README.md               # 本文档（接口参考）
├── moonep_context.py       # 通信上下文/容量校验/流级 barrier 助手/跨流护栏（共享）
├── moonep_plan.py          # 规划（7 kernel：MoonepPlanExchange 交换/GreedyInit 初态/Greedy 贪心+cu/Tables 逐 rank 表/Count 计数/Pfx 查表/Assign 段倒计时）
├── prefetch_weight.py      # 权重预取（清单 kernel + 融合多组直推 + 内联 barrier 尾段）
├── reduce_grad.py          # 副本梯度归并（清单 kernel + 两段式 UB 累加 + 内联 barrier 尾段）
└── npu_blob.py             # from_blob 扩展（对称区权重视图用；C++ 源内嵌）
```

对应测试集中在 `test/moonep/`（golden + DSL 链 NPU 用例 + ep 复用链 E2E）。

> **已知限制**：`npu_blob`（from_blob 扩展）为临时桥接——torch_npu 未暴露
> 设备地址→张量的公共 API，上游提供后删除。

## 快速开始

由平台预置 CANN、PyTorch、torch_npu 和 DSL wheel，见[依赖检查脚本](../../install_deps.sh)。
ep 复用链还需已安装含 moe_ep 封装的 `cann_ops_transformer`（ElasticBuffer）。

```bash
source ${install_path}/ascend-toolkit/set_env.sh
# 最小可运行示例见 test/moonep/test_moonep_ep.py（plan → ElasticBuffer dispatch/combine）
python -m torch.distributed.run --nproc_per_node=8 \
    -m pytest test/moonep/test_moonep_ep.py -q
```

## 精度验证

```bash
python3 -m pytest -q test/moonep          # golden 自测（CPU 常跑）；NPU 用例自动 skip
```

golden 参考实现见 [moonep_golden.py](../../test/moonep/moonep_golden.py)：纯 torch
单进程模拟整个 EP 组（含贪心冗余专家规划器），定义语义契约——每 rank 恰好接收
`S × K` 个 token、combine 按路由权重加权归并回 token 主序。

| 测试 | 内容 | 形态 |
| :--- | :--- | :--- |
| `test_moonep.py` | golden 自测 4 组（均衡不变量 / cu_seqlens 一致性 / 往返一致 ×2） | CPU 常跑 |
| `test_moonep_ep.py` | ep 复用链 E2E 3 组（前向 / 反向 / 含权重全链），三档路由参数化 | `torchrun --nproc_per_node≥2`，无 NPU 或未装 `cann_ops_transformer` 自动 skip |

已验证记录（8×Ascend950PR，DSL 0.7.1）：全量矩阵 27 用例全 PASS；vtopk 全量对拍
N=32768 mismatch=0；推荐配置三档 maxvio 全通过。

## 性能验证

8P 推荐配置（S=4096/K=8/E=256/H=7168，孤立口径）：plan ~1.6ms（多 block 化
移植后，19M+ tokens/s，三档 maxvio 平坦）、dispatch 246GB/s、combine 177GB/s
（孤立口径）、prefetch 1.1ms / reduce_grad 4.9ms（背靠背口径）。背靠背口径存在
~3× URMA 跨迭代干扰（背靠背口径固有现象，孤立口径为准）；**清单由设备侧 kernel 从
etc 生成（host 零回读/零扫描/零 H2D，plan 返回路径零同步）**；plan/prefetch/reduce
编译键均只含形状与布局——路由变化不触发重编译（清单为定长数组 + 运行时计数）。
