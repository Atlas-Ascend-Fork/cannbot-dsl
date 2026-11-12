# MoonEP 算子

专家并行（EP）五算子套件：`moonep_plan`（均衡规划）、`prefetch_weight`
（跨卡权重搬运——home rank 直推到消费 rank 的对称预取槽，支持
mxfp8 fp8 权重 + ue8m0 scale 组）、`reduce_grad`（副本梯度归并回 home）、
以及 `dispatch`/`combine`（token 路由分发/回收，经 ElasticBuffer）。
`MoonepBuffer` 将五算子收口为统一对象（构造即拥有全部通信资源）。
配套模块提供通信上下文、对称堆布局、显存视图和显存统计。
设计参考 [MoonEP](https://github.com/MoonshotAI/MoonEP)。

## 规划（moonep_plan）

`moonep_plan(context, tokens_per_expert, topk_experts_sk)` 根据本 rank 的
专家 token 计数和 top-k 路由索引完成均衡规划。跨 rank 交换计数后，在 demand
矩阵上执行 cap 均衡贪心（一卡一源），生成可跨调用、跨前后向复用的规划五件套：
接收分段、副本清单和虚拟专家 id 投影，供 `prefetch_weight` 及 ep
dispatch/combine 复用链使用。

```python
from moonep_context import create_moonep_context
from moonep_plan import moonep_plan

context = create_moonep_context()
dst_n, cu_seqlens, experts_to_copy, src_n, vtopk = moonep_plan(
    context, tokens_per_expert, topk_experts_sk,
)
```

| 参数 / 返回 | shape | dtype | 说明 |
| :--- | :--- | :--- | :--- |
| `tokens_per_expert` | `[E]` | `torch.int32` | 本 rank 各专家 token 计数 |
| `topk_experts_sk` | `[S, K]` | `torch.int32` | top-k 专家索引 |
| `cu_seqlens` | `[2*epn]` | `torch.int32` | 分组 GEMM 分段偏移（本地专家 + 预取槽，均按专家 id 升序） |
| `experts_to_copy` | `[R, epn]` | `torch.int32` | 各 rank 需预取的远程专家清单，空槽 -1；每行至多一个远程 home rank |
| `vtopk` | `[N]` | `torch.int32` | 虚拟专家 id（扩展空间 `E' = 2E`），供 ep 复用链路由改写 |
| `dst_n` / `src_n` | `[N]` | `torch.int32` | 零占位（vtopk-only 快路径不计算） |

其中 `N = S*K`，`epn = E/R`。

形状约束：`E <= 1024`、`R <= 64`、`E % R == 0`；`S*K` 任意规模原生整除
（分块粒度降档到 N 的因子，无尾补机制）。

plan 返回路径零同步：接收布局不变量（`cu_seqlens[-1] == N`）延迟到下一个
moonep 算子尾部校验。测试/验收需要当场熔断时，显式调用
`check_pending_overflow(context)`。

## 权重预取（prefetch_weight）

`prefetch_weight(context, experts_to_copy, local_weights, epn, *,
local_scales=None, slot=0, slots=1, async_finish=False)` 按调用者提供的
专家清单，把远程 home rank 的专家权重（+ 可选 mxfp8 scale）写入本地预
取槽，返回变长：sync 只返回 `views`（**(weights, scales)** 裸二元组），
（同步形态 event=None）。

`local_weights` 为 **2 组或 3 组**张量（list/tuple，N-D 直入——行宽=
尾维展平宽）——组数与各组行宽直接从 `local_weights[g]` 解析；dtype 须
全组一致且为 bf16 / float8_e4m3fn / float8_e5m2（mxfp8 原生 fp8
dtype，dtype 终态——uint8 已退役；dtype 与构造
`weight_dtype` 一致——E1）：

- 三矩阵等宽：`(w_gate, w_up, w_down)`，每组 `[epn, W]`；
- **交织两矩阵**：`(W13, W_down)`——训练侧把 gate/up 融合成
  `W13 = cat(gate, up, dim=1)`（每专家一行 `2W`，行内 gate 前 up 后），
  一次 GMM 算完交给 SITU，省去消费端 concat。`W13` 每专家以一条 `2W`
  连续 DMA 推送；两组槽区总字节数与三组等宽相同（`2W + W = 3W`）。

**mxfp8 形态（`weight_dtype="e4m3fn"/"e5m2"`，双向硬契约：scales
⟺ fp8）**：`local_scales` 与 `local_weights` 同组同序（每组 `[epn, ...]`
连续 float8_e8m0fnu——OCP MX 标准 ue8m0 编码，行宽 Sg = 每专家 scale
字节数，**须为 32B 倍数**——E3，不足 pad
到 `ceil(Sg/32)*32`）。scale 与权重共用全部控制面（同一张
`experts_to_copy` 清单、同一次 launch、同一逐 peer barrier、同
per-expert 连续 DMA）；行/列存储方向不进契约（"每专家连续字节行"抽象
吸收，消费端量化 GEMM 按同布局读视图）。bf16 权重不传也不可传 scale。

视图由 prefetch 按 `local_weights` 的形状创建（同 `(sizes, slot, slots)`
缓存复用）并返回 `(weights, scales)` 裸二元组：`weights[g]` 形状
`[2*epn, Wg]`（weight_dtype；
前 `epn` 行真实本地专家、后 `epn` 行预取槽）、`scales[g]` 形状
`[2*epn, Sg]` float8_e8m0fnu（行序与权重同构，未量化时 `()`）；槽内按各组
（字节）前缀和紧密排布（组宽可不同；scale 组跟同槽权重组之后——
与权重同槽轮转）。参数建在视图 home 行后，后续调用可传 home 行切片
（data_ptr 同址，sync_transport 跳过拷贝
——零拷贝形态）。


`experts_to_copy` 为设备端 int32 张量 `[rank_size, epn]`，空槽填 `-1`；
每个目标 rank 的非空槽至多来自同一个 home rank。

通信上下文使用独立 HCCL 通信器，不能与其他占用同一对称堆的算子共用。
规划算子调用后自动设置 `context.last_plan_shape`；独立使用预取时调用者
须自行设置该字段。

```python
from moonep_context import create_moonep_context
from prefetch_weight import prefetch_weight

context = create_moonep_context()
context.last_plan_shape = (num_tokens, top_k)
weights, scales = prefetch_weight(context, experts_to_copy, (w1, w3, w2), epn)
# weights = 3 × [2*epn, W]；scales == ()
# 交织形态：weights, scales = prefetch_weight(ctx, etc, (w13, w2), epn)
# mxfp8：weights, scales = prefetch_weight(ctx, etc, (w13_fp8, w2_fp8), epn,
#                                  local_scales=(s13, s2))

```

在仓库根目录设置 `PYTHONPATH="$PWD/samples/mc2/moonep:$PYTHONPATH"`。
接口支持多槽视图（`slot`/`slots` 参数）。`async_finish=True` 返回完成事件，
调用方通过事件等待汇合。

## 梯度归并（reduce_grad）

`reduce_grad(context, experts_to_copy, local_grads, epn, *, slot=0,
slots=None, async_finish=False)` 把各 rank 持有的副本 dW 归并回 home rank
（pull 协议），返回变长：sync 只返回 `home_views`——`home_views[g]` 为各组
`[epn, Wg]` 视图（与输入张量同存储，归并完成后即全量 dW）。

**输入/输出契约**：

| 项 | 形态 | 说明 |
| --- | --- | --- |
| 输入 `local_grads` | 2/3 组 `[2*epn, Wg]`（或 N-D `[2*epn, ...]` 直入） | GMM 对全量权重视图（home+副本行）反传的原生 full-dW 形态；**前 epn 行 = home 种子**（本 rank 自有专家的 dW 初值，归并目标行），**后 epn 行 = 副本 dW**（本 rank 持有的远端专家副本梯度，被归并的行）；组序与 prefetch 的 `local_weights` 同序（交织型传 `[2*epn, 2W] + [2*epn, W]` 两组） |
| 输入 `experts_to_copy` | 设备端 int32 `[rank_size, epn]` | 同一张 plan 清单（与 prefetch 共用——prefetch 是 owner push 权重给消费方，reduce 是 owner pull 梯度回属主，同清单两方向） |
| 输出 `home_views` | 2/3 组 `[epn, Wg]` | **与输入张量同存储的 home 段切片**（`t[:epn]`，原位归并的直接读柄）——`t[epn:]` 副本段保持输入原值不被消费；sync 返回即全组归并完成，直接读即全量 dW |
| 输出 `event`（async） | `torch.npu.Event` | `async_finish=True` 时返回 `(home_views, event)`；wait_event 汇合后读结果 |

`local_grads` 的 dtype 取 bf16/fp32：**裸模块 API 每调用从张量解析（同 ctx
双 dtype 合法）；MoonepBuffer 路径以构造 `grad_dtype`（缺省 bf16）为定型值
校验**。

数据流（全部在本算子内，prefetch 不搬任何梯度）：

```
① 入槽：副本行 t[epn:] 拷入对称堆梯度槽 [epn, Wg]（本地写，零通信）
② 屏障（pull kernel 头部 per-peer flag 握手）：全组入槽互可见
③ 拉取归并：owner 逐组远拉各持副本 rank 的槽行（xfer.get → scratch
   → UB）→ mem_copy(atomic_add=True) 原子加进 home 行 t[:epn]
④ 收尾：finish_op（async_finish=False 阻塞 / True 返回发起流 Event）
```

单 rank（R=1）无副本可归并，返回 `(home_views, None)`。原子累加顺序
不保证逐位确定（对拍用例须用容差判据）；bf16 每行累加数 ≤ R 的精度界
见设计文档 §4.5。

```python
from reduce_grad import reduce_grad

home_views = reduce_grad(ctx, experts_to_copy, [dw1, dw3, dw2], epn)
# 交织形态：home_views = reduce_grad(ctx, etc, [dw13, dw2], epn)
```

## 验证

需要匹配的 CANN、PyTorch/torch_npu、HCCL 和至少两张支持设备。

规划算子验收（前向 E2E 对 golden，另含 Count 阶段 counts 逐 chunk 对拍）：

```bash
python -m torch.distributed.run --nproc_per_node=8 -m pytest \
    test/mc2/moonep/test_moonep_plan.py -q
```

8 卡为验收标准形态，其他 `>=2` 卡环境也可运行。用例较多时可按 `-k`
将 E2E、counts 审计和 host 约束用例分批执行，避免超出单任务时限。

预取算子独立验收检查数据正确性及 Buffer 的 `(mem_loc, dtype, layout)` 调用：

```bash
python -m torch.distributed.run --nproc_per_node=4 -m pytest \
    test/mc2/moonep/test_prefetch_weight.py -q
```

其余套件（reduce_grad 归并回归与 E5/容量负例、buffer 全矩阵、rank graph
层选择逻辑、mem 端到端训练）：

```bash
python -m torch.distributed.run --nproc_per_node=8 -m pytest \
    test/mc2/moonep/test_reduce_grad.py -q
python -m torch.distributed.run --nproc_per_node=8 -m pytest \
    test/mc2/moonep/test_moonep_buffer.py -q
python -m torch.distributed.run --nproc_per_node=8 -m pytest \
    test/mc2/moonep/test_moonep_rank_graph_layers.py -q
```

⚠️ **每个测试文件独立进程/独立作业运行**：同进程先跑 mem（EB@WORLD
常驻）再跑 pfw 的 WORLD ctx 创建会报 HCCL 设备错误（实证 HCCLUtils
code 4——非代码回归，独立进程稳定通过）。R5 fp32 用例以 `--full`
门运行（xfail 登记 U5）。

设备性能基准见
[bench_prefetch_weight.py](../../../test/mc2/moonep/bench_prefetch_weight.py)，
可配置输入形状、预热和采样次数。


## MoonepBuffer 统一封装

`MoonepBuffer` 将 moonep 五算子（plan / dispatch / prefetch_weight /
combine / reduce_grad）收口为单一对象，对标 GPU MoonEP 官方 `Buffer`
形态：构造即拥有全部通信缓冲、plan 句柄贯穿五算子、destroy 显式释放。

### 构造

前五个 positional `(S, H, Hp, K, E)`：`(S, H, K, E)` 与 MoonEP
`Buffer(S, H, K, E, ...)` 同名同序，`Hp`（专家 FFN 中间维）插在 H 后；
kwargs 按"通信域 → 模式开关（dtype）→ slots 收尾"排列。

```python
import torch
import torch.distributed as dist
import torch_npu
from torch_npu._C._distributed_c10d import ProcessGroupHCCL
from moonep_buffer import MoonepBuffer

# ① 用静态方法算两组 buffer 大小（在创建 ProcessGroup 前调用）：
moon_ep_mb = MoonepBuffer.get_moon_ep_ccl_buffer_size(8192, 3584, 3072, 16, 224)
moe_ep_mb = MoonepBuffer.get_moe_ep_ccl_buffer_size(
    world_size=dist.get_world_size(),
    num_max_tokens_per_rank=8192,
    hidden=3584,
    num_experts=224,
    topk=16,
)

# ② 用返回值建两个独立 HCCL 通信域：
opts = ProcessGroupHCCL.Options()
opts.hccl_config = {"hccl_buffer_size": moon_ep_mb}
moonep_group = dist.new_group(
    ranks=list(range(dist.get_world_size())),
    backend="hccl",
    pg_options=opts,
    use_local_synchronization=True,
)
ep_opts = ProcessGroupHCCL.Options()
ep_opts.hccl_config = {"hccl_buffer_size": moe_ep_mb}
ep_group = dist.new_group(
    ranks=list(range(dist.get_world_size())),
    backend="hccl",
    pg_options=ep_opts,
    use_local_synchronization=True,
)

# ③ 构造（传 ProcessGroup——仅支持组对象）：
buf = MoonepBuffer(
    8192, 3584, 3072, 16, 224,  # (S, H, Hp, K, E)
    moonep_group=moonep_group,
    ep_group=ep_group,
)
```

构造即完成：堆布局核对（按 3W 总宽上限预留；
fp8 形态含 scale 区上界 3×ceil(W/32) 字节/专家行）→ HCCL 子组创建 →
ctx 定型（window_slots + weight/grad dtype 定型值——模块侧 E1/E5 校验
基）→ EB 通信域分离 → nmtr 定型（num_max_tokens_per_rank = S，发送侧
per-peer 上界）。

**组数运行期自适应**：权重/梯度组数不再构造期固定——prefetch/reduce
按传入张量解析（3 组等宽 `(W, W, W)` 或交织 2 组 `(2W, W)`，堆字节
两者相同）。

### 权重矩阵

三组权重为常规 NPU 参数张量，行宽统一 W = H×Hp（gate/up `[epn, Hp, H]`、
down `[epn, H, Hp]`）；**首次调用 prefetch 传入权重本体**（视图按各组
行宽创建并返回，参数拷入 home 行），此后参数本体即视图 home 行
（零拷贝复用；副本行由 prefetch 按计划填充）：

```python
import torch
import torch_npu

epn = buf.epn
w1 = torch.empty(epn, Hp, H, dtype=torch.bfloat16, device="npu")  # gate（如 nn.Parameter）
w3 = torch.empty(epn, Hp, H, dtype=torch.bfloat16, device="npu")  # up
w2 = torch.empty(epn, H, Hp, dtype=torch.bfloat16, device="npu")  # down
# 首步（plan 后）：
weights, scales = buf.prefetch_weight(plan, local_weights=(w1, w3, w2))  # N-D 直入
# 交织 2 组形态：weights, scales = buf.prefetch_weight(plan, local_weights=(w13, w2))
#   （w13 = cat(w1, w3, dim=1) 每专家一行 2W，gate 前 up 后）
# mxfp8 形态（weight_dtype="e4m3fn"/"e5m2"）：
#   weights, scales = buf.prefetch_weight(plan, local_weights=(w13_fp8, w2_fp8),
#                                  local_scales=(s13, s2))  # E2：fp8 ⟺ 必传（scales 为 e8m0fnu）
```

### 五算子链（训练前反向端到端，FFN 侧全 NPU 算子）

分组 GEMM 经 `offset` 直消费 device 侧 `cu_seqlens`（分组边界零 host
同步）；激活用 fused SwiGLU：

```python
for step in range(num_steps):
    # ---- 前向 ----
    plan = buf.plan(tpe, topk)                    # 规划（可解包五件套）
    recv, recv_w, handle = buf.dispatch(plan, x, scores, do_cpu_sync=False)  # 免 host 同步
    if step == 0:
        # N-D 直入（内部展平）——首调建视图
        weights, scales = buf.prefetch_weight(plan, local_weights=(w1, w3, w2))
    else:
        # 次步起零拷贝：参数已驻留视图 home 行——传 home 行切片（同址跳过）
        buf.prefetch_weight(plan, local_weights=[w[:epn] for w in weights])
    n, cap = S * K, recv.shape[0]                 # 有效行恒 S*K（plan 均衡不变量）
    # —— FFN 前向：分组 GEMM + SwiGLU（NPU 算子，cu_seqlens 保持 device）——
    wv1 = weights[0].view(2 * epn, Hp, H)           # home+副本全量视图
    wv3 = weights[1].view(2 * epn, Hp, H)
    wv2 = weights[2].view(2 * epn, H, Hp)
    gate = torch_npu.npu_grouped_matmul(recv[:n], wv1, offset=plan.cu_seqlens)
    up = torch_npu.npu_grouped_matmul(recv[:n], wv3, offset=plan.cu_seqlens)
    act = torch_npu.npu_swiglu(torch.cat([gate, up], dim=-1))
    y = torch_npu.npu_grouped_matmul(act, wv2, offset=plan.cu_seqlens)
    combine_in = torch.zeros(cap, H, dtype=torch.bfloat16, device=y.device)
    combine_in[:n] = (y.float() * recv_w[:n].unsqueeze(-1)).to(torch.bfloat16)
    out, out_w = buf.combine(combine_in, handle, topk_weights=recv_w)  # 归并回 token 序

    # ---- 反向 ----
    g_recv, _, handle_b = buf.dispatch(plan, g_out, topk_weights=None)  # 反向散梯度（复用 plan 免重规划；rw 不随车）
    g_in = torch.zeros(g_recv.shape[0], H, dtype=torch.bfloat16, device=g_recv.device)
    # ↑ 容量取反向 dispatch 自身 recv（缺省精确分配）——勿复用前向
    # do_cpu_sync=False 的 worst-case cap：容量与 handle metadata 失配
    # 会触发 EB packed int32 形状校验失败
    g_x, _ = buf.combine(g_in, handle_b)         # 梯度裸和回 token 序（FFN 反传行填入 g_in[:n]）
    # FFN 反向：分组 GEMM 反传产出 full-dW——张量由 GMM 反向分配（行数 2*epn：
    # 前 epn 行 home 梯度种子、后 epn 行各持有 rank 的副本梯度），直入 reduce：
    dw1 = torch._grouped_mm(g_recv[:n].mT, gate)            # [2*epn, Hp, H] 反传形态
    dw3 = torch._grouped_mm(g_recv[:n].mT, up)              # [2*epn, Hp, H]
    dw2 = torch._grouped_mm(g_recv[:n].mT, act)             # [2*epn, H, Hp]
    # 副本行入槽由 reduce 内部完成；归并结果在 dws[g][:epn] 原位读取
    dws = buf.reduce_grad(plan, local_grads=[dw1, dw3, dw2])  # sync 单值
```

### mxfp8 完整样例（fp8 权重 + ue8m0 scale + 梯度归并）

mxfp8 形态下 prefetch 返回的 `(weights, scales)` 成对喂给量化 GEMM——
scale 视图与权重视图同槽同序（`[2*epn, Sg]`，行序 home+副本同构），
消费端按分组边界切片：

```python
import torch
import torch_npu

# 构造（fp8 形态）：
buf = MoonepBuffer(S, H, Hp, K, E, moonep_group=moonep_group, ep_group=ep_group,
                   weight_dtype="e4m3fn", grad_dtype="bf16")
epn, W = buf.epn, H * Hp
Sg = (2 * W + 31) // 32          # 交织 W13 行的 scale 宽（32B 对齐后）

# 参数与 scale（首次传入本体；scale 为 float8_e8m0fnu）：
w13 = torch.empty(epn, 2 * W, dtype=torch.float8_e4m3fn, device="npu")   # cat(gate, up) 每专家一行
w2 = torch.empty(epn, W, dtype=torch.float8_e4m3fn, device="npu")
s13 = torch.empty(epn, Sg, dtype=torch.float8_e8m0fnu, device="npu")     # ue8m0 block scale
s2 = torch.empty(epn, Sg // 2, dtype=torch.float8_e8m0fnu, device="npu")

for step in range(num_steps):
    plan = buf.plan(tpe, topk)
    if step == 0:
        # 首步传参数本体：视图创建 + 参数拷入 home 行（必有一次拷入，非零拷贝）
        weights, scales = buf.prefetch_weight(
            plan, local_weights=(w13, w2), local_scales=(s13, s2)
        )
        # 参数迁移驻留堆上（此后优化器直接写 home 行——传本体即天然同址）：
        w13, w2 = weights[0][:epn], weights[1][:epn]
        s13, s2 = scales[0][:epn], scales[1][:epn]
    else:
        # 次步起零拷贝：传 home 行切片（data_ptr 与 home 段同址 ⇒ sync 跳过，
        # 含 scale 组）；若此处仍传原始 (w13, w2, s13, s2)（未迁移的独立存储），
        # 则每步重拷一次 home 行——正确但多 3×epn×W×1B 的 D2D
        buf.prefetch_weight(
            plan, local_weights=(w13, w2), local_scales=(s13, s2)
        )

    # —— 前向：量化 GEMM（weights[k] 与 scales[k] 成对按组消费）——
    w13v = weights[0].view(2 * epn, 2 * W)     # home+副本全量行
    w2v = weights[1].view(2 * epn, W)
    s13v = scales[0].view(2 * epn, Sg)         # 与 w13v 同槽同序
    s2v = scales[1].view(2 * epn, Sg // 2)
    gate_up = torch_npu.npu_grouped_matmul_quant(   # 量化 GEMM：fp8 权重 × ue8m0 scale
        recv[:n], w13v, scale=s13v, offset=plan.cu_seqlens
    )
    act = torch_npu.npu_swiglu(gate_up)
    y = torch_npu.npu_grouped_matmul_quant(
        act, w2v, scale=s2v, offset=plan.cu_seqlens
    )
    # combine 加权与 bf16 路径同款（略）

    # —— 反向：GMM 反传产出 full-dW（fp8 路径梯度仍为 bf16/fp32——ue8m0 无梯度）——
    dw13 = torch._grouped_mm(g_recv[:n].mT, gate_up)   # [2*epn, 2W] 由 GMM 反向分配
    dw2 = torch._grouped_mm(g_recv[:n].mT, act)        # [2*epn, W]
    dws = buf.reduce_grad(plan, local_grads=[dw13, dw2])  # 归并回 home（scale 不参与）
    # dws[0][:epn] 即全量 dw13 → 优化器更新后重新量化（weight/scale 更新属调用方）
    # 重量化结果写回 w13/w2/s13/s2（堆 home 行本体）→ 次步 prefetch 零拷贝直推
```

方法签名（与 MoonEP 同位）：`prefetch_weight(plan, async_finish, *,
local_weights, local_scales, slot)`、`reduce_grad(plan, async_finish, *,
local_grads)`——`local_grads` 为必传 keyword（**不支持 scales**——梯度只有
bf16/fp32，ue8m0 无梯度）。梯度搬运四步全在 reduce_grad 内部（①副本行入槽
→ ②组屏障 → ③owner 远拉原子直累 home 行 → ④收尾），prefetch 不搬任何
梯度。

### 释放

```python
buf.destroy()  # 幂等；缺省不销毁子组/EB（池化复用）
buf.destroy(release_group=True, release_eb=True)  # 显式释放全部资源
```

### 属性

| 属性 | 类型 | 说明 |
|---|---|---|
| `H` | int | hidden size（= MoonEP 的 H；dispatch 载荷宽 + 专家 FFN 输入维） |
| `Hp` | int | 专家 FFN 中间维（W = H × Hp） |
| `weight_dtype` / `grad_dtype` | torch.dtype | dtype 定型值（构造期；E1/E5 校验基） |
| `nmtr` | int | num_max_tokens_per_rank（= S，发送侧 per-peer 上界） |
| `slots` | int | 权重/scale 窗口槽数 |
| `layout_groups` | int | 权重组数（运行期定型，经 `views(slot)` 可得组型） |
| `views(slot)` | (weights, scales) | prefetch 缓存访问器（裸二元组） |
| `weight_view(g, slot)` | Tensor | views 二元组解包糖（组 g 权重视图） |
