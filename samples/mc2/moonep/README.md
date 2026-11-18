# MoonEP

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| Ascend 950PR | √ |
| Ascend 950DT | √ |
| Atlas A3 训练系列产品 | × |
| Atlas A2 训练系列产品 | × |
| Atlas 200I/500 A2 推理产品 | × |
| Atlas 推理系列产品 | × |
| Atlas 训练系列产品 | × |

MoonEP 的设备面通信基于 Ascend 950PR / 950DT 的
UbMem（UB_MEM）映射内存接口：对端 workspace 直接映射进本端地址空间，
MTE 直读直写。

## 1. MoonepBuffer：使用与对外接口

`MoonepBuffer` 把 `moonep_plan`、`prefetch_weight`、`reduce_grad`、
`dispatch`、`combine` 五个算子收在同一个对象里：构造时一次分配好
workspace、建立好通信连接、定型窗口槽位与数据类型（权重组数在首次调用时
按张量解析并冻结）；`plan()` 的输出直接作为 `prefetch_weight`、
`reduce_grad` 和 `dispatch` 的输入。

token 数据面（`dispatch` / `combine`）复用 ElasticBuffer 实现：`MoonepBuffer`
首调 `dispatch` / `combine` 时在 `ep_group` 上懒创建 ElasticBuffer（要求已安装
`cann_ops_transformer`），对外通过 `buffer.dispatch()` /
`buffer.combine()` 使用。`dispatch` 复用 `plan` 的 `vtopk`（自动接线
`topk_idx = vtopk.view(S, K)`、`num_experts = 2E`、`nmtr = S`）；`combine`
与对应 `dispatch` 通过 `handle` 配对。`experts_to_copy` 供
`prefetch_weight` / `reduce_grad` 使用，`cu_seqlens` 供分组 GEMM 使用。

### 1.1 环境要求

- `cannbotdsl` 0.8.3；CANN 9.3.0、torch / torch_npu、HCCL 与 cannbotdsl
  版本匹配。
- Ascend 950PR / 950DT，单机 ≥ 2 卡；8 卡为验收标准形态。
- 构造前完成 `dist.init_process_group(backend="hccl")`；一进程一卡，每进程
  绑定 `torch.npu.set_device(local_rank)`。

### 1.2 通信域创建

两个通信组由调用方自建后注入构造：`moonep_group`（plan/prefetch/reduce 的
通信组）与 `ep_group`（dispatch/combine 的 ElasticBuffer 组），须分属两个
独立 HCCL 通信器（通信域分离不变量，构造期自检），且均只接受
`torch.distributed.ProcessGroup`。

数据面不占通信域的 HCCL 外部 buffer（`hccl_buffer_size`）：moonep 走
MoonepContext 私有 workspace，EB 自建注册内存。该配置及其显存占用由
通信域创建方按自身需要自理（不注入即用 HCCL 缺省配置）——库不感知、
不校验，建组方式完全由调用方决定：

```python
import torch.distributed as dist

ranks = list(range(dist.get_world_size()))
moonep_group = dist.new_group(
    ranks=ranks, backend="hccl", use_local_synchronization=True,
)
ep_group = dist.new_group(
    ranks=ranks, backend="hccl", use_local_synchronization=True,
)
```

### 1.3 构造参数

```python
MoonepBuffer(
    S, H, Hp, K, E,
    *,
    num_ep_ranks=None,
    moonep_group,
    ep_group,
    weight_dtype="bf16",
    grad_dtype="bf16",
    slots=1,
)
```

| 参数名 | 输入 | 描述 | 类型 | 取值/约束 |
| :--- | :--- | :--- | :--- | :--- |
| `S` | 输入 | 每 rank token 数 | int | ≥ 1 |
| `H` | 输入 | hidden size，同时是专家 FFN 输入维 | int | ≥ 1 |
| `Hp` | 输入 | 专家 FFN 中间维，权重行宽 `W = H × Hp` | int | ≥ 1；`W` 为 64 的倍数（`reduce_grad` 尾块防护） |
| `K` | 输入 | 每 token top-k | int | ≥ 1 |
| `E` | 输入 | EP 组专家总数 | int | ≥ 1、`E <= 1024`、`E % R == 0` |
| `num_ep_ranks` | 可选输入 | EP 组 rank 数；None 取组 world size | int | 显式给出时须与组 world size 一致 |
| `moonep_group` | 输入 | `moonep_plan` / `prefetch_weight` / `reduce_grad` 通信组 | ProcessGroup | 不接受 str / None |
| `ep_group` | 输入 | `dispatch` / `combine`（ElasticBuffer）通信组 | ProcessGroup | 不能为 None |
| `weight_dtype` | 可选输入 | 权重数据类型定型值 | str 或 torch.dtype | `bf16` / `fp16` / `e4m3fn` / `e5m2` |
| `grad_dtype` | 可选输入 | 梯度数据类型定型值，缺省 `bf16` | str 或 torch.dtype | `bf16` / `fp16` / `fp32` |
| `slots` | 可选输入 | 权重窗口槽数，层间流水取 2 | int | ≥ 1 |

构造是集体调用：内含 workspace 注册、UbMem 建连与 ElasticBuffer 创建。
`moonep_group` / `ep_group` 的成员必须以相同参数、相同顺序执行；参数不一致
（形状、槽位、数据类型等）会导致容量校验或建连失败，顺序不一致会导致握手
tag 错位。

### 1.4 主要方法与属性

| 名称 | 说明 |
| :--- | :--- |
| `plan(tokens_per_expert, topk_experts_sk)` | 规划入口，返回 `dst_n` / `cu_seqlens` / `experts_to_copy` / `src_n` / `vtopk` |
| `prefetch_weight(plan, async_finish=False, *, local_weights, local_scales=None, slot=0)` | 把本 rank 权重推到各 rank 的槽位，返回 `(weights, scales)`；异步时返回 `((weights, scales), event)` |
| `reduce_grad(plan, async_finish=False, *, local_grads)` | 把副本梯度累加回 home，返回 `home_views`；异步时返回 `(home_views, event)` |
| `dispatch(plan, x, topk_weights=None, *, do_cpu_sync=True)` | 按 `plan.vtopk` 把 token 分发到各 rank，返回 `(recv_x, recv_w, handle)` |
| `combine(y, handle, *, topk_weights=None)` | 把专家输出按路由回收，返回 `(out, out_w)` |
| `views(slot=0)` | 取缓存里的 `(weights, scales)`；须先成功调用过 `prefetch_weight` |
| `weight_view(group=0, slot=0)` | 取单组权重视图 `[2*epn, Wg]`，可直接喂 GEMM |
| `barrier()` | 组内屏障；返回即全组此前流上的工作都已完成 |
| `destroy(*, release_group=False, release_eb=False)` | 幂等释放；`release_group=False` 不销毁子组（池化保活），`release_eb=False` 不销毁 EB（其 Destroy 有分钟级屏障，缺省跳过），`True` 才显式回收 |
| `rank` / `rank_size` / `epn` / `H` / `Hp` / `slots` | 运行期只读属性 |
| `weight_dtype` / `grad_dtype` / `layout_groups` | 数据类型/组数定型值；`layout_groups` 在首次算子调用后定型（此前为 None） |
| `heap_required_bytes` / `heap_bytes` / `destroyed` | 堆容量、workspace 统计与生命周期状态 |

视图和 plan 返回的张量要在 `destroy()` 之前用完；`destroy` 可重复调用。

### 1.5 典型调用流程

```python
buf = MoonepBuffer(
    S, H, Hp, K, E,
    moonep_group=moonep_group,
    ep_group=ep_group,
    weight_dtype="bf16",
    grad_dtype="bf16",
    slots=2,
)

plan = buf.plan(tpe, topk)                       # tpe [E]、topk [S,K]，均 int32
weights, scales = buf.prefetch_weight(plan, local_weights=(w13, w2))

# token 数据面：按虚拟专家 id 分发 → 专家计算 → 按路由回收
# scores: [S, K] 路由权重（fp32）
recv_x, recv_w, handle = buf.dispatch(
    plan, x, topk_weights=scores, do_cpu_sync=False,
)
# ... 分组 GEMM / FFN；gmm_out_padded 须填满 dispatch 返回的接收容量 ...
# combine 的 topk_weights 传 recv_w（与接收行对齐，[cap]），不是原始 scores
out, out_w = buf.combine(gmm_out_padded, handle, topk_weights=recv_w)

# ... 反向 ...
home_views = buf.reduce_grad(plan, local_grads=(dw13, dw2))

buf.destroy()
```

**权重形态**（2 组 / 3 组 / fp16 / FP8）：

```python
import torch

epn, W = buf.epn, H * Hp

# ① 3 组等宽：gate / up / down 分离，每组 [epn, ...]，组间同数据类型
w1 = torch.empty(epn, Hp, H, dtype=torch.bfloat16, device="npu")   # gate
w3 = torch.empty(epn, Hp, H, dtype=torch.bfloat16, device="npu")   # up
w2 = torch.empty(epn, H, Hp, dtype=torch.bfloat16, device="npu")   # down
weights, scales = buf.prefetch_weight(plan, local_weights=(w1, w3, w2))
# scales == ()；weights[g] 为 [2*epn, Wg]

# ② 2 组交织：gate/up 合并为 W13（每专家一行 2W，gate 在前 up 在后）
w13 = torch.cat([w1, w3], dim=1)                                   # [epn, Hp, 2H]
weights, scales = buf.prefetch_weight(plan, local_weights=(w13, w2))

# ③ fp16：与 bf16 同为 2 字节、不配 scale（构造期 weight_dtype="fp16"）
buf16 = MoonepBuffer(
    S, H, Hp, K, E,
    moonep_group=moonep_group, ep_group=ep_group, weight_dtype="fp16",
)
weights, scales = buf16.prefetch_weight(plan, local_weights=(w13_fp16, w2_fp16))

# ④ mxfp8：fp8 权重 + ue8m0 block scale 必须成对（构造期 weight_dtype="e4m3fn"）
buf8 = MoonepBuffer(
    S, H, Hp, K, E,
    moonep_group=moonep_group, ep_group=ep_group, weight_dtype="e4m3fn",
)
def _align32(n):
    return (n + 31) // 32 * 32

s13 = torch.empty(epn, _align32(2 * W // 32), dtype=torch.float8_e8m0fnu, device="npu")
s2 = torch.empty(epn, _align32(W // 32), dtype=torch.float8_e8m0fnu, device="npu")
w13 = torch.empty(epn, 2 * W, dtype=torch.float8_e4m3fn, device="npu")
w2 = torch.empty(epn, W, dtype=torch.float8_e4m3fn, device="npu")
weights, scales = buf8.prefetch_weight(
    plan, local_weights=(w13, w2), local_scales=(s13, s2)
)
```

以上四种形态按需选择其一（`buf16` / `buf8` 仅作示意，不必同时构造）。

首调要传权重本体（创建视图并把参数拷进 home 行）；之后可以传回视图的
home 行切片（`weights[g][:epn]`）——与视图同址，零拷贝；若传独立张量，
每次调用会同步拷贝一次 home 行。`e5m2` 把 `weight_dtype` 换成 `"e5m2"`，
契约与 `e4m3fn` 相同。梯度只有 bf16 / fp16 / fp32，没有 scale。

### 1.6 dispatch / combine（`MoonepBuffer` 方法）

token 数据面由 `MoonepBuffer` 提供两个方法，内部委托 ElasticBuffer；裸 API
只覆盖 `moonep_plan` / `prefetch_weight` / `reduce_grad`，不提供这两个方法。

#### dispatch

```python
dispatch(plan, x, topk_weights=None, *, do_cpu_sync=True)
```

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| `plan` | 输入 | `buf.plan()` 返回的 `MoonepPlan`；内部取 `plan.vtopk` 作为路由 | - | - |
| `x` | 输入 | 前向 token 载荷 / 反向输出梯度，shape `(S, H)` | BFLOAT16（token 数据类型） | ND |
| `topk_weights` | 可选输入 | 路由权重，shape `(S, K)`，内部转 fp32 | FLOAT32 | ND |
| `do_cpu_sync` | 可选输入 | True（缺省）精确分配接收区；False 按 worst-case 容量分配 | BOOL | - |
| `recv_x` | 输出 | 接收区，按专家分组，shape `(cap, H)` | 同 `x` | ND |
| `recv_w` | 输出 | 接收行对应的路由权重，shape `(cap,)`；`topk_weights=None` 时为 None | FLOAT32 | ND |
| `handle` | 输出 | 本次分发的路由结果，供对应 `combine` 使用 | - | - |

接口约束：

- `plan` 应为同一 `MoonepBuffer` 的 `plan()` 结果（实现按 `MoonepPlan` 类型校验）；`x` 形状严格为
  `(S, H)`；传 `topk_weights` 时形状严格为 `(S, K)`；张量须在 NPU 上。
- 路由和参数由 `MoonepBuffer` 自动准备好，调用方不用再传：内部用
  `plan.vtopk.view(S, K)` 作为 `topk_idx`、`num_experts = 2E`（虚拟专家
  扩展空间）、每 rank 容量 `nmtr = S`；本 rank 号和路由都在运行时从设备侧
  读取，不需要重新规划。
- `do_cpu_sync=True` 会做一次 CPU 同步，按实际到达行数精确分配 `recv_x`
  （显存最优）；`False` 免同步，`recv_x` 按 worst-case 容量分配。
- 前向和反向各自的 handle 只与同方向的 `combine` 配对，不能混用。

行为说明：

- 按虚拟专家 id（`[0, 2E)`）把本 rank 的 token 路由到各 rank 的接收区，
  接收行按专家分组；plan 均衡不变量保证每 rank 的扩展行数为 `S*K`。
- 反向形态复用同一份 plan，`x` 传输出梯度、`topk_weights=None`，此时
  `recv_w` 为 `None`，梯度按 K 份裸和由 `combine` 回收。

#### combine

```python
combine(y, handle, *, topk_weights=None)
```

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| `y` | 输入 | 专家输出，shape `(cap, H)`；`cap` 等于对应 `dispatch` 的接收容量 | 同 `recv_x` | ND |
| `handle` | 输入 | 对应方向 `dispatch` 返回的 handle | - | - |
| `topk_weights` | 可选输入 | `None` = 裸和（反向）；传 `dispatch` 返回的 `recv_w`（shape `(cap,)`）= 按权重回收（前向） | FLOAT32 | ND |
| `out` | 输出 | 回收结果，token 主序，shape `(S, H)` | 同 `y` | ND |
| `out_w` | 输出 | 回收的路由权重，shape `(S, K)`；`topk_weights=None` 时为 None | FLOAT32 | ND |

接口约束：

- `y` 须为二维、末维等于 `H`、在 NPU 上；`cap` 直接取对应 `dispatch`
  返回的 `recv_x.shape[0]`（host 侧形状，零同步可得）。
- 有效行不足容量时，调用方须把尾行零填充到 `cap`（裸和语义不受影响）；
  本方法不代做零填充，避免引入隐式 host 同步。
- `handle` 必须来自同方向的 `dispatch`；`topk_weights` 传 `None` 时
  `out_w` 也为 `None`。

行为说明：

- 按 handle 里记录的路由把专家输出回收到 token 主序：前向按路由权重加权
  并回收权重（`out_w` 应恰等于本 rank 的路由权重）；反向 `topk_weights=None`
  时按 K 份裸和，得到各 token 的梯度。

### 1.7 裸 API 用法

```python
import torch
import torch.distributed as dist
import torch_npu
from moonep_buffer import _workspace_need
from moonep_context import create_moonep_context
from moonep_plan import moonep_plan
from prefetch_weight import prefetch_weight
from reduce_grad import reduce_grad

# workspace 分配即所需：跑 prefetch_weight / reduce_grad 的裸 API 须按形状
# 精算传入（缺省只够 plan 交换区）。W = H × Hp；2 组交织时仍传基组宽 W
# （w13 的行宽是 2W，不要传 2W）。
W = H * Hp
ctx = create_moonep_context(
    group=dist.group.WORLD,
    workspace_bytes=_workspace_need(
        dist.get_world_size(), E // dist.get_world_size(), W, 1,
        torch.bfloat16, torch.bfloat16,
    ),
)
R = ctx.rank_size
epn = E // R

topk = topk_idx.to(torch.int32).npu()                            # [S, K]
tpe = torch.bincount(topk.cpu().flatten(), minlength=E).to(torch.int32).npu()
dst_n, cu_seqlens, copy_list, src_n, vtopk = moonep_plan(ctx, tpe, topk)

# 前向：prefetch_weight 推远程权重（2 组交织形态）
weights, scales = prefetch_weight(ctx, copy_list, (w13, w2), epn)

# 反向：reduce_grad 把 full-dW 累加回 home
home_views = reduce_grad(ctx, copy_list, (dw13, dw2), epn)
```

workspace 一律分配即所需：`MoonepBuffer` 构造期按形状精算注入；裸 API
`create_moonep_context` 缺省只分配 plan 交换区下界，`prefetch_weight` /
`reduce_grad` 的形状相关区域要由调用方按 `_workspace_need()` 显式传入。

`_workspace_need()` 为内部辅助函数（签名可能随版本调整）；其最后两个参数
是梯度 / 权重 dtype，须与实际使用一致（如 fp8 权重传
`torch.float8_e4m3fn`、fp32 梯度传 `torch.float32`），否则容量会偏小。

## 2. 约束说明

- **规模与形状**：`E <= 1024`、`R <= 128`（布局常量 `MAX_R`）、`E % R == 0`、
  `S >= 1`、`K >= 1`；各 rank 的本地 token 数 S 须一致（均衡契约以
  `cu_seqlens` 尾值等于本地 `N = S*K` 作为布局不变量）。
- **数据类型**：`tokens_per_expert` / `topk_experts_sk` / `experts_to_copy`
  必须为 int32；权重支持 bf16 / fp16 / e4m3 / e5m2，组间一致且运行期不可
  漂移；梯度支持 bf16 / fp16 / fp32，组间一致且等于构造定型值。
- **设备与连续性**：所有输入张量须在 NPU 上且连续；`experts_to_copy`
  形状 `[R, epn]`，空槽为 -1。
- **分组与行宽**：权重/梯度支持 2 组（交织 `(2W, W)`）或 3 组（常见等宽
  `(W, W, W)`），总宽上限 `3W`（构造期按此预留，`W = H × Hp`）；权重行数
  `epn`、梯度行数 `2*epn`；`W` 须为 64 的倍数（`reduce_grad` 尾块防护，
  `MoonepBuffer` 构造期与 `reduce_grad` 入口都会校验）。
- **槽位**：`slots >= 1`，`prefetch_weight` / `reduce_grad` / `MoonepBuffer`
  全链一致；`reduce_grad` 的 `slot` 参数只校验一致性，不影响累加落点。
- **通信域**：`moonep_group` 与 `ep_group` 仅接受 `ProcessGroup`，且应为
  两个独立通信器（调用方保证）。数据面不占通信域 HCCL 外部 buffer——
  `hccl_buffer_size` 由创建方按自身需要自理（默认/显式值均可），库不感知。
- **通信后端**：UbMem（映射内存直接寻址对端 workspace）。全部 peer 须
  位于同一可映射层（层内 FullMesh）；部分可达 / 跨层多组形态会直接报错。
  UbMem 写路径经 UB 中转（`mem_copy` 不支持 GM→GM），读路径可直接以远端
  mapped GM 为源。
- **调用一致性**：全 EP rank 须以相同参数、相同顺序调用；`plan()` 返回时
  不做同步，溢出/布局不变量延迟到下一次 moonep 调用校验，需要当场报错时
  显式调用 `check_pending_overflow(context)`。
- **生命周期**：视图与 plan 返回的张量须在 `destroy()` 之前使用。

## 3. 算子接口与约束

### 3.1 moonep_plan

```python
moonep_plan(context, tokens_per_expert, topk_experts_sk)
```

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| `context` | 输入 | MoonEP 通信上下文 | - | - |
| `tokens_per_expert` | 输入 | 本 rank 各专家 token 计数，shape `(E,)` | INT32 | ND |
| `topk_experts_sk` | 输入 | 本 rank 的 top-k 专家索引，shape `(S, K)`，取值 `[0, E)` | INT32 | ND |
| `dst_n` | 输出 | 零占位（vtopk-only 快路径不计算），shape `(N,)` | INT32 | ND |
| `cu_seqlens` | 输出 | 分组 GEMM 分段偏移，shape `(2*epn,)`，末元素等于本地 N | INT32 | ND |
| `experts_to_copy` | 输出 | 各 rank 的副本清单，shape `(R, epn)`，空槽为 -1，每行单源 | INT32 | ND |
| `src_n` | 输出 | 零占位（vtopk-only 快路径不计算），shape `(N,)` | INT32 | ND |
| `vtopk` | 输出 | 虚拟专家 id，shape `(N,)`，取值 `[0, 2E)` | INT32 | ND |

接口约束：

- `tokens_per_expert` 与 `topk_experts_sk` 都必须是 int32（`bincount` /
  `topk` 默认 int64，需显式转换）；形状分别严格为 `[E]` 与 `[S, K]`。
- `E <= 1024`、`R <= 128`、`E % R == 0`；`topk_experts_sk` 的取值须在
  `[0, E)`，且 `tokens_per_expert` 与它统计出的计数一致。
- 各 rank 的 S 必须一致；不均衡会在下一次算子的
  `check_pending_overflow` 处报错。

行为说明：

- 先跨 rank 交换各专家需求，再按 `cap = N`（每 rank 恰好承载本地行数）做
  贪心均衡：超载 rank 把尚未服务的专家 token 段迁给最空闲 rank，直到全部
  rank 达到 `cap`。
- 每个目的 rank 至多接收一个源 rank 的副本（一卡一源），因此
  `experts_to_copy` 每行都是单源的；空槽写 -1。
- 分段与虚拟 id：`cu[j]` 是前 j 个槽位的 token 累计，`vtopk[i]` =
  服务 token i 的 rank × `2*epn` + 槽位号，落在 `[0, 2E)`。
- 返回时不与 host 同步；`experts_to_copy` / `vtopk` 等输出可跨调用、跨前反向复用。

### 3.2 prefetch_weight

```python
prefetch_weight(
    context, experts_to_copy, local_weights, epn,
    *, local_scales=None, slot=0, slots=1, async_finish=False,
)
```

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| `context` | 输入 | MoonEP 通信上下文 | - | - |
| `experts_to_copy` | 输入 | `moonep_plan` 输出的副本清单，shape `(R, epn)`（连续、NPU） | INT32 | ND |
| `local_weights` | 输入 | 本 rank home 权重，2 或 3 组，每组 shape `(epn, ...)`（至少 2 维，展平为 `(epn, Wg)`） | BFLOAT16、FLOAT16、FLOAT8_E4M3FN、FLOAT8_E5M2 | ND |
| `epn` | 输入 | 每 rank 本地专家数 `E / R` | INT | - |
| `local_scales` | 可选输入 | mxfp8 的 scale 组，2 或 3 组，每组 shape `(epn, Sg)`（`Sg = ceil(Wg/32)`，32B 对齐），与权重同序等长 | FLOAT8_E8M0FNU | ND |
| `slot` | 可选输入 | 窗口槽位，缺省 0 | INT | - |
| `slots` | 可选输入 | 窗口槽数，缺省 1；层间流水取 2 | INT | - |
| `async_finish` | 可选输入 | 异步完成开关，缺省 False | BOOL | - |
| `weights` | 输出 | 权重视图元组，各组 shape `(2*epn, Wg)`（home 行 + 副本槽） | 同 `local_weights` | ND |
| `scales` | 输出 | scale 视图元组，各组 shape `(2*epn, Sg)`；未量化时为空元组 | FLOAT8_E8M0FNU | ND |
| `event` | 可选输出 | `async_finish=True` 时追加的发起流事件 | - | - |

接口约束：

- 权重只能是 2 组或 3 组（kernel 入口按组数固定）；每组至少 2 维、首维
  严格等于 `epn`、连续；组间数据类型必须一致，且落在白名单内。
- 组数与各组行宽在首次调用后定型，后续调用必须一致（否则发布区/槽区会
  重叠，直接报错）。
- mxfp8 双向契约：fp8 权重必须传 `local_scales`；传了 `local_scales`
  权重就必须是 fp8。scale 数据类型固定 `float8_e8m0fnu`，语义宽度为
  `ceil(Wg / 32)` 字节（每 32 个权重 1 个 ue8m0）；实际行宽必须为正、
  且是 32B 的倍数（不足时调用方 pad 到 32B 倍数，消费端按实际行宽解）。
- `0 <= slot < slots`；`slots` 与构造定型值一致。
- 总宽不超过构造期按 `3W` 预留的容量，超出时容量校验报错。

行为说明：

- 输出视图恒为 `[2*epn, Wg]`：前 `epn` 行是 home 行（与本 rank 权重同值），
  后 `epn` 行是副本槽，非空槽由推送写入；空槽在视图创建时填 `-1`
  （bf16/fp16/fp32），fp8 视图保留 0 初始化。
- 首调传权重本体：创建视图并把参数拷进 home 行；之后传回
  `weights[g][:epn]`（与视图同址）则跳过拷贝，传独立张量则每次同步拷一次。
- 推送按对端 rank 组织：每块负责若干 peer，先扫 `experts_to_copy` 对应行
  得到源行号（-1 为空槽），再逐行把权重组从源地址经 UB 中转写到对端槽。
- 每个 peer 的数据写完后，用 MTE3 同通道写一个本轮 epoch 信号；信号与
  数据同通道保序，对端收到信号即数据已全部落地。随后等待对端回写同轮
  信号，收尾同步就在 kernel 内完成。
- 同一进程内按 `R / epn / 各组宽度 / 各组元素字节宽 / 组数 / 窗口字节数 /
  常驻块数 / 每块负责的 peer 数` 缓存编译结果；本 rank 号不进编译键，运行时
  从上下文读取。
- `async_finish=True` 时返回发起流事件，`wait_event` 汇合后可复用缓冲区，
  便于层间流水（配 `slots=2`）。

### 3.3 reduce_grad

```python
reduce_grad(
    context, experts_to_copy, local_grads, epn,
    *, slot=0, slots=None, async_finish=False,
)
```

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| `context` | 输入 | MoonEP 通信上下文 | - | - |
| `experts_to_copy` | 输入 | `moonep_plan` 输出的副本清单，shape `(R, epn)`（连续、NPU） | INT32 | ND |
| `local_grads` | 输入 | full-dW 梯度，2 或 3 组，每组 shape `(2*epn, Wg)` 或 `(2*epn, Hp, H)`（前 epn 行 home 种子、后 epn 行副本） | BFLOAT16、FLOAT16、FLOAT32 | ND |
| `epn` | 输入 | 每 rank 本地专家数 `E / R` | INT | - |
| `slot` | 可选输入 | 窗口槽位，仅校验与 `prefetch_weight` 一致，缺省 0 | INT | - |
| `slots` | 可选输入 | 窗口槽数；None 取构造定型值 | INT（可为 None） | - |
| `async_finish` | 可选输入 | 异步完成开关，缺省 False | BOOL | - |
| `home_views` | 输出 | 原位累加结果，各组 shape `(epn, Wg)` | 同 `local_grads` | ND |
| `event` | 可选输出 | `async_finish=True` 时追加的发起流事件 | - | - |

接口约束：

- `local_grads` 是 2 或 3 组 full-dW 张量（GMM 对全量视图反传的形态）：
  前 `epn` 行是 home 种子、后 `epn` 行是副本 dW，行序与 `prefetch_weight` 视图
  一致；每组连续，组间数据类型一致。
- 梯度数据类型只能是 bf16 / fp16 / fp32，且必须等于构造期 `grad_dtype`
  定型值，运行期不允许漂移。
- 组数和各组行宽在首次调用后定型；`Wg` 须为 64 的倍数（尾块防护）。
- `slots` 为 None 时取构造定型值，显式给出时必须一致；`slot` 只做一致性
  校验，不影响累加落点。
- 时序：本算子要晚于本层权重最后一次读、早于下一层 `prefetch_weight`（单槽）。

行为说明：

- 发送侧由 host 把副本行（`local_grads` 后 `epn` 行）拷进本 rank
  workspace 的副本区 `[epn, Wg] × 组`；副本区大小恒定，不随 R 增长。
- owner 侧用一个 kernel 完成累加：头部屏障（把本 rank 的 epoch 写给全部
  peer，并等齐全部 peer 的信号）→ 各读块按 peer 轮转，从对端映射内存
  逐行远读 → UB 流水环 → 用拷贝引擎原子加直接累到本 rank 的 home 段
  （原位，不额外搬种子）：

  $$
  \mathbf{G}'_{home}[j,\ :] = \mathbf{G}_{seed}[j,\ :]
  + \sum_{r \in owners(j)} \mathbf{G}_r[j,\ :].
  $$

- 累加期不做 vf 运算：原子加按张量数据类型进行，bf16 / fp16 逐加舍入，
  fp32 为精确档。
- 头部屏障的写者和读块数都封顶在常驻核数内，并按 peer 轮转分配
  （R>64 时每块分管多个 peer）；写者全部落在首波，不依赖块派发顺序。
- kernel 入队后再追加一次组内屏障（入队形态，不做 host 同步），
  保证下一轮 host 侧副本行拷贝开始前，全组本轮累加都已结束；同步模式
  返回即本轮完成，`async_finish=True` 时由返回的发起流事件表达完成。
- 读块的行分片数与流水环槽深有缺省值（`MOONEP_ACC_SPLIT=8`、
  `MOONEP_ACC_DEPTH=3`），一般无需调整；改动会重新编译 kernel。

## 4. 验证目标

功能正确性：

- **plan**：前向端到端对 golden；`cu_seqlens` 尾值等于本地 `N`、
  `experts_to_copy` 每行单源、Count 阶段逐 chunk 计数与 host 统计一致。
- **prefetch_weight**：视图 `[2*epn, Wg]` 中 home 行等于本 rank 权重、
  命中槽等于源 rank 对应专家行、空槽保持 `-1`（fp8 视图为 0）；2 组交织 /
  3 组 / 变宽 / fp16 / mxfp8 / 多槽轮转逐字节对拍。
- **reduce_grad**：`home_views` 逐行等于「种子 + 各副本贡献」，覆盖多组、
  多 chunk、bf16 / fp16 / fp32 三档；负例（数据类型漂移、容量越界）明确报错。
- **dispatch / combine**：前向端到端对 golden（含 `out_w` 恰等于本 rank
  路由权重）；反向 `dispatch` 散梯度 + `combine` 裸和回收；`handle` 配对、
  容量填充与 `do_cpu_sync` 两种分配形态都覆盖。
- **MoonepBuffer / 端到端**：
  `[moonep_plan | prefetch_weight | dispatch | combine | reduce_grad]`
  共用 workspace 跑通，前后向不互相覆盖；异步事件语义正确。
- **R>64 静态门禁**：布局不越界、屏障写者与读块轮转覆盖全部 peer
  （不占卡）。

测试命令（每个测试文件须独立进程 / 作业运行）：

```bash
# 规划（E2E golden + count audit）
python -m torch.distributed.run --nproc_per_node=8 -m pytest \
    test/mc2/moonep/test_moonep_plan.py -q

# prefetch_weight / reduce_grad / MoonepBuffer（4 卡）
python -m torch.distributed.run --nproc_per_node=4 -m pytest \
    test/mc2/moonep/test_prefetch_weight.py -q
python -m torch.distributed.run --nproc_per_node=4 -m pytest \
    test/mc2/moonep/test_reduce_grad.py -q
python -m torch.distributed.run --nproc_per_node=4 -m pytest \
    test/mc2/moonep/test_moonep_buffer.py -q

# 端到端与 token 数据面（EB）
python -m torch.distributed.run --nproc_per_node=4 -m pytest \
    test/mc2/moonep/test_e2e.py -q
python -m torch.distributed.run --nproc_per_node=8 -m pytest \
    test/mc2/moonep/test_moonep_ep.py -q

# R>64 布局 / 块划分静态回归（不占卡）
python -m pytest test/mc2/moonep/test_moonep_large_r.py -q
```

性能基准：

```bash
python -m torch.distributed.run --nproc_per_node=4 \
    test/mc2/moonep/bench_plan.py
python -m torch.distributed.run --nproc_per_node=8 \
    test/mc2/moonep/bench_prefetch_weight.py
python -m torch.distributed.run --nproc_per_node=8 \
    test/mc2/moonep/bench_reduce_grad.py
```
