# MoonEPBuffer

`MoonepBuffer` 把 `moonep_plan`、`prefetch_weight`、`reduce_grad`、
`dispatch`、`combine` 五个算子收在同一个对象里：构造时一次分配好
workspace、建立好通信连接、定型窗口槽位与数据类型（权重组数在首次调用时
按张量解析并冻结）；`plan()` 的输出直接作为 `prefetch_weight`、
`reduce_grad` 和 `dispatch` 的输入。

token 数据面（`dispatch` / `combine`）复用 ElasticBuffer 实现：
`MoonepBuffer` 首调 `dispatch` / `combine` 时在 `ep_group` 上懒创建
ElasticBuffer（要求已安装 `cann_ops_transformer`），对外通过
`buffer.dispatch()` / `buffer.combine()` 使用。`dispatch` 复用 `plan` 的
`vtopk`（自动接线 `topk_idx = vtopk.view(S, K)`、`num_experts = 2E`、
`nmtr = S`）；`combine` 与对应 `dispatch` 通过 `handle` 配对。
`experts_to_copy` 供 `prefetch_weight` / `reduce_grad` 使用，`cu_seqlens`
供分组 GEMM 使用。

## 1. 环境要求

- `cannbotdsl` 0.8.4；CANN 9.3.0、torch / torch_npu、HCCL 与 cannbotdsl
  版本匹配。
- Ascend 950PR / 950DT，单机 ≥ 2 卡；8 卡为验收标准形态。
- 构造前完成 `dist.init_process_group(backend="hccl")`；一进程一卡，每进程
  绑定 `torch.npu.set_device(local_rank)`。

## 2. 通信域创建

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

## 3. 构造参数

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

workspace 按形状精算（`_workspace_need()`）：prefetch 发布区 + 梯度副本区，
分配即所需，不做粗预留。

## 4. 主要方法与属性

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

- `plan()` 有构造期形状守卫：`tokens_per_expert` 必须为 `[E]`，
  `topk_experts_sk` 必须为 `[S, K]`，异形直接报错。
- `views()` / `weight_view()` 依赖首次 `prefetch_weight` 创建的视图，未调用
  过会报错。

### 4.1 plan

```python
plan(tokens_per_expert, topk_experts_sk) -> MoonepPlan
```

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| `tokens_per_expert` | 输入 | 本 rank 各专家 token 计数，shape `(E,)` | INT32 | ND |
| `topk_experts_sk` | 输入 | 本 rank 的 top-k 专家索引，shape `(S, K)`，取值 `[0, E)` | INT32 | ND |
| `MoonepPlan` | 输出 | 五件套：`dst_n` / `cu_seqlens` / `experts_to_copy` / `src_n` / `vtopk` | - | - |

接口约束：

- 两个入参都必须是 int32（`bincount` / `topk` 默认 int64，需显式转换）；
  形状必须分别等于构造期 `E` 与 `(S, K)`，异形直接报错。
- `tokens_per_expert` 与 `topk_experts_sk` 统计出的计数须一致；
  `topk_experts_sk` 取值须在 `[0, E)`。
- 返回路径不做 host 同步；`cu_seqlens` 尾值等于本地 `N = S*K` 是布局
  不变量，溢出/不均衡延迟到下一次算子调用校验。
- 完整语义（均衡、单源清单、虚拟 id）见 [plan.md](plan.md)。

### 4.2 prefetch_weight

```python
prefetch_weight(
    plan, async_finish=False, *,
    local_weights, local_scales=None, slot=0,
) -> (weights, scales) | ((weights, scales), event)
```

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| `plan` | 输入 | 同一 `MoonepBuffer` 的 `plan()` 结果（按 `MoonepPlan` 类型校验） | - | - |
| `async_finish` | 可选输入 | True 跳过末尾屏障的 host 同步，返回发起流事件 | BOOL | - |
| `local_weights` | 输入 | 2 或 3 组 home 权重，每组 `(epn, ...)`（至少 2 维，展平为 `(epn, Wg)`）；必选 keyword | 构造期 `weight_dtype` | ND |
| `local_scales` | 可选输入 | mxfp8 scale 组，与权重同组同序，每组 `(epn, Sg)`（`Sg` 为 32B 倍数） | FLOAT8_E8M0FNU | ND |
| `slot` | 可选输入 | 窗口槽位，缺省 0，须 `0 <= slot < slots` | INT | - |
| `weights` | 输出 | 各组 `[2*epn, Wg]` 视图（home 行 + 副本槽） | 同 `local_weights` | ND |
| `scales` | 输出 | 各组 scale 视图；未量化时为空元组 | FLOAT8_E8M0FNU | ND |
| `event` | 可选输出 | `async_finish=True` 时追加的发起流事件 | - | - |

接口约束：

- 组数与各组行宽在首次调用后定型（`layout_groups` 随此定型），后续必须
  一致；组数只能是 2 或 3，组间 dtype 一致且等于 `weight_dtype`。
- 首调传权重本体（创建视图并把参数拷进 home 行）；之后传
  `weights[g][:epn]`（与视图同址）零拷贝，传独立张量则每次同步拷一次。
- mxfp8 双向硬契约：fp8 权重必传 `local_scales`；传了 `local_scales`
  权重必须是 fp8；scale 行宽须为正且为 32B 倍数。
- 总宽不超过构造期按 `3W` 预留的容量；输出视图空槽填 `-1`
  （bf16/fp16/fp32），fp8 视图保留 0 初始化。
- 返回的视图按 `slot` 缓存，可用 `views(slot)` / `weight_view(group, slot)` 取回。
  完整语义与推送流程见 [prefetch_weight.md](prefetch_weight.md)。

### 4.3 reduce_grad

```python
reduce_grad(
    plan, async_finish=False, *, local_grads,
) -> home_views | (home_views, event)
```

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| `plan` | 输入 | 同一 `MoonepBuffer` 的 `plan()` 结果（副本清单与 prefetch 共用） | - | - |
| `async_finish` | 可选输入 | True 返回发起流事件；False（缺省）阻塞到全组归并完成 | BOOL | - |
| `local_grads` | 输入 | 2 或 3 组 full-dW 张量，每组 `(2*epn, Wg)` 或 `(2*epn, Hp, H)`：前 epn 行 home 种子、后 epn 行副本 dW；必传 keyword | 构造期 `grad_dtype` | ND |
| `home_views` | 输出 | 各组 `(epn, Wg)` 归并结果视图（与全量 dW 同存储） | 同 `local_grads` | ND |
| `event` | 可选输出 | `async_finish=True` 时追加的发起流事件 | - | - |

接口约束：

- `local_grads` 必须是 2 或 3 组的 list/tuple，不接受单张量；行序与
  `prefetch_weight` 视图一致，组间 dtype 一致且等于构造期 `grad_dtype`
  （bf16 / fp16 / fp32），运行期不可漂移。
- 组数与各组行宽在首次调用后定型；`Wg` 须为 64 的倍数。
- 副本行入槽与方法内发起的组屏障、原子直累均由本方法完成，梯度槽视图
  不对外暴露；梯度槽区单窗口，不随 `slots` 倍增。
- 时序：晚于本层权重最后一次读、先于下一层 `prefetch_weight`（单槽）。
  完整语义与调优旋钮见 [reduce_grad.md](reduce_grad.md)。

## 5. 典型调用流程

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

### 5.1 权重形态（2 组 / 3 组 / fp16 / FP8）

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

## 6. dispatch / combine

token 数据面由 `MoonepBuffer` 提供两个方法，内部委托 ElasticBuffer；裸 API
只覆盖 `moonep_plan` / `prefetch_weight` / `reduce_grad`，不提供这两个方法。

### 6.1 dispatch

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

- `plan` 应为同一 `MoonepBuffer` 的 `plan()` 结果（实现按 `MoonepPlan`
  类型校验）；`x` 形状严格为 `(S, H)`；传 `topk_weights` 时形状严格为
  `(S, K)`；张量须在 NPU 上。
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

### 6.2 combine

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

## 7. 约束说明

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
- **通信后端**：UbMem（映射内存直接寻址对端 workspace）。建连优先选择
  「全部 peer 均 UbMem 可达」的最低网络层，以单条连接承载全部 peers；
  不存在全覆盖层时，部分可达 / 跨层多组形态会直接报错。
- **调用一致性**：全 EP rank 须以相同参数、相同顺序调用；`plan()` 返回时
  不做同步，溢出/布局不变量延迟到下一次 moonep 调用校验，需要当场报错时
  显式调用 `check_pending_overflow(context)`。
- **生命周期**：视图与 plan 返回的张量须在 `destroy()` 之前使用。
