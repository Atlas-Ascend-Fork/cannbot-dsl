# MoonEP

## 产品支持情况

| 产品 | 是否支持 |
| :--- | :---: |
| <term>Ascend 950DT系列产品</term> | √ |
| <term>Ascend 950PR系列产品</term> | √ |
| <term>Atlas A3系列产品</term> | × |
| <term>Atlas A2系列产品</term> | × |
| <term>Atlas 200I/500 A2推理产品</term> | × |
| <term>Atlas推理系列产品</term> | × |
| <term>Atlas训练系列产品</term> | × |

## 功能说明

MoonEP 是面向 EP（Expert Parallelism）的通信库：用动态冗余专家让各 rank 的 token 负载完全均衡。

- 算子链路：`plan` 生成规划结果（`cu_seqlens`、`experts_to_copy`、`vtopk` 等），`prefetch_weight` 把本 rank 权重推到各 rank 槽位，`dispatch` / `combine` 完成 token 数据面收发，`reduce_grad` 把副本梯度累加回 home。
- 规划结果可跨调用、跨前反向复用；`plan` / `prefetch_weight` / `reduce_grad` 返回路径不做 host 同步。
- 全部计算为整型；固定输入下输出逐位一致（确定性）。
- `dispatch` / `combine` 由 `cann_ops_transformer` 的 ElasticBuffer 承载，其余三个算子走 MoonEP 私有 workspace。

`MoonepBuffer` 是五算子的统一封装，构造、参数与调用示例见 [MoonEPBuffer.md](MoonEPBuffer.md)。

## 参数说明

### MoonepBuffer 构造

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

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| `S` | 输入 | 每 rank token 数 | int | ND |
| `H` | 输入 | hidden size，同时是专家 FFN 输入维 | int | ND |
| `Hp` | 输入 | 专家 FFN 中间维，权重行宽 `W = H × Hp` | int | ND |
| `K` | 输入 | 每 token top-k | int | ND |
| `E` | 输入 | EP 组专家总数 | int | ND |
| `num_ep_ranks` | 可选输入 | EP 组 rank 数；None 取组 world size | int | ND |
| `moonep_group` | 输入 | `plan` / `prefetch_weight` / `reduce_grad` 通信组 | ProcessGroup | - |
| `ep_group` | 输入 | `dispatch` / `combine`（ElasticBuffer）通信组 | ProcessGroup | - |
| `weight_dtype` | 可选输入 | 权重数据类型定型值 | str 或 torch.dtype | ND |
| `grad_dtype` | 可选输入 | 梯度数据类型定型值，缺省 `bf16` | str 或 torch.dtype | ND |
| `slots` | 可选输入 | 权重窗口槽数，层间流水取 2 | int | ND |

构造是集体调用：内含 workspace 注册、UbMem 建连与 ElasticBuffer 创建；`moonep_group` / `ep_group` 的成员须以相同参数、相同顺序执行。

### moonep_plan

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

- 两入参必须为 int32；形状分别严格为 `[E]` 与 `[S, K]`；`tokens_per_expert` 与 `topk_experts_sk` 统计出的计数须一致。
- `E <= 1024`、`R <= 128`、`E % R == 0`；各 rank 的 S 须一致。
- 返回不做 host 同步；不均衡在下一个算子入口报错。

### prefetch_weight

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
| `epn` | 输入 | 每 rank 本地专家数 `E / R` | int | - |
| `local_scales` | 可选输入 | mxfp8 的 scale 组，2 或 3 组，每组 shape `(epn, Sg)`（`Sg = ceil(Wg/32)`，32B 对齐），与权重同序等长 | FLOAT8_E8M0FNU | ND |
| `slot` | 可选输入 | 窗口槽位，缺省 0 | int | - |
| `slots` | 可选输入 | 窗口槽数，缺省 1；层间流水取 2 | int | - |
| `async_finish` | 可选输入 | 异步完成开关，缺省 False | BOOL | - |
| `weights` | 输出 | 权重视图元组，各组 shape `(2*epn, Wg)`（home 行 + 副本槽；dtype 同 `local_weights`） | BFLOAT16、FLOAT16、FLOAT8_E4M3FN、FLOAT8_E5M2 | ND |
| `scales` | 输出 | scale 视图元组，各组 shape `(2*epn, Sg)`；未量化时为空元组 | FLOAT8_E8M0FNU | ND |
| `event` | 可选输出 | `async_finish=True` 时追加的发起流事件 | torch.npu.Event | - |

接口约束：

- 权重只能是 2 组或 3 组；每组至少 2 维、首维严格等于 `epn`、连续；组间数据类型一致且在白名单内。
- 组数与各组行宽在首次调用后定型，后续必须一致。
- mxfp8 双向契约：fp8 权重必须传 `local_scales`；传了 `local_scales` 权重就必须是 fp8；scale 类型固定 `float8_e8m0fnu`，行宽须为 32B 倍数。
- `0 <= slot < slots`；总宽不超过构造期按 `3W` 预留的容量；空槽填 `-1`（bf16/fp16/fp32），fp8 视图保留 0。

### reduce_grad

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
| `epn` | 输入 | 每 rank 本地专家数 `E / R` | int | - |
| `slot` | 可选输入 | 窗口槽位，仅校验与 `prefetch_weight` 一致，缺省 0 | int | - |
| `slots` | 可选输入 | 窗口槽数；None 取构造定型值 | int | - |
| `async_finish` | 可选输入 | 异步完成开关，缺省 False | BOOL | - |
| `home_views` | 输出 | 原位累加结果，各组 shape `(epn, Wg)`；dtype 同 `local_grads` | BFLOAT16、FLOAT16、FLOAT32 | ND |
| `event` | 可选输出 | `async_finish=True` 时追加的发起流事件 | torch.npu.Event | - |

接口约束：

- `local_grads` 为 2 或 3 组 full-dW：前 `epn` 行 home 种子、后 `epn` 行副本 dW，行序与 `prefetch_weight` 视图一致；组间数据类型一致且等于构造期 `grad_dtype`。
- 组数和各组行宽首次调用后定型；`Wg` 须为 64 的倍数；`slot` 只做一致性校验，不影响累加落点。
- 时序：晚于本层权重最后一次读、早于下一层 `prefetch_weight`。

### dispatch

```python
dispatch(plan, x, topk_weights=None, *, do_cpu_sync=True)
```

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| `plan` | 输入 | `buf.plan()` 返回的 `MoonepPlan`；内部取 `plan.vtopk` 作为路由 | MoonepPlan | - |
| `x` | 输入 | 前向 token 载荷 / 反向输出梯度，shape `(S, H)` | BFLOAT16 | ND |
| `topk_weights` | 可选输入 | 路由权重，shape `(S, K)`，内部转 fp32 | FLOAT32 | ND |
| `do_cpu_sync` | 可选输入 | True（缺省）精确分配接收区；False 按 worst-case 容量分配 | BOOL | - |
| `recv_x` | 输出 | 接收区，按专家分组，shape `(cap, H)`；dtype 同 `x` | BFLOAT16 | ND |
| `recv_w` | 输出 | 接收行对应的路由权重，shape `(cap,)`；`topk_weights=None` 时为 None | FLOAT32 | ND |
| `handle` | 输出 | 本次分发的路由结果，供对应 `combine` 使用 | EPHandle | - |

接口约束：

- `plan` 须为同一 `MoonepBuffer` 的 `plan()` 结果；`x` 形状严格为 `(S, H)`；`topk_weights` 形状严格为 `(S, K)`；张量须在 NPU 上。
- 内部用 `plan.vtopk.view(S, K)` 作为 `topk_idx`、`num_experts = 2E`、每 rank 容量 `nmtr = S`；rank 与路由在设备侧读取。
- 前向和反向的 handle 只与同方向 `combine` 配对。

### combine

```python
combine(y, handle, *, topk_weights=None)
```

| 参数名 | 输入/输出 | 描述 | 数据类型 | 数据格式 |
| :--- | :--- | :--- | :--- | :--- |
| `y` | 输入 | 专家输出，shape `(cap, H)`；`cap` 等于对应 `dispatch` 的接收容量；dtype 同 `recv_x` | BFLOAT16 | ND |
| `handle` | 输入 | 对应方向 `dispatch` 返回的 handle | EPHandle | - |
| `topk_weights` | 可选输入 | `None` = 裸和（反向）；传 `recv_w`（shape `(cap,)`）= 按权重回收（前向） | FLOAT32 | ND |
| `out` | 输出 | 回收结果，token 主序，shape `(S, H)`；dtype 同 `y` | BFLOAT16 | ND |
| `out_w` | 输出 | 回收的路由权重，shape `(S, K)`；`topk_weights=None` 时为 None | FLOAT32 | ND |

接口约束：

- `y` 须为二维、末维等于 `H`、在 NPU 上；有效行不足容量时须零填充到 `cap`（本方法不代做零填充）。
- `handle` 必须来自同方向 `dispatch`；`topk_weights=None` 时 `out_w` 也为 None。

### 辅助接口

| 接口 | 说明 |
| :--- | :--- |
| `views(slot=0)` | 取 `prefetch_weight` 缓存的 `(weights, scales)`；须先成功调用过 `prefetch_weight` |
| `weight_view(group=0, slot=0)` | 取单组权重视图 `[2*epn, Wg]`，可直接喂 GEMM |
| `barrier()` | 组内屏障；返回即全组此前流上的工作都已完成 |
| `destroy(*, release_group=False, release_eb=False)` | 幂等释放；缺省不销毁子组/EB，传 `True` 才显式回收 |

## 约束说明

- **环境**：构造前完成 `dist.init_process_group(backend="hccl")`；一进程一卡并绑定 `torch.npu.set_device(local_rank)`；单机 ≥ 2 卡，8 卡为验收形态。
- **通信域**：`moonep_group` 与 `ep_group` 须为两个独立 HCCL 通信器，仅接受 `ProcessGroup`；数据面不占通信域 HCCL 外部 buffer，`hccl_buffer_size` 由创建方自理。
- **规模与形状**：`E <= 1024`、`R <= 128`、`E % R == 0`、`S >= 1`、`K >= 1`；各 rank 的 S 一致。
- **数据类型**：`tokens_per_expert` / `topk_experts_sk` / `experts_to_copy` 为 int32；权重支持 bf16 / fp16 / e4m3 / e5m2，组间一致；梯度支持 bf16 / fp16 / fp32 且等于构造定型值。
- **分组与行宽**：权重/梯度支持 2 组（交织 `(2W, W)`）或 3 组（常见等宽 `(W, W, W)`），总宽上限 `3W`（`W = H × Hp`）；`W` 须为 64 的倍数。
- **槽位**：`slots >= 1`，全链一致；`reduce_grad` 的 `slot` 只校验一致性。
- **生命周期**：视图和 `plan` 返回的张量须在 `destroy()` 前用完；`destroy` 可重复调用。

## 调用说明

### 通信域创建

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

### 快速开始

```python
import torch
import torch.distributed as dist
import torch_npu
from moonep_buffer import MoonepBuffer

S, H, Hp, K, E = 4096, 7168, 2048, 8, 64
epn, W = E // dist.get_world_size(), H * Hp

buf = MoonepBuffer(
    S, H, Hp, K, E,
    moonep_group=moonep_group,
    ep_group=ep_group,
    weight_dtype="bf16",
    grad_dtype="bf16",
    slots=2,
)

plan = buf.plan(tpe, topk)                    # tpe [E]、topk [S,K]，均为 int32
weights, scales = buf.prefetch_weight(plan, local_weights=(w13, w2))

recv_x, recv_w, handle = buf.dispatch(
    plan, x, topk_weights=scores, do_cpu_sync=False,
)
out, out_w = buf.combine(gmm_out_padded, handle, topk_weights=recv_w)
home_views = buf.reduce_grad(plan, local_grads=(dw13, dw2))

buf.destroy()
```

完整的权重形态（2 组 / 3 组 / fp16 / FP8）见 [MoonEPBuffer.md](MoonEPBuffer.md)。

### 裸 API 用法

```python
from moonep_buffer import _workspace_need
from moonep_context import create_moonep_context
from moonep_plan import moonep_plan
from prefetch_weight import prefetch_weight
from reduce_grad import reduce_grad

ctx = create_moonep_context(
    workspace_bytes=_workspace_need(
        dist.get_world_size(), E // dist.get_world_size(), W, 1,
        torch.bfloat16, torch.bfloat16,
    ),
)
topk = topk_idx.to(torch.int32).npu()
tpe = torch.bincount(topk.cpu().flatten(), minlength=E).to(torch.int32).npu()
dst_n, cu_seqlens, copy_list, src_n, vtopk = moonep_plan(ctx, tpe, topk)
weights, scales = prefetch_weight(ctx, copy_list, (w13, w2), epn)
home_views = reduce_grad(ctx, copy_list, (dw13, dw2), epn)
```

`create_moonep_context` 缺省只分配 plan 交换区下界；`prefetch_weight` / `reduce_grad` 的形状相关区域由调用方按 `_workspace_need()` 传入。

### 测试与性能

每个测试文件须独立进程 / 作业运行。

```bash
# 规划（E2E golden + count audit）
python -m torch.distributed.run --nproc_per_node=8 -m pytest test/mc2/moonep/test_moonep_plan.py -q

# prefetch_weight / reduce_grad / MoonepBuffer（4 卡）
python -m torch.distributed.run --nproc_per_node=4 -m pytest test/mc2/moonep/test_prefetch_weight.py -q
python -m torch.distributed.run --nproc_per_node=4 -m pytest test/mc2/moonep/test_reduce_grad.py -q
python -m torch.distributed.run --nproc_per_node=4 -m pytest test/mc2/moonep/test_moonep_buffer.py -q

# 端到端与 token 数据面（EB）
python -m torch.distributed.run --nproc_per_node=4 -m pytest test/mc2/moonep/test_e2e.py -q
python -m torch.distributed.run --nproc_per_node=8 -m pytest test/mc2/moonep/test_moonep_ep.py -q

# R>64 布局 / 块划分静态回归（不占卡）
python -m pytest test/mc2/moonep/test_moonep_large_r.py -q
```
